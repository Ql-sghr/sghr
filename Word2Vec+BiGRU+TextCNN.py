import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Conv1D, BatchNormalization, Activation, \
    MaxPooling1D, Flatten, concatenate, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# 载入数据
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']


# 设置GPU
def set_gpu(gpu_id="0", per_process_gpu_memory_fraction=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if per_process_gpu_memory_fraction < 1.0:
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=per_process_gpu_memory_fraction * 10000)]
            )


# 构建模型
def build_model(vocab_size, embedding_matrix):
    model_input = Input(shape=(200,))
    wordembed = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=200, trainable=False)(model_input)
    sen2vec = Bidirectional(GRU(300, return_sequences=True))(wordembed)

    convs = []
    filter_sizes = [2, 3, 4, 5]
    for size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=size, activation='relu')(sen2vec)
        conv = BatchNormalization()(conv)
        pool = MaxPooling1D(pool_size=200 - size + 1, strides=1)(conv)
        pool = Flatten()(pool)
        convs.append(pool)

    conv_out = concatenate(convs, axis=-1)
    conv_out = Dropout(0.5)(conv_out)
    model_output = Dense(7, activation='softmax')(conv_out)
    model = Model(inputs=model_input, outputs=model_output)
    return model


# 主函数
def main():
    X, y, word2vec_metrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    set_gpu()

    model = build_model(len(word2vec_metrix), word2vec_metrix)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint('./model/Word2VecBiGRUTextCNN.keras', monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    model.fit(X_train, y_train, batch_size=200, epochs=15, validation_data=(X_val, y_val), callbacks=[checkpoint])

    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print(f'macro_precision:\t{precision_score(y_true, y_pred, average="macro")}')
    print(f'macro_recall:\t\t{recall_score(y_true, y_pred, average="macro")}')
    print(f'macro_f1:\t\t{f1_score(y_true, y_pred, average="macro")}')


if __name__ == "__main__":
    main()
