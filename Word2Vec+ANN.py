import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle

# 加载数据
def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']

# 配置环境
def configure_environment(gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# 构建模型
def build_model(word2vec_metrix, sent_maxlen=200):
    model_input = Input(shape=(sent_maxlen,))
    wordembed = Embedding(len(word2vec_metrix), 300, weights=[word2vec_metrix], input_length=sent_maxlen, trainable=False)(model_input)
    wordembed = Flatten()(wordembed)
    wordembed = Dense(600, activation='relu')(wordembed)
    conv_out = Dropout(0.5)(wordembed)
    model_output = Dense(7, activation='softmax')(conv_out)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

def main():
    X, y, word2vec_metrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    configure_environment()

    model = build_model(word2vec_metrix)
    model.summary()

    checkpoint = ModelCheckpoint('./model/Word2VecANN.keras', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    model.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpoint])

    y_pred = model.predict(X_test)
    y_test_max = [np.argmax(i) for i in y_test]
    y_pred_max = [np.argmax(i) for i in y_pred]

    print('macro_precision:\t', metrics.precision_score(y_test_max, y_pred_max, average='macro'))
    print('macro_recall:\t\t', metrics.recall_score(y_test_max, y_pred_max, average='macro'))
    print('macro_f1:\t\t', metrics.f1_score(y_test_max, y_pred_max, average='macro'))

if __name__ == "__main__":
    main()
