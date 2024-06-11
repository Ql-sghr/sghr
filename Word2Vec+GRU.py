import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

# 加载数据
def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']

# 设置GPU
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU内存增长，避免占满全部内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# 构建模型
def build_model(word2vec_metrix, input_length=200):
    model_input = Input(shape=(input_length,))
    wordembed = Embedding(len(word2vec_metrix), 300, weights=[word2vec_metrix], input_length=input_length, trainable=False)(model_input)
    sen2vec = GRU(300, return_sequences=False)(wordembed)
    conv_out = Dropout(0.5)(sen2vec)
    model_output = Dense(7, activation='softmax')(conv_out)
    model = Model(inputs=model_input, outputs=model_output)
    return model

# 主函数
def main():
    configure_gpu()
    X, y, word2vec_metrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = build_model(word2vec_metrix)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint('./model/Word2VecGRU.keras', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred_max = np.argmax(y_pred, axis=1)

    print('macro_precision:\t', precision_score(y_true, y_pred_max, average='macro'))
    print('macro_recall:\t\t', recall_score(y_true, y_pred_max, average='macro'))
    print('macro_f1:\t\t', f1_score(y_true, y_pred_max, average='macro'))

if __name__ == "__main__":
    main()
