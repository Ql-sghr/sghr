import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Softmax, Lambda, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def build_and_train_model(X_train, y_train, X_val, y_val, word2vec_metrix, sent_maxlen=200):
    with tf.device('/gpu:0'):
        model_input = Input(shape=(sent_maxlen,))
        wordembed = Embedding(len(word2vec_metrix), 300, weights=[word2vec_metrix], input_length=sent_maxlen,
                              trainable=False)(model_input)
        attention_pre = Dense(300)(wordembed)
        attention_probs = Softmax()(attention_pre)
        attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, wordembed])
        attention_mul = Dropout(0.5)(attention_mul)
        attention_mul = Flatten()(attention_mul)
        conv_out = Dropout(0.5)(attention_mul)
        model_output = Dense(7, activation='softmax')(conv_out)
        model = Model(inputs=model_input, outputs=model_output)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        checkpoint = ModelCheckpoint('./model/Word2VecATT.keras', monitor='val_categorical_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        history = model.fit(X_train, y_train, epochs=70, batch_size=200, validation_data=(X_val, y_val),
                            callbacks=[checkpoint, early_stop])
        model.save('./model/Word2VecATT.keras')
    return model


def main():
    X, y, word2vec_metrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    set_gpu()
    model = build_and_train_model(X_train, y_train, X_val, y_val, word2vec_metrix)

    y_pred = model.predict(X_test)
    y_test_max = [np.argmax(i) for i in y_test]
    y_pred_max = [np.argmax(i) for i in y_pred]

    cm = confusion_matrix(y_test_max, y_pred_max)
    sns.heatmap(cm, annot=True)

    precision = metrics.precision_score(y_test_max, y_pred_max, average='macro')
    recall = metrics.recall_score(y_test_max, y_pred_max, average='macro')
    f1 = metrics.f1_score(y_test_max, y_pred_max, average='macro')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')

    plt.show()


if __name__ == "__main__":
    main()
