import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], data["word2vec_metrix"]


def build_model(word2vec_matrix, sent_maxlen=200):
    inputs = Input(shape=(sent_maxlen,))
    word_embedding = Embedding(len(word2vec_matrix), 300, weights=[word2vec_matrix], input_length=200, trainable=False)(
        inputs)

    conv_layers = []
    for filter_size in [2, 3, 4, 5]:
        conv = Conv1D(filters=300, kernel_size=filter_size)(word_embedding)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = Dense(units=300, activation='relu')(conv)
        pool = MaxPooling1D(200 - filter_size + 1, 1)(conv)
        conv_layers.append(Flatten()(pool))

    conv_output = concatenate(conv_layers, axis=1)
    conv_output = Dropout(0.5)(conv_output)
    outputs = Dense(7, activation='softmax')(conv_output)

    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    X, y, word2vec_matrix = load_data('./data/alldata7000_new.pkl')

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    model = build_model(word2vec_matrix)
    model.summary()

    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    checkpoint_cb = ModelCheckpoint('./model/Word2VecTextCNN.keras', monitor='val_categorical_accuracy', verbose=1,
                                    save_best_only=True, mode='max')

    model.fit(X_train, y_train, batch_size=200, epochs=15, validation_data=(X_val, y_val), callbacks=[checkpoint_cb])

    model.save('./model/Word2VecTextCNN.keras')

    y_test = [np.argmax(i) for i in y_test]
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in y_pred]

    print('macro_precision: ', precision_score(y_test, y_pred, average='macro'))
    print('macro_recall: ', recall_score(y_test, y_pred, average='macro'))
    print('macro_f1: ', f1_score(y_test, y_pred, average='macro'))
