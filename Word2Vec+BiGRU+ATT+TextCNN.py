import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import os


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']


def configure_gpu(gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)


def build_model(word2vec_metrix, sent_maxlen=200):
    model_input = tf.keras.layers.Input(shape=(sent_maxlen,))

    wordembed = tf.keras.layers.Embedding(len(word2vec_metrix), 300, weights=[word2vec_metrix],
                                          input_length=sent_maxlen, trainable=False)(model_input)
    sen2vec = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(300, return_sequences=True))(wordembed)

    attention_pre = tf.keras.layers.Dense(600)(sen2vec)
    attention_probs = tf.keras.layers.Softmax()(attention_pre)
    attention_mul = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([attention_probs, sen2vec])
    attention_mul = tf.keras.layers.Dropout(0.5)(attention_mul)

    convs = []
    for i in [7, 8, 9]:
        conv = tf.keras.layers.Conv1D(312, i, activation='relu')(attention_mul)
        conv = tf.keras.layers.BatchNormalization()(conv)
        pool = tf.keras.layers.MaxPooling1D(200 - i + 1)(conv)
        convs.append(tf.keras.layers.Flatten()(pool))

    conv_out = tf.keras.layers.concatenate(convs)
    conv_out = tf.keras.layers.Dropout(0.5)(conv_out)

    model_output = tf.keras.layers.Dense(7, activation='softmax')(conv_out)
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('./model/Word2VecBiGRUATTTextCNN.keras', save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val),
                        callbacks=[checkpoint_cb, early_stopping_cb], batch_size=200)

    model.save('./model/FinalModel.keras')
    y_pred = model.predict(X_test)
    y_pred_max = np.argmax(y_pred, axis=1)

    return y_test, y_pred_max


def main():
    configure_gpu()
    X, y, word2vec_metrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = build_model(word2vec_metrix)
    model.summary()

    y_test, y_pred_max = train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, np.argmax(y_test, axis=1))

    print('macro_precision:\t', precision_score(y_test, y_pred_max, average='macro'))
    print('macro_recall:\t\t', recall_score(y_test, y_pred_max, average='macro'))
    print('macro_f1:\t\t', f1_score(y_test, y_pred_max, average='macro'))


if __name__ == "__main__":
    main()
