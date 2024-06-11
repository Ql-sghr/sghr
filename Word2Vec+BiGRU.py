import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data['word2vec_metrix']

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def create_model(vocab_size, embedding_matrix):
    model_input = tf.keras.Input(shape=(None,))
    x = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                  output_dim=300, 
                                  weights=[embedding_matrix], 
                                  trainable=False)(model_input)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(300, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    model_output = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(model_input, model_output)
    return model

def main():
    configure_gpu()
    X, y, word2vec_matrix = load_data('./data/alldata7000_new.pkl')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = create_model(len(word2vec_matrix), word2vec_matrix)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./model/Word2VecBiGRU.keras', monitor='val_categorical_accuracy',
                                                    verbose=1, save_best_only=True, mode='max')



    model.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpoint])

    model.save('./model/Word2VecBiGRU_final.keras')
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(f'macro_precision: {precision_score(y_test, y_pred, average="macro")}')
    print(f'macro_recall: {recall_score(y_test, y_pred, average="macro")}')
    print(f'macro_f1: {f1_score(y_test, y_pred, average="macro")}')

if __name__ == "__main__":
    main()
