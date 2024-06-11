# -*- codeing = utf-8 -*-
# @Time : 2024/4/3  1:26
# @Sofaware : PyCharm

import pickle
import random
import numpy as np

if __name__ == '__main__':

    X = pickle.load(open("./data/X_data.pkl","rb"))
    y = pickle.load(open("./data/y_data.pkl","rb"))
    word2id, id2word = pickle.load(open("./data/word2idandid2word.pkl","rb"))
    label2id, id2label = pickle.load(open("./data/label2idandid2label.pkl","rb"))
    word2vec_matrix = pickle.load(open("./data/word2vec_matrix.pkl","rb"))

    print (X.shape)
    print (y.shape)

    index = [i for i in range(len(y))]
    random.shuffle(index)

    X_new = []
    y_new = []

    for i in index:
        X_new.append(X[i])
        y_new.append(y[i])

    X_new = np.array(X_new)
    y_new = np.array(y_new)
    pickle.dump(X_new, open("./data/X_data_random.pkl","wb"))
    pickle.dump(y_new, open("./data/y_data_random.pkl","wb"))