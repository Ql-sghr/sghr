# -*- codeing = utf-8 -*-
# @Time : 2024/4/3  1:44
# @Sofaware : PyCharm

import pickle

# 假设你的数据已经被保存在相应的文件中
x_data_file = './data/X_data.pkl'
y_data_file = './data/y_data.pkl'
word2vec_file = './data/word2vec_matrix.pkl'

# 加载数据
with open(x_data_file, 'rb') as f:
    X_data = pickle.load(f)
with open(y_data_file, 'rb') as f:
    y_data = pickle.load(f)
with open(word2vec_file, 'rb') as f:
    word2vec_metrix = pickle.load(f)

# 整合数据,存储为.pkl文件
alldata = {
    'X': X_data,
    'y': y_data,
    'word2vec_metrix': word2vec_metrix
}

# 保存整合后的数据到新的Pickle文件
with open('./data/alldata7000_new.pkl', 'wb') as f:
    pickle.dump(alldata, f)
