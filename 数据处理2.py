# -*- codeing = utf-8 -*-
# @Sofaware : PyCharm

import pandas as pd
import pickle
import jieba
import numpy as np
from keras_preprocessing.sequence import pad_sequences


# 加载停用词
def getstopword(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as stopwordfile:
        w = {line.strip() for line in stopwordfile}
    return w


stopwordset = getstopword('./data/cn_stopwords.txt')


# 分词并去除停用词
def cutWords(sentence):
    word_list = jieba.cut(sentence)
    tempX = ' '.join([i for i in word_list if i not in stopwordset])
    return tempX.strip()


# NaN值处理
def DropNan(sentence):
    if pd.isnull(sentence):
        return ''
    return sentence

# 加载CSV文件并合并，同时添加标签列
def load_and_merge_csv(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path, encoding='utf-8')
        # 提取CSV文件名作为标签，这里假设路径格式为'./data/LabelName.csv'
        label_name = path.split('/')[-1].split('.')[0]  # 从文件路径中提取标签名称
        df['label'] = label_name  # 将文件名（不带扩展名）作为标签添加到每一行
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


# 主函数
def main():
    # CSV文件路径
    file_paths = ['./data/G09G.csv', './data/G10L.csv', './data/G11B.csv', './data/G11C.csv', './data/G16B.csv',
                  './data/G16C.csv', './data/G16H.csv']
    df_files = load_and_merge_csv(file_paths)

    # 文本预处理
    df_files['all_sentence'] = df_files['发明名称'] + df_files['摘要']  # 根据实际列名合并文本数据，合并名称和摘要
    df_files['all_sentence'] = df_files['all_sentence'].apply(DropNan)  #这个函数的作用是删除包含 NaN（缺失值）的行。
    df_files['all_sentence'] = df_files['all_sentence'].apply(lambda x: x.replace('\n', ' '))  #为了去除换行符，使文本数据更易于处理。
    df_files['all_sentence'] = df_files['all_sentence'].apply(cutWords) #对文本数据进行分词处理，用于将文本数据分割成单词或词组  就是以空格分隔

    # 单词到ID的映射
    word2id, id2word = pickle.load(open("./data/word2idandid2word.pkl", "rb"))  #通过使用 pickle 库，从文件中加载了之前保存的字典数据，其中包含了单词到单词ID以及单词ID到单词的映射关系。word2id 是单词到单词ID的映射字典，id2word 是单词ID到单词的映射字典。
    df_files['all_sentence'] = df_files['all_sentence'].apply(
        lambda sentence: [word2id.get(word, 1) for word in sentence.split(' ')])
    #对数据框中的每个句子（以空格分隔的字符串）进行处理。使用了一个匿名函数，首先将句子拆分为单词列表，然后对每个单词进行转换为对应的单词ID。
    # 序列填充 这是一个 Keras 的序列填充函数，用于确保输入的序列具有相同的长度
    X_all = pad_sequences(df_files['all_sentence'].values, maxlen=200, padding='pre', truncating='post')
    print(X_all.shape)

    # 提取标签并映射到数值ID
    label2id = {label: idx for idx, label in enumerate(df_files['label'].unique())}
    id2label = {idx: label for label, idx in label2id.items()}

    # 将标签列转换为数值ID
    df_files['label_id'] = df_files['label'].apply(lambda x: label2id[x])
    # 这里使用get_dummies进行one-hot编码
    y_onehot = pd.get_dummies(df_files['label_id']).values

    # 保存数据到pickle文件
    pickle.dump(X_all, open("./data/X_data.pkl", "wb"))
    pickle.dump(y_onehot, open("./data/y_data.pkl", "wb"))
    # 也保存标签映射，以便后续使用
    pickle.dump([label2id, id2label], open("./data/label2idandid2label.pkl", "wb"))


if __name__ == "__main__":
    main()