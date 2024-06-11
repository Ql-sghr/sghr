import pandas as pd
import pickle
import jieba
import numpy as np
from gensim.models import Word2Vec

# https://github.com/newzhoujian/LCASPatentClassification

# 加载数据
def load_data(filepath):
    return pd.read_csv(filepath, encoding='utf-8')


# 加载停用词
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = {line.strip() for line in f}
    return stopwords


# 分词并移除停用词
def preprocess_text(text, stopwords):
    words = jieba.cut(text, cut_all=False)
    filtered_words = ' '.join([word for word in words if word not in stopwords])
    return filtered_words


# 创建词汇映射
def create_vocabulary(all_texts):
    word2id = {'padding': 0, 'unknown': 1}
    id2word = {0: 'padding', 1: 'unknown'}
    index = 2
    for text in all_texts:
        for word in text.split():
            if word not in word2id:
                word2id[word] = index
                id2word[index] = word
                index += 1
    return word2id, id2word


# 创建词向量矩阵
def create_word2vec_matrix(word2id, word2vec_model_path):
    model = Word2Vec.load(word2vec_model_path)
    matrix_size = len(word2id)
    word2vec_matrix = np.zeros((matrix_size, model.vector_size))
    for word, idx in word2id.items():
        if word in model.wv:
            word2vec_matrix[idx] = model.wv[word]
    return word2vec_matrix


def main():
    # 配置路径
    data_paths = ['./data/G09G.csv', './data/G10L.csv', './data/G11B.csv', './data/G11C.csv', './data/G16B.csv', './data/G16C.csv', './data/G16H.csv']
    stopwords_path = './data/cn_stopwords.txt'
    word2vec_model_path = './model/300features_40minwords_10context.model'
    word2id_path = "./data/word2idandid2word.pkl"
    word2vec_matrix_path = "./data/word2vec_matrix.pkl"

    # 初始化空的DataFrame
    df_all = pd.DataFrame()

    # 加载停用词
    stopwords = load_stopwords(stopwords_path)

    for data_path in data_paths:
        # 加载数据
        df = load_data(data_path)

        # 数据预处理
        df['all_sentence'] = df['IPC主分类'].fillna('') + df['发明名称'].fillna('') + df['摘要'].fillna('')
        print(df['all_sentence'])
        df['all_sentence'] = df['all_sentence'].apply(lambda x: x.replace('\n', ' '))
        df['all_sentence'] = df['all_sentence'].apply(lambda x: preprocess_text(x, stopwords))

        # 合并DataFrame
        df_all = pd.concat([df_all, df])

    # 创建词汇映射和词向量矩阵
    word2id, id2word = create_vocabulary(df_all['all_sentence'].values)
    word2vec_matrix = create_word2vec_matrix(word2id, word2vec_model_path)

    # 保存结果   #得到 wordid 和 word2vec_matrix
    with open(word2id_path, "wb") as f:
        pickle.dump([word2id, id2word], f)
    with open(word2vec_matrix_path, "wb") as f:
        pickle.dump(word2vec_matrix, f)

if __name__ == "__main__":
    main()
