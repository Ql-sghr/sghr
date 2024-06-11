import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tqdm import tqdm


# 加载数据
def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, encoding='utf-8')
    # 结合发明名称和摘要作为文本特征
    df['text'] = df['发明名称'] + '。' + df['摘要']
    return df['IPC主分类'], df['text']


# 文本预处理:分词和去停用词
def preprocess_text(text_series, stopwords_path='./data/cn_stopwords.txt'):
    print("Preprocessing text...")
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])
    return text_series.progress_apply(lambda x: ' '.join([word for word in jieba.cut(x) if word not in stopwords]))


# 主函数
def main():
    labels, texts = load_data('./data/alldata7000.csv')

    # 使用 tqdm 显示预处理进度
    tqdm.pandas(desc="Preprocessing progress")
    texts = preprocess_text(texts)

    print("Vectorizing text...")
    # 文本向量化
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    y = pd.factorize(labels)[0]  # 将标签转化为数字ID

    print("Splitting data into train and test sets...")
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 初始化模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # 训练模型并评估
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)

        print(f"Evaluating {name} model...")
        y_pred = model.predict(X_test)
        print(f"{name} - Precision: {precision_score(y_test, y_pred, average='macro')}")
        print(f"{name} - Recall: {recall_score(y_test, y_pred, average='macro')}")
        print(f"{name} - F1 Score: {f1_score(y_test, y_pred, average='macro')}\n")


# 执行
if __name__ == "__main__":
    main()
