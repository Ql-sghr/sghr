import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# 加载 BERT 模型和 tokenizer
model = BertModel.from_pretrained('D:\\Bert\\bert-wwm')
tokenizer = BertTokenizer.from_pretrained('D:\\Bert\\bert-wwm')

# 读取 CSV 文件
data = pd.read_csv('E:\\third work\\数据处理\\output_modified.csv')

# 初始化一个空的列表来存储所有文本的嵌入表示
embeddings = []

# 循环遍历 CSV 文件中的每一行文本
for text in data['merged_column']:  # 将 'column_name' 替换为实际的列名
    # 使用 tokenizer 对文本进行标记化并添加特殊标记
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

    # 使用 BERT 模型获取文本的嵌入表示
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
    # 获取 [CLS] 标记对应的池化输出（句子级表示）
    #pooler_output = outputs.pooler_output
    cls_token_embeddings = last_hidden_states[:, 0, :]
    # 将嵌入表示添加到列表中
    embeddings.append(cls_token_embeddings.numpy())

# 将嵌入表示列表转换为 numpy 数组
embeddings = np.array(embeddings)

# 保存嵌入表示为 .npy 文件
np.save('E:\\third work\\\数据处理\\cls_token.npy', embeddings)
