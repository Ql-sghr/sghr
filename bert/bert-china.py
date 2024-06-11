from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('D:\\Bert\\bert-wwm')
model = BertModel.from_pretrained('D:\\Bert\\bert-wwm')
# 假设你有一段中文文本
text1 = "我喜欢看电影"
text = "我喜欢看电影,我爱看小马宝莉"
# 使用分词器对文本进行编码
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 获取输入ID、注意力掩码等
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 将输入传递给模型以获取嵌入表示
# 注意：BERT的嵌入是包括token嵌入、位置嵌入和segment嵌入的加和
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# outputs是一个包含各种嵌入的元组
# 最后一层的隐藏状态（即嵌入表示）在outputs.last_hidden_state中
last_hidden_states = outputs.last_hidden_state  #最后一层隐藏状态:[batch_size, sequence_length, hidden_size]

# 提取第一个句子的嵌入（对于单句输入，它通常就是整个句子的嵌入）
# 注意：BERT会为[CLS]标记生成一个特殊的嵌入，通常用于分类任务
sentence_embedding = last_hidden_states[0][0, :]  # 假设我们只对[CLS]标记的嵌入感兴趣
#选择[CLS]标记的嵌入作为文本分类任务的嵌入表示向量的维度是固定的，这一维度与BERT模型中Token嵌入的维度相同。
#具体来说，对于BERT的基础模型（如bert-base-uncased、bert-base-chinese等），这个维度通常是768。
#对于BERT的基础模型，这个维度通常是768；对于BERT的大型模型，这个维度通常是1024。
print("11111111111111",sentence_embedding)        #[CLS] token被放置在输入序列的第一个位置，用于表示整个序列的概括信息或上下文。
pooler_output = outputs.pooler_output           #pooler就是将[CLS]这个token再过一下全连接层+Tanh激活函数，作为该句子的特征向量。
print('---pooler_output: ', pooler_output)  #pooler_output可以理解成该句子语义的特征向量表示。
print(last_hidden_states[:, 0])  #和sentence_embedding一样
print(last_hidden_states[:, 0, :])