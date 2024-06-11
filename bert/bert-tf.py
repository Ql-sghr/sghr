import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# 注意：这里假设chinese_rbt6_L-6_H-768_A-12模型在transformers库中有对应的TensorFlow实现
# 如果没有，你可能需要寻找其他方式加载模型，比如通过tensorflow_hub或自己实现

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('E:\\third work\chinese_bert_wwm_ext_L-12_H-768_A-12')

# 加载预训练模型
model = TFBertModel.from_pretrained('E:\\third work\chinese_bert_wwm_ext_L-12_H-768_A-12')  # from_pt=True 如果模型是用PyTorch训练的

# 假设你有一段文本
text = "你的文本数据"

# 使用tokenizer对文本进行编码
inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)

# 获取输入ID、类型ID和注意力掩码（如果有的话）
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 使用模型获取嵌入
outputs = model(input_ids, attention_mask=attention_mask)

# 输出的last_hidden_state就是文本数据的预训练嵌入表示
last_hidden_states = outputs.last_hidden_state

# last_hidden_states的形状通常是 [batch_size, sequence_length, hidden_size]
# 在这个例子中，batch_size=1（因为我们只有一个文本），sequence_length是文本token的数量，hidden_size是768

# 如果你想获取句子级别的嵌入（比如[CLS] token的嵌入），你可以这样做：
sentence_embedding = tf.squeeze(last_hidden_states[0, 0, :], axis=0)

# sentence_embedding现在是一个形状为[hidden_size]的张量，它代表了整个句子的嵌入