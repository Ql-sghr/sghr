import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# 假设你有一个文本数据集，每个样本是一个文本字符串，标签是对应的类别索引
texts = ["这是一个非正面评论", "这是一个灌灌灌灌负面评论","这是一个正公共广告栏面评论", "这是一个负面公共广告栏评论","这是一个正面汪汪汪评论", "这是一个负面将计就计评论"]  # 你的文本数据
labels = [1, 1,0, 1,0, 1]  # 对应的标签，0代表正面，1代表负面等

# 1. 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('D:\\Bert\\bert-china')  # 使用中文BERT
model_bert = BertModel.from_pretrained('D:\\Bert\\bert-china')

# 2. 准备数据
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 3. 划分训练集和验证集（可选）
input_ids_train, input_ids_val, attention_mask_train, attention_mask_val, labels_train, labels_val = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)

# 4. 创建DataLoader
train_data = TensorDataset(input_ids_train, attention_mask_train, torch.tensor(labels_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=16)


# 5. 定义你的分类模型（这里以BERT + 一个简单的分类层为例）
class BertForClassification(nn.Module):
    def __init__(self, n_classes):
        super(BertForClassification, self).__init__()
        self.bert = BertModel.from_pretrained('D:\\Bert\\bert-china')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        print(input_ids)
        print(attention_mask)
        outputs= self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #last_hidden_states = outputs.last_hidden_state 可以选择BERT模型的最后一层隐藏状态作为文本的嵌入表示。
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

    # 初始化你的分类模型


n_classes = len(set(labels))  # 假设你的标签是连续的整数，从0开始
model = BertForClassification(n_classes)

# 6. 定义损失函数和优化器
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# 7. 训练模型（这里只是示例性的伪代码，你需要添加更多的逻辑，如验证、保存模型等）
for epoch in range(3):  # 假设我们训练3个epoch
    model.train()
    for batch in train_loader:
        b_input_ids = batch[0]
        b_attention_mask = batch[1]
        b_labels = batch[2]
        print(b_input_ids)
        print(b_attention_mask)
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        loss = loss_fn(outputs, b_labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #用于裁剪模型参数的梯度，以避免在训练过程中梯度爆炸的问题。
        optimizer.step()
        optimizer.zero_grad()

