import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class bert_Config:
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train_data.txt'  # 训练集
        self.dev_path = dataset + '/data/val_data.txt'  # 验证集
        self.test_path = dataset + '/data/test_data.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单                               # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 6  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 70 # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

    def build_dataset(self, text):
        lin = text.strip()
        pad_size = 64
        token = self.tokenizer.tokenize(lin)
        token = ['[CLS]'] + token
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = []
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device), torch.tensor([mask], device=self.device)


class bert_Model(nn.Module):

    def __init__(self, config):
        super(bert_Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
