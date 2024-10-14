# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class TextRCNN_Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/data/train_data.txt'  # 训练集
        self.dev_path = dataset + '/data/val_data.txt'  # 验证集
        self.test_path = dataset + '/data/test_data.txt'  # 测试集                             # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.vocab = pkl.load(open(self.vocab_path, 'rb'))
        self.n_vocab = len(self.vocab)  # 词表大小                                            # 词表大小，在运行时赋值
        self.num_epochs = 40  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 1  # lstm层数
        self.tokenizer = lambda x: [y for y in x]

    def build_dataset(self, text):
        content = text.strip()
        pad_size = 64
        words_line = []
        token = self.tokenizer(content)
        if pad_size:
            if len(token) < pad_size:
                seq_len = len(token)
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
        return torch.tensor([words_line], dtype=torch.long, device=self.device), seq_len


'''Recurrent Convolutional Neural Networks for Text Classification'''


class TextRCNN_Model(nn.Module):
    def __init__(self, config):
        super(TextRCNN_Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out.unsqueeze(0) if out.dim() == 1 else out
