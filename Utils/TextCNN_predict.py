# coding: UTF-8
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class TextCNN_Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train_data.txt'  # 训练集
        self.dev_path = dataset + '/data/val_data.txt'  # 验证集
        self.test_path = dataset + '/data/test_data.txt'  # 测试集                                   # 测试集
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
        self.n_vocab = len(self.vocab)  # 词表大小
        self.num_epochs = 40  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256
        # 卷积核数量(channels数)
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


'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN_Model(nn.Module):
    def __init__(self, config):
        super(TextCNN_Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
