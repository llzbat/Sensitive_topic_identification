# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

UNK, PAD = '<UNK>', '<PAD>'


class FastText_Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        self.train_path = dataset + '/data/train_data.txt'  # 训练集
        self.dev_path = dataset + '/data/val_data.txt'  # 验证集
        self.test_path = dataset + '/data/test_data.txt'  # 测试集                               # 测试集
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
        self.n_vocab = len(self.vocab)  # 词表大小                                             # 词表大小，在运行时赋值
        self.num_epochs = 40  # epoch数
        self.batch_size = 256  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 256  # 隐藏层大小
        self.n_gram_vocab = 250499  # ngram 词表大小
        self.tokenizer = lambda x: [y for y in x]

    def build_dataset(self, content):

        def biGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            return (t1 * 14918087) % buckets

        def triGramHash(sequence, t, buckets):
            t1 = sequence[t - 1] if t - 1 >= 0 else 0
            t2 = sequence[t - 2] if t - 2 >= 0 else 0
            return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

        words_line = []
        token = self.tokenizer(content)
        seq_len = len(token)
        if self.pad_size:
            if len(token) < self.pad_size:
                token.extend([PAD] * (self.pad_size - len(token)))
            else:
                token = token[:self.pad_size]
                seq_len =self.pad_size
        # word to id
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))

        # fasttext ngram
        buckets = self.n_gram_vocab
        bigram = []
        trigram = []
        # ------ngram------
        for i in range(self.pad_size):
            bigram.append(biGramHash(words_line, i, buckets))
            trigram.append(triGramHash(words_line, i, buckets))
        # -----------------
        datas=[(words_line, -1, seq_len, bigram, trigram)]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram)


class FastText_Model(nn.Module):
    def __init__(self, config):
        super(FastText_Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
