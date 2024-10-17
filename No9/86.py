# 86
from gensim.models import KeyedVectors
import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import random
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm
import torch.nn.functional as F

def pretrained_vocab(word2id, vocab_size, emb_size):
    model = KeyedVectors.load_word2vec_format(f'{main_path}/GoogleNews-vectors-negative300.bin.gz', binary=True)

    weights = np.zeros((vocab_size, emb_size))
    words_in_pretrained = 0
    for i, word in enumerate(word2id.keys()):
        try:
            weights[i] = model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.1, size=(emb_size,))

    weights = torch.from_numpy(weights.astype((np.float32)))
    return weights

def load_csv2df(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def make_word2id(train_df):
    words = []
    for text in train_df['TITLE']:
        for word in text.rstrip().split():
            words.append(word)

    c = Counter(words)
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    word2id = OrderedDict({token: i for i, token in enumerate(special_tokens)})
    id = 4
    for char, cnt in c.most_common():
        if cnt > 1:
            word2id[char] = id
            id += 1
    return word2id

def text2id(text, word2id):
    words = text.rstrip().split()
    return [word2id.get(word, word2id['<UNK>']) for word in words]

class MyDataset(data.Dataset):
    def __init__(self, X, y, vocab, phase='train'):
        self.X = X['TITLE']
        self.y = y
        self.phase = phase
        self.vocab = vocab

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inputs = torch.tensor(text2id(self.X[idx], self.vocab))
        return inputs, self.y[idx]

def collate_fn(batch):
    sequences = [x[0] for x in batch]
    labels = torch.LongTensor([x[1] for x in batch])
    x = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return x, labels

def make_dataloader(df, y_df, vocab, phase, batch_size):
    dataset = MyDataset(df, y_df, vocab, phase)
    dataloader = data.DataLoader(dataset, batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights=3, stride=1, padding=1, num_layers=1, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(out_channels, output_size)

    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)
        act = F.relu(conv.squeeze(3))
        max_pool = F.max_pool1d(act, act.size()[2])
        outs = self.out(self.drop(max_pool.squeeze(2)))
        return outs

def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_file = f'{main_path}/train.txt'
    valid_file = f'{main_path}/valid.txt'
    test_file = f'{main_path}/test.txt'
    train_df, valid_df, test_df = load_csv2df(train_file, valid_file, test_file)
    word2id = make_word2id(train_df)
    vocab_size = len(set(word2id.values()))

    emb_size = 300
    PAD = word2id['<PAD>']
    output_size = 4
    out_channels = 100

    emb_weights = pretrained_vocab(word2id, vocab_size, emb_size)
    model = CNN(vocab_size, emb_size, PAD, output_size, out_channels, emb_weights=emb_weights)

    ex = train_df.at[0, 'TITLE']
    x_sample = torch.tensor([text2id(ex, word2id)], dtype=torch.int64)
    out = nn.Softmax(dim=-1)
    print(x_sample)
    print(x_sample.size())
    print(out(model(x_sample)))

if __name__ == '__main__':
    main()