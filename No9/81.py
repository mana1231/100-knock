# 81
from collections import Counter, OrderedDict
import pandas as pd
import random
import torch
from torch import nn
import torch.utils.data as data

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
    return [word2id.get(word) for word in words]

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.rnn(x, h0)
        x = x[:, -1, :]
        y = self.out(x)
        return y
    
def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    vocabfile = f'{main_path}/train.txt'
    df = pd.read_csv(vocabfile, encoding='utf-8', sep='\t')
    word2id = make_word2id(df)
    vocab_size = len(set(word2id.values()))
    emb_size = 300
    PAD = word2id['<PAD>']
    output_size = 4
    hidden_size = 50

    model = RNN(vocab_size, emb_size, PAD, hidden_size, output_size)
    ex = df.at[0, 'TITLE']
    x_sample = torch.tensor(text2id(ex), dtype=torch.int64)
    print(x_sample.size())
    out = nn.Softmax(dim=-1)
    print(out(model(x_sample)))

if __name__ == '__main__':
    main()