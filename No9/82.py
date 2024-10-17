# 82
from collections import Counter, OrderedDict
import pandas as pd
import random
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

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

def make_dataloader(df, y_df, vocab, phase, batch_size):
    dataset = MyDataset(df, y_df, vocab, phase)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)
    return dataloader

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

def train(model, dataloaders, criterion, optimizer, num_epochs):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for e in range(num_epochs):
        print('--------------------------------------------')
        print(f'Epoch {e + 1} / {num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
    return train_loss, train_acc, val_loss, val_acc

def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    train_file = f'{main_path}/train.txt'
    valid_file = f'{main_path}/valid.txt'
    test_file = f'{main_path}/test.txt'
    train_df, valid_df, test_df = load_csv2df(train_file, valid_file, test_file)
    word2id = make_word2id(train_df)
    vocab_size = len(set(word2id.values()))

    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = torch.from_numpy(train_df['CATEGORY'].map(category_dict).values)
    y_valid = torch.from_numpy(valid_df['CATEGORY'].map(category_dict).values)
    y_test = torch.from_numpy(test_df['CATEGORY'].map(category_dict).values)

    batch_size = 1
    train_loader = make_dataloader(train_df, y_train, word2id, 'train', batch_size)
    val_loader = make_dataloader(valid_df, y_valid, word2id, 'val', batch_size)
    test_loader = make_dataloader(train_df, y_train, word2id, 'val', batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    emb_size = 300
    PAD = word2id['<PAD>']
    output_size = 4
    hidden_size = 50
    lr = 0.1
    num_epochs = 10

    model = RNN(vocab_size, emb_size, PAD, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train_loss, train_acc, val_loss, val_acc = train(model, dataloaders, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()