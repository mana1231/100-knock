# 89
import numpy as np
from collections import Counter, OrderedDict
import pandas as pd
import random
import torch
from torch import nn
import torch.utils.data as data
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def load_csv2df(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

class BERTDataset(data.Dataset):
    def __init__(self, X, y, phase='train'):
        self.X = X['TITLE']
        self.y = y
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.phase = phase

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sentence = self.X[idx]
        sentence = str(sentence)
        sentence = " ".join(sentence.split())

        bert_normed = self.tokenizer.encode_plus(
                                sentence,
                                add_special_tokens = True,
                                max_length = 20,
                                pad_to_max_length = True,
                                truncation=True)

        ids = torch.tensor(bert_normed['input_ids'], dtype=torch.long)
        mask = torch.tensor(bert_normed['attention_mask'], dtype=torch.long)
        labels = self.y[idx]

        item = {'ids': ids, 'mask': mask, 'labels': labels}
        return item

def make_dataloader(df, y_df, phase, batch_size, shuffle=True):
    dataset = BERTDataset(df, y_df, phase)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=shuffle)
    return dataloader

class BERT(nn.Module):
    def __init__(self, drop_rate, hidden_size, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Sequential(
                                nn.Linear(768, hidden_size),
                                nn.ReLU(),
                                nn.BatchNorm1d(hidden_size),
                                nn.Linear(hidden_size, output_size)
                                )

    def forward(self, idxs, mask):
        out = self.bert(idxs, attention_mask=mask)[-1]
        out = self.fc(self.drop(out))
        return out

def train(model, dataloaders, criterion, optimizer, num_epochs, device, early_stopping=5):
    model.to(device)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0

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
            for data in tqdm(dataloaders[phase]):
                ids = data['ids'].to(device)
                mask = data['mask'].to(device)
                labels = data['labels'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids, mask)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * ids.size(0)
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
            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= early_stopping:
                        print(f'Early stopping triggered after epoch {e+1}')
                        model.load_state_dict(best_model)
                        return train_loss, train_acc, val_loss, val_acc

    model.load_state_dict(best_model)
    return train_loss, train_acc, val_loss, val_acc

def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_file = f'{main_path}/train.txt'
    valid_file = f'{main_path}/valid.txt'
    test_file = f'{main_path}/test.txt'
    train_df, valid_df, test_df = load_csv2df(train_file, valid_file, test_file)

    category_dict = {'b': 0, 't': 1, 'e':2, 'm':3}
    y_train = torch.from_numpy(train_df['CATEGORY'].map(category_dict).values)
    y_valid = torch.from_numpy(valid_df['CATEGORY'].map(category_dict).values)
    y_test = torch.from_numpy(test_df['CATEGORY'].map(category_dict).values)

    batch_size = 64
    train_loader = make_dataloader(train_df, y_train, 'train', batch_size, shuffle=True)
    val_loader = make_dataloader(valid_df, y_valid, 'val', batch_size, shuffle=False)
    test_loader = make_dataloader(train_df, y_train, 'val', batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    output_size = 4
    lr = 0.0001
    num_epochs = 10
    hidden_size = 256
    dropout = 0.2

    model = BERT(dropout, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    train_loss, train_acc, val_loss, val_acc = train(model, dataloaders, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()