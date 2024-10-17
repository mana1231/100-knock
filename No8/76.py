import torch
from torch import nn
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class SingleNet(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.layer = nn.Linear(insize, outsize)

    def forward(self, x):
        outs = self.layer(x)
        return outs

def load_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

class MyDataset(Dataset):
    def __init__(self, x, y, use='train'):
        self.x = x
        self.y = y
        self.use = use

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def make_dataloaders(x_train, y_train, x_valid, y_valid, x_test, y_test, batchs=1):
    train_ds = MyDataset(x_train, y_train, use='train')
    valid_ds = MyDataset(x_valid, y_valid, use='valid')
    test_ds = MyDataset(x_test, y_test, use='test')
    train_dl = DataLoader(train_ds, batch_size=batchs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batchs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batchs, shuffle=True)

    dataloaders = {'train': train_dl, 'val': valid_dl, 'test': test_dl}
    return dataloaders

def train_epoch(model, dataloaders, optimizer, criterion):
    for use in ['train', 'val']:
        model.train() if use == 'train' else model.eval()
        e_loss, e_acc_cnt = 0, 0

        for inputs, labels in tqdm(dataloaders[use]):
            optimizer.zero_grad()
            with torch.set_grad_enabled(use == 'train'):
                outs = model(inputs)
                loss = criterion(outs, labels)
                _, preds = torch.max(outs, 1)
                if use == 'train':
                    loss.backward()
                    optimizer.step()

                e_loss += loss.item()
                e_acc_cnt += torch.sum(preds == labels.data)

        e_loss = e_loss / len(dataloaders[use].dataset)
        e_acc = e_acc_cnt.double() / len(dataloaders[use].dataset)
        print(f'{use} Loss: {e_loss}, Acc: {e_acc}')
        if use == 'train':
            train_loss = e_loss
            train_acc = e_acc
        else:
            valid_loss = e_loss
            valid_acc = e_acc
    return train_loss, train_acc, valid_loss, valid_acc

def train(model, dataloaders, lr, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    for e in range(epochs):
        print(f'---------- epoch {e+1} / {epochs} ----------')
        tloss, tacc, vloss, vacc = train_epoch(model, dataloaders, optimizer, criterion)
        # ------- add -------
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'result76_checkpoint/checkpoint{e+1}.pth')
        # -------------------
        train_loss.append(tloss)
        train_acc.append(tacc)
        valid_loss.append(vloss)
        valid_acc.append(vacc)


    return train_loss, train_acc, valid_loss, valid_acc

def show_loss_acc_graph(epochs, loss_tup, acc_tup, use_list, output):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    epochs = np.arange(epochs)
    for use, loss, acc in zip(use_list, loss_tup, acc_tup):
        ax[0].plot(epochs, loss, label=use)
        ax[1].plot(epochs, acc, label=use)
    ax[0].set_title('loss')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[1].set_title('acc')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('acc')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    plt.tight_layout()
    plt.savefig(output)

def main():
    model_SN = SingleNet(300, 4)
    x_train = load_pkl('x_train.pkl')
    y_train = load_pkl('y_train.pkl')
    x_valid = load_pkl('x_valid.pkl')
    y_valid = load_pkl('y_valid.pkl')
    x_test = load_pkl('x_test.pkl')
    y_test = load_pkl('y_test.pkl')

    batchs = 1 # non minibatch
    epochs = 10
    lr = 0.01
    dataloaders = make_dataloaders(x_train, y_train, x_valid, y_valid, x_test, y_test, batchs)
    train_loss, train_acc, valid_loss, valid_acc = train(model_SN, dataloaders, lr, epochs)
    # output = 'result75.png'
    # loss_tup = (train_loss, valid_loss)
    # acc_tup = (train_acc, valid_acc)
    # use_list = ['train', 'valid']
    # show_loss_acc_graph(epochs, loss_tup, acc_tup, use_list, output)

    # model_save = 'model76.pth'
    # torch.save(model_SN, model_save)

if __name__ == '__main__':
    main()