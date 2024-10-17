import torch
from torch import nn
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

class MLNet(nn.Module):
    def __init__(self, insize, hiddensize, outsize):
        super(MLNet, self).__init__()
        self.layers = nn.Sequential(
                    nn.Linear(insize, hiddensize),
                    nn.BatchNorm1d(hiddensize),
                    nn.ReLU(),
                    nn.Linear(hiddensize, hiddensize),
                    nn.BatchNorm1d(hiddensize),
                    nn.ReLU(),
                    nn.Linear(hiddensize, outsize),
                    )

    def forward(self, x):
        outs = self.layers(x)
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

def train_epoch(model, dataloaders, optimizer, criterion, device):
    st = time.time()
    for use in ['train', 'val']:
        model.train() if use == 'train' else model.eval()
        e_loss, e_acc_cnt = 0, 0

        for inputs, labels in tqdm(dataloaders[use]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(use == 'train'):
                outs = model(inputs)
                loss = criterion(outs, labels)
                _, preds = torch.max(outs, 1)
                if use == 'train':
                    loss.backward()
                    optimizer.step()

                e_loss += loss.item() * inputs.size(0)
                e_acc_cnt += torch.sum(preds == labels.data)

        e_loss = e_loss / len(dataloaders[use].dataset)
        e_acc = e_acc_cnt.double() / len(dataloaders[use].dataset)
        if use == 'train':
            train_loss = e_loss
            train_acc = e_acc
        else:
            valid_loss = e_loss
            valid_acc = e_acc
    en = time.time()
    epoch_time = en-st

    return train_loss, train_acc, valid_loss, valid_acc, epoch_time

def train(model, dataloaders, lr, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    total_time = 0
    best_valid_acc = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6, last_epoch=-1)

    for e in range(epochs):
        tloss, tacc, vloss, vacc, epoch_time = train_epoch(model, dataloaders, optimizer, criterion, device)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 'checkpoint/79checkpoint{e+1}.pth')
        if vacc > best_valid_acc:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'checkpoint/79best.pth')
        train_loss.append(tloss)
        train_acc.append(tacc.cpu())
        valid_loss.append(vloss)
        valid_acc.append(vacc.cpu())
        total_time += epoch_time
        print(f'Epoch {e+1} / {epochs} (train) Loss: {tloss:.4f}, Acc: {tacc:.4f}, (val) Loss: {vloss:.4f}, Acc: {vacc:.4f}')
        scheduler.step()
    avg_time = total_time/epochs

    return train_loss, train_acc, valid_loss, valid_acc, avg_time

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

def get_acc(model, dataloader, device):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs = model(inputs)
            _, preds = torch.max(outs, 1)
            corrects += torch.sum(preds == labels.data)
    return corrects / len(dataloader.dataset)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_train = load_pkl('x_train.pkl')
    y_train = load_pkl('y_train.pkl')
    x_valid = load_pkl('x_valid.pkl')
    y_valid = load_pkl('y_valid.pkl')
    x_test = load_pkl('x_test.pkl')
    y_test = load_pkl('y_test.pkl')

    batchs = 512
    lr = 0.001
    epochs = 1000
    print(f'device: {device}')
    model_ML = MLNet(300, 256, 4)
    dataloaders = make_dataloaders(x_train, y_train, x_valid, y_valid, x_test, y_test, batchs)
    train_loss, train_acc, valid_loss, valid_acc, avg_time = train(model_ML, dataloaders, lr, epochs, device)

    output = 'result79.png'
    loss_tup = (train_loss, valid_loss)
    acc_tup = (train_acc, valid_acc)
    use_list = ['train', 'valid']
    show_loss_acc_graph(epochs, loss_tup, acc_tup, use_list, output)

    best_model = 'checkpoint/79best.pth'
    checkpoint = torch.load(best_model)
    model_ML.load_state_dict(checkpoint['model_state_dict'])

    test_acc = get_acc(model_ML, dataloaders['test'], device)
    print(f'test acc: {test_acc}')

if __name__ == '__main__':
    main()