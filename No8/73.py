import torch
from torch import nn
import pickle
import torch.utils.data as data
from tqdm import tqdm

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

class MyDataset(data.Dataset):
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
    train_dl = data.DataLoader(train_ds, batch_size=batchs, shuffle=True)
    valid_dl = data.DataLoader(valid_ds, batch_size=batchs, shuffle=True)
    test_dl = data.DataLoader(test_ds, batch_size=batchs, shuffle=True)

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

def train(model, dataloaders, lr, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for e in range(epochs):
        print(f'---------- epoch {e+1} / {epochs} ----------')
        train_epoch(model, dataloaders, optimizer, criterion)

def main():
    model_SN = SingleNet(300, 4)
    x_train = load_pkl('x_train.pkl')
    y_train = load_pkl('y_train.pkl')
    x_valid = load_pkl('x_valid.pkl')
    y_valid = load_pkl('y_valid.pkl')
    x_test = load_pkl('x_test.pkl')
    y_test = load_pkl('y_test.pkl')

    batchs = 1
    epochs = 10
    lr = 0.01
    dataloaders = make_dataloaders(x_train, y_train, x_valid, y_valid, x_test, y_test, batchs)
    train(model_SN, dataloaders, lr, epochs)

    model_save = 'model73.pth'
    torch.save(model_SN, model_save)

if __name__ == '__main__':
    main()