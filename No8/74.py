import torch
import pickle
import torch.utils.data as data

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


def get_acc(model, dataloader):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outs = model(inputs)
            _, preds = torch.max(outs, 1)
            corrects += torch.sum(preds == labels.data)
    return corrects / len(dataloader.dataset)

def main():
    model_path = 'model73.pth'
    model = torch.load(model_path)
    x_train = load_pkl('x_train.pkl')
    y_train = load_pkl('y_train.pkl')
    x_valid = load_pkl('x_valid.pkl')
    y_valid = load_pkl('y_valid.pkl')
    x_test = load_pkl('x_test.pkl')
    y_test = load_pkl('y_test.pkl')
    dataloaders = make_dataloaders(x_train, y_train, x_valid, y_valid, x_test, y_test, batchs=1)

    acc_train = get_acc(model, dataloaders['train'])
    print(f'acc_train: {acc_train}')
    acc_test = get_acc(model, dataloaders['test'])
    print(f'acc_test: {acc_test}')

if __name__ == '__main__':
    main()