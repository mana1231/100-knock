from torch import nn
import pickle

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

def show_loss_and_grad(model, x, y):
    criterion = nn.CrossEntropyLoss()
    outs = model(x)
    loss = criterion(outs, y)
    print(f'loss:{loss}')
    model.zero_grad()
    loss.backward()
    print(f'grad:{model.layer.weight.grad}')

def main():
    x_train = load_pkl('x_train.pkl')
    y_train = load_pkl('y_train.pkl')
    model_SN = SingleNet(300, 4)
    show_loss_and_grad(model_SN, x_train[0], y_train[0])
    show_loss_and_grad(model_SN, x_train[:4], y_train[:4])

if __name__ == '__main__':
    main()