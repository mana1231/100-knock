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

def main():
    x_train = load_pkl('x_train.pkl')
    model_SN = SingleNet(300, 4)
    outs = model_SN(x_train[0])
    y1_hat = nn.Softmax(dim=-1)(outs)
    print(f'y1_hat:{y1_hat}')
    outs = model_SN(x_train[:4])
    Y_hat = nn.Softmax(dim=-1)(outs)
    print(f'Y_hat:{Y_hat}')

if __name__ == '__main__':
    main()
