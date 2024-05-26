import pandas as pd
import numpy as np
import pickle

def load_csv(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def score(lg, X):
    pred = lg.predict(X)
    proba = np.max(lg.predict_proba(X), axis=1)
    return pred, proba

def main():
    train_x_file = 'train.feature.txt'
    valid_x_file = 'valid.feature.txt'
    test_x_file = 'test.feature.txt'
    train_x, valid_x, test_x = load_csv(train_x_file, valid_x_file, test_x_file)

    with open('6model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    features = train_x.columns.values
    for i in range(len(model.classes_)):
        coef = model.coef_[i].round(3)
        features = [(features[j], np.abs(coef[j])) for j in range(len(features))]
        print(f'-----category={model.classes_[i]}-----')
        bests = sorted(features, key=lambda w: w[1], reverse=True)
        worsts = sorted(features, key=lambda w: w[1])
        print('top10 of w high')
        for k in range(10):
            print(f'{k+1}\t{bests[k][0]}\t{bests[k][1]}')
        print('top10 of w low')
        for k in range(10):
            print(f'{k+1}\t{worsts[k][0]}\t{worsts[k][1]}')

if __name__ == '__main__':
    main()