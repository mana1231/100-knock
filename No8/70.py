from gensim.models import KeyedVectors
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pickle


def load_csv(file_name, tgts):
    df = pd.read_csv(file_name, encoding='utf-8', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    extract_df = df.loc[df['PUBLISHER'].isin(tgts), ['TITLE', 'CATEGORY']].sample(frac=1, random_state=0, ignore_index=True)
    return extract_df

def split_data(data):
    train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=0, stratify=data['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=0, stratify=valid_test['CATEGORY'])
    
    return train, valid, test

def w2v(text, model):
    words = text.strip().split()
    v = [model[w] for w in words if w in model]
    avg_v = np.array(sum(v)/len(v))
    return avg_v

def make_vec(data, model):
    vecs = np.array([])
    for i, text in data.iterrows():
        if len(vecs) == 0:
            vecs = w2v(text['TITLE'], model)
        else:
            vecs = np.vstack([vecs, w2v(text['TITLE'], model)])

    x = torch.from_numpy(vecs)
    y = data['CATEGORY'].map({'b': 0, 'e': 1, 't': 2, 'm': 3})

    return x, y

def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def main():
    data_file = 'newsCorpora.csv'
    tgts = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
    data = load_csv(data_file, tgts)

    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)

    train, valid, test = split_data(data)
    x_train, y_train = make_vec(train, model)
    x_valid, y_valid = make_vec(valid, model)
    x_test, y_test = make_vec(test, model)

    save_pkl(x_train, 'x_train.pkl')
    save_pkl(y_train, 'y_train.pkl')
    save_pkl(x_valid, 'x_valid.pkl')
    save_pkl(y_valid, 'y_valid.pkl')
    save_pkl(x_test, 'x_test.pkl')
    save_pkl(y_test, 'y_test.pkl')

if __name__ == '__main__':
    main()