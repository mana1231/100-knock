import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def score(lg, X):
    pred = lg.predict(X)
    proba = np.max(lg.predict_proba(X), axis=1)
    return pred, proba

def show_cnfusion_matrix(data, pred, output):
    cm = confusion_matrix(data, pred)
    cm = pd.DataFrame(cm, columns=['b', 't', 'e', 'm'])
    cm['TARGET'] = ['b', 't', 'e', 'm']
    cm = cm.set_index('TARGET')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(output)
    plt.close('all')

def main():
    train_file = 'train.txt'
    train_x_file = 'train.feature.txt'
    valid_file = 'valid.txt'
    valid_x_file = 'valid.feature.txt'
    test_file = 'test.txt'
    test_x_file = 'test.feature.txt'

    train_df, valid_df, test_df = load_csv(train_file, valid_file, test_file)
    train_x, valid_x, test_x = load_csv(train_x_file, valid_x_file, test_x_file)

    with open('6model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    train_pred, train_proba = score(model, train_x)
    valid_pred, valid_proba = score(model, valid_x)
    
    show_cnfusion_matrix(train_df['CATEGORY'], train_pred, 'train_sm.png')
    show_cnfusion_matrix(valid_df['CATEGORY'], valid_pred, 'valid_sm.png')

if __name__ == '__main__':
    main()