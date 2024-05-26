from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def load_csv(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def get_result_by_norms(train_df, train_x, valid_df, valid_x, test_df, test_x):
    results = []
    for c in tqdm(np.logspace(-2, 2, 10, base=10)):
        model = LogisticRegression(random_state=0, max_iter=10000, C=c)
        model.fit(train_x, train_df['CATEGORY'])
        
        train_pred = model.predict(train_x)
        valid_pred = model.predict(valid_x)
        test_pred = model.predict(test_x)
        
        train_acc = accuracy_score(train_df['CATEGORY'], train_pred)
        valid_acc = accuracy_score(valid_df['CATEGORY'], valid_pred)
        test_acc = accuracy_score(test_df['CATEGORY'], test_pred)
        results.append([c, train_acc, valid_acc, test_acc])
    
    return results

def show_fig(results):
    results = np.array(results)
    fig, ax = plt.subplots()
    ax.plot(results[:, 0], results[:, 1], label='train')
    ax.plot(results[:, 0], results[:, 2], label='valid')
    ax.plot(results[:, 0], results[:, 3], label='test')
    ax.set_xlabel('C')
    ax.set_ylabel('accuracy')
    ax.set_xscale('log')
    ax.legend()
    plt.show()

def main():
    train_file = 'train.txt'
    train_x_file = 'train.feature.txt'
    valid_file = 'valid.txt'
    valid_x_file = 'valid.feature.txt'
    test_file = 'test.txt'
    test_x_file = 'test.feature.txt'

    train_df, valid_df, test_df = load_csv(train_file, valid_file, test_file)
    train_x, valid_x, test_x = load_csv(train_x_file, valid_x_file, test_x_file)

    results = get_result_by_norms(train_df, train_x, valid_df, valid_x, test_df, test_x)
    show_fig(results)

if __name__ == '__main__':
    main()