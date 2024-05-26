import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import optuna

def load_csv(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

class Objective:
    def __init__(self, train_x, train_y, valid_x, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

    def __call__(self, trial):
        params = {
            'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            'C': trial.suggest_uniform('C', 0.001, 10),
            'max_iter': 10000
        }

        model = LogisticRegression(**params)
        model.fit(self.train_x, self.train_y)
        valid_pred = model.predict(self.valid_x)
        acc = accuracy_score(self.valid_y, valid_pred)
        
        return acc

def main():
    train_file = 'train.txt'
    train_x_file = 'train.feature.txt'
    valid_file = 'valid.txt'
    valid_x_file = 'valid.feature.txt'
    test_file = 'test.txt'
    test_x_file = 'test.feature.txt'
    train_df, valid_df, test_df = load_csv(train_file, valid_file, test_file)
    train_x, valid_x, test_x = load_csv(train_x_file, valid_x_file, test_x_file)

    objective = Objective(train_x, train_df['CATEGORY'], valid_x, valid_df['CATEGORY'])
    study = optuna.create_study(direction='maximize')
    study.optimize(objective)
    print(f"valid_acc:{study.best_trial.value}")
    print(f'params:{study.best_params}')

if __name__ == '__main__':
    main()