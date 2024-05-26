import pandas as pd
import re
from nltk import stem
from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stemmer = stem.PorterStemmer()
    res = [stemmer.stem(x) for x in text.split()]
    return ' '.join(res)

def make_vec(train, valid, test):
    df = pd.concat([train, valid, test], axis=0).reset_index(drop=True)
    df['TITLE'] = df['TITLE'].apply(preprocess)

    vec = TfidfVectorizer()
    x = vec.fit_transform(df['TITLE']).toarray()
    x_df = pd.DataFrame(x, columns=vec.get_feature_names_out())
    train_x = x_df.iloc[:len(train), :]
    valid_x = x_df.iloc[len(train):len(train)+ len(valid), :]
    test_x = x_df.iloc[len(train)+ len(valid):, :]
    train_x.to_csv('train.feature.txt', sep='\t', index=False)
    valid_x.to_csv('valid.feature.txt', sep='\t', index=False)
    test_x.to_csv('test.feature.txt', sep='\t', index=False)

def main():
    train_file = 'train.txt'
    valid_file = 'valid.txt'
    test_file = 'test.txt'

    train_df, valid_df, test_df = load_csv(train_file, valid_file, test_file)
    make_vec(train_df, valid_df, test_df)
    
if __name__ == '__main__':
    main()