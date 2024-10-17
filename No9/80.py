# 80
import pandas as pd
import re
from nltk import stem
from collections import Counter, OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '0', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def load_csv(file_name, tgts):
    df = pd.read_csv(file_name, encoding='utf-8', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    extract_df = df.loc[df['PUBLISHER'].isin(tgts), ['TITLE', 'CATEGORY']].sample(frac=1, random_state=0, ignore_index=True)
    extract_df['TITLE'] = extract_df['TITLE'].apply(preprocess)
    return extract_df

def split_data_save(data):
    train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=0, stratify=data['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=0, stratify=valid_test['CATEGORY'])
    train.to_csv(f'{main_path}/train.txt', sep='\t', index=False)
    valid.to_csv(f'{main_path}/valid.txt', sep='\t', index=False)
    test.to_csv(f'{main_path}/test.txt', sep='\t', index=False)

def load_csv2df(train_file, valid_file, test_file):
    train_df = pd.read_csv(train_file, encoding='utf-8', sep='\t')
    valid_df = pd.read_csv(valid_file, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_file, encoding='utf-8', sep='\t')
    return train_df, valid_df, test_df

def make_word2id(train_df):
    words = []
    for text in train_df['TITLE']:
        for word in text.rstrip().split():
            words.append(word)

    c = Counter(words)

    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    word2id = OrderedDict({token: i for i, token in enumerate(special_tokens)})
    id = 4
    for char, cnt in c.most_common():
        if cnt > 1:
            word2id[char] = id
            id += 1
    return word2id

def text2id(text, word2id):
    words = text.rstrip().split()
    return [word2id.get(word) for word in words]


def main():
    data_file = f'{main_path}/newsCorpora.csv'
    tgts = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
    data = load_csv(data_file, tgts)
    split_data_save(data)

    train_file = f'{main_path}/train.txt'
    valid_file = f'{main_path}/valid.txt'
    test_file = f'{main_path}/test.txt'

    train_df, valid_df, test_df = load_csv2df(train_file, valid_file, test_file)
    word2id = make_word2id(train_df)
    print(word2id)

    ex = train_df.at[0, 'TITLE']
    print(ex)
    print(text2id(ex, word2id))

if __name__ == '__main__':
    main()