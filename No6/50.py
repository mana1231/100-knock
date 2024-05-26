# ! unzip news+aggregator.zip
# ! pip install scikit-learn
# ! pip install pandas

import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(file_name, tgts):
    df = pd.read_csv(file_name, encoding='utf-8', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    extract_df = df.loc[df['PUBLISHER'].isin(tgts), ['TITLE', 'CATEGORY']].sample(frac=1, random_state=0, ignore_index=True)
    return extract_df

def split_data_save(data):
    train, valid_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=0, stratify=data['CATEGORY'])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=0, stratify=valid_test['CATEGORY'])
    train.to_csv('train.txt', sep='\t', index=False)
    valid.to_csv('valid.txt', sep='\t', index=False)
    test.to_csv('test.txt', sep='\t', index=False)

    print(f'train\n{train['CATEGORY'].value_counts()}')
    print(f'valid\n{valid['CATEGORY'].value_counts()}')
    print(f'test\n{test['CATEGORY'].value_counts()}')

def main():
    data_file = 'newsCorpora.csv'
    tgts = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
    data = load_csv(data_file, tgts)

    print(data.head())
    split_data_save(data)

if __name__ == '__main__':
    main()


# FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP
# where:
# ID		Numeric ID
# TITLE		News title 
# URL		Url
# PUBLISHER	Publisher name
# CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)
# STORY		Alphanumeric ID of the cluster that includes news about the same story
# HOSTNAME	Url hostname
# TIMESTAMP 	Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970

