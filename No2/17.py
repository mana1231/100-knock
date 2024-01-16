import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

col1_set = sorted(set(data['name'])) # print(df.iloc[:,0].unique()) でも可能。 ただし、names指定なしの場合。

for item in col1_set:
    print(item)


'''
cut -f 1 popular-names.txt | sort | uniq

-f 1 : 1列目を処理する
sort : 文字列順に並べ替え
uniq : 重複なし
'''