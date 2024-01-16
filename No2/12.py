import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])
data['name'].to_csv('col1.txt', index=False, header=False)
data['sex'].to_csv('col2.txt', index=False, header=False)

'''
cut -f 1 popular-names.txt > col1.txt
cut -f 2 popular-names.txt > col2.txt

-f n : n列目の処理を行う。
> filename : filenameへ出力結果を記録
'''