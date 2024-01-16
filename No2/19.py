import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

col1_frequency = data['name'].value_counts()

print(col1_frequency)

'''
cut -f 1 popular-names.txt | sort | uniq -c | sort -nr

cut -f 1 : 1列目の処理を行う
sort : 並べ替え
uniq -c : 重複なし -cで横に出現頻度を見せる
sort -nr : その出現頻度に基づいてsort -rは降順、-nは数字として処理する
'''