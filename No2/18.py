import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

sorted_data = data.sort_values(by='num', ascending=False)

print(sorted_data)

'''
sort -k3nr popular-names.txt

-k3 : 3番目のkeyについて処理(,を利用することで範囲指定も可能)
n : 数値としてsort
r : 逆順にsort
'''