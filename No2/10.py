import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "cnt", "year"])

print(len(data))

'''
wc -l popular-names.txt

-l : 行数のみを表示
'''