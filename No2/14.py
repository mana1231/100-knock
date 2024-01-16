import sys
import pandas as pd

if len(sys.argv) != 2:
    print("エラー : 自然数 N を指定して下さい。")
    sys.exit(1)

try:
    N = int(sys.argv[1])
except ValueError:
    print("エラー : N は自然数を指定して下さい。")
    sys.exit(1)

data = pd.read_table('popular-names.txt', names=["name", "sex", "cnt", "year"])

print(data.head(N))

'''
head -n N popular-names.txt

-n N : 先頭からN行だけを表示するNを指定
'''