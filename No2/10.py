import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "cnt", "year"])

print(len(data))
