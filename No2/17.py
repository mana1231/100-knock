import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

col1_set = set(data['name'])

for item in col1_set:
    print(item)