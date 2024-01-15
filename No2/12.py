import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])
data['name'].to_csv('col1.txt', index=False, header=False)
data['sex'].to_csv('col2.txt', index=False, header=False)