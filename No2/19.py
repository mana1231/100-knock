import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

col1_frequency = data['name'].value_counts()

print(col1_frequency)