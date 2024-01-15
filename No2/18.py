import pandas as pd

data = pd.read_table('popular-names.txt', names=["name", "sex", "num", "year"])

sorted_data = data.sort_values(by='num', ascending=False)

print(sorted_data)