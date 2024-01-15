import pandas as pd

col1 = pd.read_csv('col1.txt', header=None, names=['name'])
col2 = pd.read_csv('col2.txt', header=None, names=['sex'])

merged_data = pd.concat([col1, col2], axis=1)
merged_data.to_csv('merged.txt', header=None, index=False, sep='\t')