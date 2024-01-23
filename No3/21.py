import pandas as pd
import re

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def extract_category(text):
    category_pattern = re.compile(r'.*\[\[Category:.*')
    category_lines = category_pattern.findall(text)
    
    return category_lines

text_uk = load_json_for_uk()
category_lines = extract_category(text_uk)

for line in category_lines:
    print(line)