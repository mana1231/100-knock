import pandas as pd
import re

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def main():
    text_uk = load_json_for_uk()
    category_name_pattern = re.compile(r'\[\[Category:(.*?)(?:\|.*?)?\]\]')
    category_names = category_name_pattern.findall(text_uk)

    for category_name in category_names:
        print(category_name)

if __name__ == '__main__':
    main()