import pandas as pd

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def main():
    text_uk = load_json_for_uk()
    print(text_uk)

if __name__ == '__main__':
    main()