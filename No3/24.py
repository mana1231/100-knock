import pandas as pd
import re

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def extract_media_files(text):
    file_pattern = re.compile(r'\[\[ファイル:(.+?)(?:\||\])')
    media_files = file_pattern.findall(text)

    return media_files

text_uk = load_json_for_uk()
media_files = extract_media_files(text_uk)

for file in media_files:
    print(file)