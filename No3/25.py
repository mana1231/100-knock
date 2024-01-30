import pandas as pd
import re

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def extract_info(text):
    info_pattern = re.compile(r'^\{\{基礎情報.*?$(.*?)^\}\}', re.MULTILINE + re.S)
    info_text = info_pattern.findall(text)

    field_pattern = re.compile(r'^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))', re.MULTILINE + re.S)
    info = field_pattern.findall(info_text[0])
    info_dict = dict(info)

    return info_dict

text_uk = load_json_for_uk()
info_dict = extract_info(text_uk)
for key, value in info_dict.items():
    print(f"{key} : {value}")