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

def rm_markup(info_d):
    pattern = re.compile(r'\'{2,5}', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub('', value) for key, value in info_d.items()}

    return new_dict

text_uk = load_json_for_uk()
info_dict = extract_info(text_uk)
no_markup = rm_markup(info_dict)

for key, value in no_markup.items():
    print(f"{key} : {value}")