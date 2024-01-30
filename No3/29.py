import pandas as pd
import re
import requests

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

def rm_inlink(info_d):
    pattern = re.compile(r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub(r'\1', value) for key, value in info_d.items()}

    return new_dict

def rm_file(info_d):
    pattern = re.compile(r'\[\[ファイル(?:[^|]*?\|)*?([^|]*?)\]\]', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub(r'\1', value) for key, value in info_d.items()}

    return new_dict
    
def rm_lang(info_d):
    pattern = re.compile(r'\{\{lang(?:[^|]*?\|)*?([^|]*?)\}\}', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub(r'\1', value) for key, value in info_d.items()}

    return new_dict

def rm_exlink(info_d):
    pattern = re.compile(r'\[http:\/\/(?:[^\s]*?\s)?([^]]*?)\]', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub(r'\1', value) for key, value in info_d.items()}

    return new_dict

def rm_code(info_d):
    pattern = re.compile(r'<\/?[br|ref|blockquote|cite|p|code|del|ins][^>]*?>', re.MULTILINE + re.S)
    new_dict = {key: pattern.sub('', value) for key, value in info_d.items()}

    return new_dict

def remove_mediawiki_markup(info_d):
    new_dict = rm_code(rm_exlink(rm_lang(rm_file(rm_inlink(rm_markup(info_d))))))

    return new_dict

def get_url_for(tgt):
    url_head = 'https://www.mediawiki.org/w/api.php'  
    params = {'action': 'query',
              'titles': f'File:{tgt}',
              'format': 'json',
              'prop': 'imageinfo',
              'iiprop': 'url'}
    res = requests.get(url_head, params)
    tgt_url = res.json()['query']['pages']['-1']['imageinfo'][0]['url']

    return tgt_url

text_uk = load_json_for_uk()
info_dict = extract_info(text_uk)
new_dict = remove_mediawiki_markup(info_dict)

flag = new_dict['国旗画像']

url = get_url_for(flag)
print(url)