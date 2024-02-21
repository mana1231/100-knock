import pandas as pd
import re

def load_json_for_uk():
    data = pd.read_json('jawiki-country.json', lines=True, encoding='utf-8')
    uk_article = data.query('title=="イギリス"')['text'].values[0]

    return uk_article

def extract_sections(text):
    section_pattern = re.compile(r'(={2,})([^=]+)\1') # (==+) -> ={2,}
    matches = section_pattern.findall(text)

    sections = [(match[1].strip(), len(match[0]) - 1) for match in matches]

    return sections

def main():
    text_uk = load_json_for_uk()
    sections = extract_sections(text_uk)

    for section, level in sections:
        print(f"name:{section}    level:{level}")

if __name__ == '__main__':
    main()