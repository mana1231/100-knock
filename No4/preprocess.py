# ---------------------------
# need mecab-ipadic-neologd
# ---------------------------

import MeCab

def mecab_analysis(text):
    tokenizer = MeCab.Tagger("-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
    token = tokenizer.parse(text)
    return token

def save_mecab_result(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            result = mecab_analysis(line)
            outfile.write(result)

f_in = "neko.txt"
f_out = "neko.txt.mecab"

save_mecab_result(f_in, f_out)