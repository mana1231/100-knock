# ---------------------------------------------------------------
# MacOSにて実行
# brew install mecab
# brew isntall mecab-ipadic
# この時、文字化けが起こる場合には
# brew reinstall --build-from-source mecab
# brew reinstall --build-from-source mecab-ipadic
# でOK。事前にneologdとか設定してしまうとchar.binが既存になってしまい
# エラーを出すのでその時はそのファイルを削除すればOK
# ---------------------------------------------------------------

import MeCab

def mecab_analysis(text):
    tokenizer = MeCab.Tagger()
    token = tokenizer.parse(text)
    return token

def save_mecab_result(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            result = mecab_analysis(line)
            outfile.write(result)

def main():
    f_in = "neko.txt"
    f_out = "neko.txt.mecab"
    save_mecab_result(f_in, f_out)

if __name__ == '__main__':
    main()