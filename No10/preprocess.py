import MeCab # type: ignore
import nltk # type:ignore
import sys

nltk.download('punkt', quiet=True)

def tokenize_japanese(text):
    tagger = MeCab.Tagger()
    parsed = tagger.parse(text)
    
    tokens = []
    for line in parsed.split('\n'):
        if line == 'EOS' or line == '':
            continue
        surface, feature = line.split('\t')
        tokens.append(surface)
    
    return tokens

def tokenize_english(text):
    return nltk.word_tokenize(text)

def tokenize_text(text, lang):
    if lang == 'ja':
        return tokenize_japanese(text)
    else:
        return tokenize_english(text)

def process_file(input_file, output_file, lang):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            tokens = tokenize_text(line.strip(), lang)
            f_out.write(' '.join(tokens) + '\n')

def main():
    if len(sys.argv) != 4:
        print("error : format -> python name.py infile outfile lang")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    lang = sys.argv[3]
    
    process_file(input_file, output_file, lang)
    print(f"done")

if __name__ == "__main__":
    main()