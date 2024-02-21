def load_mecab_file(file_path):
    morphemes = []
    flag = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() != 'EOS':
                flag = 0
                surface, feature = line.split('\t')
                features = feature.split(',')
                morph = {'surface': surface, 'base': features[6], 'pos': features[0], 'pos1': features[1]}
                morphemes.append(morph)
            elif flag:
                continue
            else:
                flag = 1
                yield morphemes
                morphemes = []

def extract_longest_noun_seq(lines):
    series_noum = set()
    current_seq = []
    for morphemes in lines:
        for morph in morphemes:
            if morph['pos'] == '名詞':
                current_seq.append(morph['surface'])
            else:
                if current_seq:
                    text = "".join(current_seq)
                    series_noum.add(text)
                current_seq = []
    return series_noum

mecab_file = "neko.txt.mecab"
morphs = load_mecab_file(mecab_file)

noun_sequences = extract_longest_noun_seq(morphs)
print(noun_sequences)