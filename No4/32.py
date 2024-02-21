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
            
def extract_verb_bases(lines):
    verb_bases = set()
    for morphemes in lines:
        for morph in morphemes:
            if morph['pos'] == '動詞':
                verb_bases.add(morph['base'])
    return verb_bases

mecab_file = "neko.txt.mecab"
morphs = load_mecab_file(mecab_file)

verb_bases = extract_verb_bases(morphs)
print(verb_bases)