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

def extract_noun_phrases(lines):
    noun_phrases = set()
    for morphemes in lines:
        for i in range(len(morphemes) - 2):
            if morphemes[i]['pos'] == '名詞' and morphemes[i + 1]['surface'] == 'の' and morphemes[i + 2]['pos'] == '名詞':
                noun_phrase = morphemes[i]['surface'] + morphemes[i + 1]['surface'] + morphemes[i + 2]['surface']
                noun_phrases.add(noun_phrase)
    return noun_phrases

mecab_file = "neko.txt.mecab"
morphs = load_mecab_file(mecab_file)

noun_phrases = extract_noun_phrases(morphs)
print(noun_phrases)