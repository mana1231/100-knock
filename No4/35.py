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

from collections import Counter
def count_word_frequency(morphs):
    all_word_frequency = Counter()
    for sentence in morphs:
        words = [morph['base'] for morph in sentence if morph['pos'] != '記号']
        word_frequency = Counter(words)
        all_word_frequency += word_frequency

    return all_word_frequency

def main():
    mecab_file = "neko.txt.mecab"
    morphs = load_mecab_file(mecab_file)
    all_word_freq = count_word_frequency(morphs)
    sorted_word_freq = all_word_freq.most_common()
    for word, freq in sorted_word_freq:
        print(f'{word}: {freq}回')

if __name__ == '__main__':
    main()