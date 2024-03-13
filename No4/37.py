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
def find_cooccurrences(target_word, morphs):
    # find_cooccurrences(target_word, morphs, window_size=num_of_size)
    cooccurrs = Counter()
    for sentence in morphs:
        for i, morph in enumerate(sentence):
            if morph['base'] == target_word:
                # start = max(0, i - window_size)
                # end = min(len(sentence), i + window_size + 1)
                context_words = [sentence[j]['base'] for j in range(0, len(sentence)) if i != j]
                cooccurrs.update(context_words)
    return cooccurrs

import matplotlib.pyplot as plt
import japanize_matplotlib
def make_bar_graph(x, y, xlabel, ylabel, title):
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def main():
    mecab_file = "neko.txt.mecab"
    morphs = load_mecab_file(mecab_file)
    tgt_word = '猫'
    cooccurs = find_cooccurrences(tgt_word, morphs)
    words, freqs = zip(*cooccurs.most_common(10))
    xlabel = "単語"
    ylabel = "出現頻度"
    title = f'「{tgt_word}」と共起頻度が高い10語'
    make_bar_graph(words, freqs, xlabel, ylabel, title)

if __name__ == '__main__':
    main()