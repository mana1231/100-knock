import matplotlib.pyplot as plt
import japanize_matplotlib
from collections import Counter
import numpy as np

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

def count_word_frequency(morphs):
    all_word_frequency = Counter()
    for sentence in morphs:
        words = [morph['base'] for morph in sentence if morph['pos'] != '記号']
        word_frequency = Counter(words)
        all_word_frequency += word_frequency

    return all_word_frequency

import numpy as np
def make_loglog_graph(x, y, xlabel, ylabel, title):
    plt.scatter(np.log(x), np.log(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    mecab_file = "neko.txt.mecab"
    morphs = load_mecab_file(mecab_file)
    all_word_freq = count_word_frequency(morphs)
    sorted_word_freq = all_word_freq.most_common()
    _, freqs = zip(*sorted_word_freq)
    ranks = range(1, len(freqs)+1)
    xlabel = "出現頻度順位(log)"
    ylabel = "出現頻度(log)"
    title = "両対数グラフ"
    make_loglog_graph(ranks, freqs, xlabel, ylabel, title)

if __name__ == '__main__':
    main()
