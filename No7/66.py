from gensim.models import KeyedVectors
from scipy.stats import spearmanr

def show_spearmanr(data_file, model):
    human = []
    w2v = []
    with open(data_file, 'r', encoding='utf-8') as f:
        next(f) # skip the header
        for line in f:
            cols = line.strip().split(',')
            human.append(float(cols[2]))
            w2v.append(model.similarity(cols[0], cols[1]))

    r, p = spearmanr(human, w2v)
    print(f'spearmanr:{r}\tp:{p}')

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    data_file = f'WordSimilarity-353 collection/combined.csv'
    show_spearmanr(data_file, model)

if __name__ == '__main__':
    main()