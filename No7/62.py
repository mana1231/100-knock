from gensim.models import KeyedVectors

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    similars = model.most_similar('United_States', topn=10)
    for i, name, num in enumerate(similars):
        print(f'{i+1}\t{name}\t{num}')

if __name__ == '__main__':
    main()