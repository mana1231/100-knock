from gensim.models import KeyedVectors

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    print(model.similarity('United_States', 'U.S.'))

if __name__ == '__main__':
    main()