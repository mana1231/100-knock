from gensim.models import KeyedVectors

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    print(model['United_States'])

if __name__ == '__main__':
    main()
