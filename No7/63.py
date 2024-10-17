from gensim.models import KeyedVectors

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    similars = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
    for i, item in enumerate(similars):
        name, num = item[0], item[1]
        print(f'{i+1}\t{name}\t{num}')

if __name__ == '__main__':
    main()