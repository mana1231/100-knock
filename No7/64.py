from gensim.models import KeyedVectors

def show_result64(question_file, model, output):
    category = ''
    with open(question_file, 'r', encoding='utf-8') as f1, open(output, 'w', encoding='utf-8') as f2:
        for line in f1:
            if line.startswith(':'):
                category = line.rstrip()[2:]
                continue
            else:
                cols = line.rstrip().split()
                word, similarity = model.most_similar(positive=[cols[1], cols[2]], negative=[cols[0]], topn=1)[0]
                f2.write(f'{category}\t{line.rstrip()}\t{word}\t{similarity}\n')

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    question_file = 'questions-words.txt'
    output = '.questions-words_similarity.txt'
    show_result64(question_file, model, output)

if __name__ == '__main__':
    main()