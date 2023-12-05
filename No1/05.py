def generate_ngram(seq, n):
    ngrams = []
    for i in range(len(seq) - n + 1):
        ngram = seq[i:i + n]
        ngrams.append(ngram)
    return ngrams

main_text = "I am an NLPer"
word = main_text.split()

word_bigrams = generate_ngram(word, 2)
print("Word Bigrams:", word_bigrams)

char_bigrams = generate_ngram(main_text, 2)
print("Character Bigrams:", char_bigrams)
