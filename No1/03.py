def remove_punctuation(text):
    text = ''.join(tmp if tmp.isalpha() or tmp.isspace() else ' ' for tmp in text)
    return text

def words_len(text):
    text = remove_punctuation(text)
    word_len = [len(word) for word in text.split()]
    return word_len


main_text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
result = words_len(main_text)
print(result)