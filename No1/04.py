def remove_punctuation(text):
    text = ''.join(tmp if tmp.isalpha() or tmp.isspace() else ' ' for tmp in text)
    return text

main_text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
text = remove_punctuation(main_text)
words = text.split()

result_dict = {}
for i, word in enumerate(words, 1):
    if i in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
        result_dict[i] = word[:1]
    else:
        result_dict[i] = word[:2]

print(result_dict)