main_text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
text = ''.join(tmp if tmp.isalpha() or tmp.isspace() else ' ' for tmp in main_text)
result = [len(word) for word in text.split()]
print(result)
