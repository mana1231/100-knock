import random

def shuffle_word(word):
    if len(word) <= 4:
        return word
    else:
        middle_chars = list(word[1:-1])
        random.shuffle(middle_chars)
        return word[0] + ''.join(middle_chars) + word[-1]
    
def shuffle_sentence(sentence):
    words = sentence.split()
    shuffled_sentence = ' '.join(shuffle_word(word) for word in words)
    return shuffled_sentence

main_text = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind."
result= shuffle_sentence(main_text)

print("元の文:", main_text)
print("シャッフル後の文:", result)