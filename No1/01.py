def take_odd_words(text):
    result = ""
    for i in range(0,8,2):
        result += text[i]
    return result

main_text = "パタトクカシーー"
result =  take_odd_words(main_text)
print(result)