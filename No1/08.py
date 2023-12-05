def cipher(text):
    result = ""
    for word in text:
        if word.islower():
            result += chr(219 - ord(word))
        else:
            result += word

    return result

main_text = "Hello, World!"

encrypted = cipher(main_text)
print("暗号化されたメッセージ:", encrypted)

decrypted = cipher(encrypted)
print("復号化されたメッセージ:", decrypted)