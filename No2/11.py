with open('popular-names.txt', 'r') as file:
    content = file.read()

modified_content = content.replace('\t', ' ')

print(modified_content)