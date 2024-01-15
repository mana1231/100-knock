with open('popular-names.txt', 'r') as file:
    content = file.read()

modified_content = content.replace('\t', ' ')

print(modified_content)

# UNIX : sed 's/\t/ /g' popular-names.txt
# UNIX : tr '\t' ' ' < popular-names.txt
# UNIX : expand -t 1 popular-names.txt
