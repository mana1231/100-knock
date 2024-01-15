with open('popular-names.txt', 'r') as file:
    lines = file.readlines()

with open('col1.txt', 'w') as col1_file:
    for line in lines:
        col1_file.write(line.split('\t')[0] + '\n')

with open('col2.txt', 'w') as col2_file:
    for line in lines:
        col2_file.write(line.split('\t')[1] + '\n')