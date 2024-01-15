with open('col1.txt', 'r') as col1_file:
    col1_lines = col1_file.readlines()

with open('col2.txt', 'r') as col2_file:
    col2_lines = col2_file.readlines()

with open('merge_with_tab.txt', 'w') as merged_file:
    for col1, col2 in zip(col1_lines, col2_lines):
        merged_file.write(f"{col1.rstrip()}\t{col2.rstrip()}\n")