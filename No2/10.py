with open('popular-names.txt', 'r') as file:
    lines = file.readlines()
    
line_count = len(lines)
print("Pythonプログラムによる行数:", line_count)

# UNIX : wc -l popular-names.txt
