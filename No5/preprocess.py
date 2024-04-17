import CaboCha

def parse_with_cabocha(input_file, output_file):
    cabocha = CaboCha.Parser()
    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read()

    tree = cabocha.parse(input_text)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tree.toString(CaboCha.FORMAT_LATTICE))

def main():    
    f_in = 'ai.ja.txt'
    f_out = 'ai.ja.txt.parsed'
    parse_with_cabocha(f_in, f_out)

if __name__ == '__main__':
    main()