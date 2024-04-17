class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __str__(self):
        return f'surface: {self.surface}, base: {self.base}, pos: {self.pos}, pos1: {self.pos1}'
    
    def __repr__(self):
        return self.surface

def load_parsed_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            if line.startswith('*'):
                continue
            elif line == 'EOS\n':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                cols = line.split('\t')
                res_cols = cols[1].strip().split(',')
                morph = Morph(cols[0], res_cols[6], res_cols[0], res_cols[1])
                sentence.append(morph)
    return sentences

def main():
    file_path = 'ai.ja.txt.parsed'
    parsed_sentences = load_parsed_file(file_path)
    for i in range(len(parsed_sentences)):
        print(f"-----parsed{i}-----")
        for morph in parsed_sentences[i]:
            print(str(morph))
            
if __name__ == '__main__':
    main()