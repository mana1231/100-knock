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

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []

def load_chunk_file(file_path):
    sentences = []
    chunks = []
    morphs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('*'):
                if morphs:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                cols = line.split()
                dst = int(cols[2].strip().rstrip('D'))
            elif line == 'EOS\n':
                if morphs:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                if chunks:
                    for i, chunk in enumerate(chunks):
                        dst = chunk.dst
                        if dst != -1:
                            chunks[dst].srcs.append(i)
                    sentences.append(chunks)
                    chunks = []
            else:
                cols = line.split('\t')
                res_cols = cols[1].strip().split(',')
                morphs.append(Morph(cols[0], res_cols[6], res_cols[0], res_cols[1]))
    return sentences

def extract_verb_case(sentences, f_out):
    verb_case = []
    for sentence in sentences:
        for chunk in sentence:
            for morph in chunk.morphs:
                if morph.pos == '動詞':
                    if len(chunk.srcs):
                        temp = []
                        for src in chunk.srcs:
                            temp += [word.base for word in sentence[src].morphs if word.pos == '助詞']
                        if len(temp):
                            temp = sorted(list(set(temp)))
                            verb_case.append(f"{morph.base}\t{' '.join(temp)}")
                    break
    
    with open(f_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(verb_case))

def main():
    file_path = 'ai.ja.txt.parsed'
    f_out = "result45.txt"
    chunk_sentences = load_chunk_file(file_path)
    extract_verb_case(chunk_sentences, f_out)

if __name__ == '__main__':
    main()