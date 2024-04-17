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

def extract_src_dst_n2v(sentence):
    src_dst_n2v = []
    for chunk in sentence:
        if chunk.dst != -1:
            src_pos = [morph.pos for morph in chunk.morphs]
            dst_pos = [morph.pos for morph in sentence[chunk.dst].morphs]
            if '名詞' in src_pos and '動詞' in dst_pos:
                src_text = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
                dst_text = ''.join([morph.surface for morph in sentence[chunk.dst].morphs if morph.pos != '記号'])
                src_dst_n2v.append(f'{src_text}\t{dst_text}')
    return src_dst_n2v

def main():
    file_path = 'ai.ja.txt.parsed'
    chunk_sentences = load_chunk_file(file_path)
    for i in range(len(chunk_sentences)):
        print(f"-----parsed{i}-----")
        src_dst = extract_src_dst_n2v(chunk_sentences[i])
        for text in src_dst:
            print(text)

if __name__ == '__main__':
    main()