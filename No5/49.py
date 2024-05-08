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

def extract_n2root_path(now, sentence):
    chunk = sentence[now]
    path = []
    if chunk.dst != -1:
        path.append(now)
        next = chunk.dst
        while next != -1:
            path.append(next)
            next = sentence[next].dst    
    return path

def extract_n_chunks(sentence):
    n_chunks = []
    for i, chunk in enumerate(sentence):
        if '名詞' in [word.pos for word in chunk.morphs]:
            n_chunks.append(i)
    return n_chunks

from itertools import combinations

def make_pairs(nouns, sentence):
    for i, j in combinations(nouns, 2):
        x = extract_n2root_path(i, sentence)
        y = extract_n2root_path(j, sentence)
        path1 = []
        path2 = []
        path3 = []
        flag = 0
        if j in x:
            for k in x:
                if k == i:
                    morphs = 'X' + ''.join([word.surface if word.pos == '助詞' else '' for word in sentence[k].morphs])
                    path1.append(''.join(morphs))
                elif k != j:
                    morphs = ''.join([word.surface if word.pos != '記号' else '' for word in sentence[k].morphs])
                    path1.append(''.join(morphs))
                else:
                    morphs = 'Y' + ''.join([word.surface if word.pos == '助詞' else '' for word in sentence[k].morphs])
                    path1.append(''.join(morphs))
                    break
            print(' -> '.join(path1))
        else:
            for k in x:
                if k == i:
                    morphs = 'X' + ''.join([word.surface if word.pos == '助詞' else '' for word in sentence[k].morphs])
                    path1.append(''.join(morphs))
                elif k not in y:
                    morphs = ''.join([word.surface if word.pos != '記号' else '' for word in sentence[k].morphs])
                    path1.append(''.join(morphs))
                else:
                    break
            for l in y:
                if l == j:
                    morphs = 'Y' + ''.join([word.surface if word.pos == '助詞' else '' for word in sentence[l].morphs])
                    path2.append(''.join(morphs))
                elif l != k:
                    morphs = ''.join([word.surface if word.pos != '記号' else '' for word in sentence[l].morphs])
                    path2.append(''.join(morphs))
                elif l == k or flag:
                    flag = 1
                    morphs = ''.join([word.surface if word.pos != '記号' else '' for word in sentence[l].morphs])
                    path3.append(''.join(morphs))
            print(' -> '.join(path1) + ' | ' + ' -> '.join(path2) + ' | ' + ' -> '.join(path3))

def main():
    file_path = 'ai.ja.txt.parsed'
    chunk_sentences = load_chunk_file(file_path)
    for i in range(len(chunk_sentences)):
        print(f"-----parsed{i}-----")
        nouns = extract_n_chunks(chunk_sentences[i])
        make_pairs(nouns, chunk_sentences[i])

if __name__ == '__main__':
    main()
    