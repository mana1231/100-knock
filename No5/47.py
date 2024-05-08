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

def extract_verb_mining(sentence):
    infos = []
    for chunk in sentence:
        pred = ''
        for morph in chunk.morphs:
            if morph.pos == '動詞':
                if len(chunk.srcs):
                    temp_list = []
                    temp_dict = {}
                    temp = ''
                    flag_wo = 0
                    for src in chunk.srcs:
                        flag = 0
                        for word in sentence[src].morphs:
                            if word.pos == '名詞' and word.pos1 == 'サ変接続' and not flag_wo:
                                pred += word.surface
                                flag = 1
                                continue
                            elif word.pos == '助詞' and word.surface == 'を' and flag and not flag_wo:
                                pred += word.surface
                                flag = 0
                                flag_wo = 1
                                continue
                            elif word.pos != '記号':
                                temp += word.surface
                                flag = 0

                            if word.pos == '助詞':
                                temp_list.append(word.surface)
                                temp_dict[word.surface] = temp
                                temp = ''
                                flag = 0
                            else:
                                flag = 0
                        temp = ''
                    if len(temp_list) and flag_wo:
                        temp_list = sorted(list(set(temp_list)))
                        infos.append(f"{pred}{morph.base}\t{' '.join(temp_list)}\t{' '.join([temp_dict[word] for word in temp_list])}")
                break
    return infos

def main():
    file_path = 'ai.ja.txt.parsed'
    chunk_sentences = load_chunk_file(file_path)
    for i in range(len(chunk_sentences)):
        print(f"-----parsed{i}-----")
        infos = extract_verb_mining(chunk_sentences[i])
        for info in infos:
            print(info)
    
if __name__ == '__main__':
    main()