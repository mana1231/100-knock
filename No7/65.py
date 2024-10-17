def analogy_task_acc(sim_file):
    sem_cnt, sem = 0, 0
    syn_cnt, syn = 0, 0

    with open(sim_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            tgt = cols[1].split()[-1]
            pred = cols[2]
            if not cols[0].startswith('gram'):
                sem_cnt += 1
                if tgt == pred:
                    sem += 1
            else:
                syn_cnt += 1
                if tgt == pred:
                    syn += 1

    sem_acc = sem / sem_cnt
    syn_acc = syn / syn_cnt
    print(f'sem_acc:{sem_acc}\tsyn_acc:{syn_acc}')

def main():
    sim_file = f'questions-words_similarity.txt'
    analogy_task_acc(sim_file)

if __name__ == '__main__':
    main()