import matplotlib.pyplot as plt # type:ignore
import re
import sys

def read_score(filename):
    with open(filename) as f:
        x = f.readlines()[1]
        x = re.search(r'(?<=BLEU4 = )\d*\.\d*(?=,)', x)
        return float(x.group())
    
def main():
    xs = range(1, 6)
    path = sys.argv[1]
    output_file = sys.argv[2]
    ys = [read_score(f'{path}/eval98.{x}.score') for x in xs]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o')
    plt.grid(True)
    plt.savefig(f'{path}/{output_file}', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
