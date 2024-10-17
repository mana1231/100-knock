from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

def get_countries_list(sim_file):
    categories1 = ['capital-common-countries', 'capital-world'] # capital-common-countries	Athens "Greece" Baghdad Iraq	Iraqi	0.635187029838562
    categories2 = ['currency', 'gram6-nationality-adjective'] # currency	"Algeria" dinar Angola kwanza	kwanza	0.5409914255142212
    countries = set()
    with open(sim_file, 'r', encoding='utf-8') as f:
        for line in f:
            cols = line.strip().split('\t')
            if cols[0] in categories1:
                country = cols[1].split()[1]
                countries.add(country)
            elif cols[0] in categories2:
                country = cols[1].split()[0]
                countries.add(country)
            else:
                continue
  
    return list(countries)

def show_result68(vec, labels, output):
    linkage_result = linkage(vec, method='ward')
    plt.figure(figsize=(16, 8))
    dendrogram(linkage_result, labels=labels)
    plt.savefig(output)
    plt.show()

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    
    sim_file = 'questions-words_similarity.txt'
    countries = get_countries_list(sim_file)
    countries_vec = [model[country] for country in countries]

    show_result68(countries_vec, countries, 'result68.png')

if __name__ == '__main__':
    main()