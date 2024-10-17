from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
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


def main():    
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    
    sim_file = 'questions-words_similarity.txt'
    countries = get_countries_list(sim_file)
    countries_vec = [model[country] for country in countries]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(countries_vec)
    for i in range(5):
        cluster = np.where(kmeans.labels_ == i)[0]
        print(f'-----cluster:{i}-----')
        print('\n'.join([countries[k] for k in cluster]))

if __name__ == '__main__':
    main()