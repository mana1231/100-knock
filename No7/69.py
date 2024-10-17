from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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

def show_result69(kmeans, vec, labels, output):
    tsne = TSNE(n_components=2)
    x_items = tsne.fit_transform(np.array(vec))
    plt.figure(figsize=(10, 10))
    for x, country, color in zip(x_items, labels, kmeans.labels_):
        plt.scatter(x[0], x[1], s=5, marker='o', c=f'C{color}')
        plt.text(x[0], x[1], country, color=f'C{color}')

    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.savefig(output)
    plt.show()

def main():
    file = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(file, binary=True)
    
    sim_file = 'questions-words_similarity.txt'
    countries = get_countries_list(sim_file)
    countries_vec = [model[country] for country in countries]

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(countries_vec)
    show_result69(kmeans, countries_vec, countries, 'result69.png')

if __name__ == '__main__':
    main()