from minisom import MiniSom
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# open the file with text and read the whole file, then split the text into words
with open('input_text.txt', 'r') as input_file:
    input_text = input_file.read()
words = input_text.split()


# we have to transform our words into vectors
vectorizer = TfidfVectorizer()
data = vectorizer.fit_transform(words).toarray()


# Initialization and training (we will have a 1 x 6 SOM)
som_shape = (1, 6)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.9, learning_rate=0.9,
              neighborhood_function='gaussian')
som.train_batch(data, 10000, verbose=True)


# get the winner
winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, (som_shape[0], som_shape[1]))


# now get the number of words for every cluster
number_of_words_per_cluster = {}
for cluster in range(som_shape[1]):
    for index in iter(cluster_index):
        if index == cluster:
            if cluster not in number_of_words_per_cluster:
                number_of_words_per_cluster[cluster] = 0
            number_of_words_per_cluster[cluster] += 1

# Plot histogram
plt.figure(figsize=(6, 6))
plt.title("Word frequency in every cluster")
plt.xlabel('Cluster index')
plt.ylabel('Word frequency')
plt.bar(list(number_of_words_per_cluster.keys()), list(number_of_words_per_cluster.values()), color='orange')
plt.xticks(list(number_of_words_per_cluster.keys()))
plt.show()


# Create a dict to store all words for every cluster
cluster_words = {}
for cluster in range(som_shape[1]):
    cluster_words[cluster] = []

# Adding words
for i in range(len(cluster_index)):
    cluster = cluster_index[i]
    cluster_words[cluster].append(words[i])

# Converting it to table and printing it
table = tabulate(cluster_words, tablefmt="pipe", floatfmt=".3f")
print(table)
