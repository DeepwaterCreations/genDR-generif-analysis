import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

import dataload

def get_kmeans_clusters(data):
    model = KMeans()
    clusters = model.fit_predict(data)
    return clusters

def get_matrix_factorization(data):
    model = NMF(n_components = 20)
    W = model.fit_transform(data)
    print "Number of iterations:", model.n_iter_
    return W

def get_top_words(model, feature_names, num_words=5):
    topics = []
    for topic in model.components_:
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(topic_words)
    return topics

if __name__ == "__main__":
    data = dataload.get_tfidf()
    print get_matrix_factorization(data)
