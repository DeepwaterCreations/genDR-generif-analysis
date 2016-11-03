import pandas as pd
import numpy as np
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
    return model, W

def get_top_words(model, feature_names, num_words=5):
    topics = []
    for topic in model.components_:
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(topic_words)
    return topics

if __name__ == "__main__":
    vectorizer, vectors = dataload.get_tfidf()
    model, W = get_matrix_factorization(vectors)
    print W.shape
    feature_names = np.array(vectorizer.get_feature_names())
    print get_top_words(model, feature_names)
