import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

import dataload

def get_kmeans_clusters(data):
    model = KMeans()
    clusters = model.fit_predict(data)
    return clusters

def get_matrix_factorization(data):
    model = NMF()
    W = model.fit_transform(data)
    print "Number of iterations:", model.n_iter_
    return W

if __name__ == "__main__":
    data = dataload.get_tfidf()
    print get_matrix_factorization(data)
