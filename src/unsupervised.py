import pandas as pd
from sklearn.cluster import KMeans

import dataload

def get_kmeans_clusters(data):
    model = KMeans()
    clusters = model.fit_predict(data)
    return clusters

if __name__ == "__main__":
    data = dataload.get_tfidf()
    


