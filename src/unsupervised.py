import pandas as pd
from sklearn.cluster import KMeans

import dataload

if __name__ == "__main__":
    data = dataload.get_tfidf()
    model = KMeans()
    clusters = model.fit_predict(data)
    print clusters
    


