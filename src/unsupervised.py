import collections

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from wordcloud import WordCloud

import dataload

def get_kmeans_clusters(data):
    """ Fits a KMeans model to the data
    and returns the fitted model
    """
    model = KMeans()
    clusters = model.fit_predict(data)
    return clusters

def get_matrix_factorization(data):
    """Fits a non-negative matrix factorization to the data
        Returns the model and the output matrix
        """
    model = NMF(n_components = 20)
    W = model.fit_transform(data)
    print "Number of iterations:", model.n_iter_
    #W: Matrix of samples to components
    #components_: Matrix of components to features, AKA H
    return model, W

def get_top_words(model, feature_names, num_words=5):
    """Returns a list of lists, where each sublist is the top num_words most
    important feature words for that component.
    """
    topics = []
    for topic in model.components_:
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(topic_words)
    return topics

def get_categorized_rifs(W, data):
    """Calculates, for each data row, which component in W it's most strongly associated with,
    and returns a dictionary mapping components to lists of assocated data rows.
    """
    component_datapoints = collections.defaultdict(list)
    for i, sample in enumerate(W):
        best_category_idx = sample.argmax()
        datarow = data.iloc[i]
        component_datapoints[best_category_idx].append(datarow)
    return component_datapoints


def build_wordcloud_files(vectorizer, model):
    """Generates a word cloud image for each component
    displaying the most important word-features in that component, and saves
    them to separate files.
    """
    wc = WordCloud()
    featurenames = vectorizer.get_feature_names()
    for i, component in enumerate(model.components_):
        weights = zip(featurenames, component)
        weights.sort(key=lambda x:x[1])
        wc.fit_words(weights[:-201:-1])
        wc.to_file('wordcloud{0}.png'.format(i))


if __name__ == "__main__":
    vectorizer, vectors = dataload.get_tfidf()
    vectors = vectors.todense()
    model, W = get_matrix_factorization(vectors)
    print W.shape
    feature_names = np.array(vectorizer.get_feature_names())
    print get_top_words(model, feature_names)
