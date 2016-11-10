import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.pipeline
import sklearn.utils

def get_gendr_rifs_df():
    df = get_labeled_df()
    return df[df['in_genDR'] == 1]

def get_labeled_df():
    df = pd.read_csv('data/gendr_rifs.csv')
    return df

def get_tfidf(corpus=None):
    if corpus is None:
        data = get_gendr_rifs_df()
        corpus = data['GeneRIF text']
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.transform(corpus)

def get_subset_data():
    df = get_labeled_df()
    #I have orders of magnitude more geneRIFs that aren't for genDR genes than ones that are,
    #so I want to take a subset of the 'false' values and only train/test on those.
    subset_rows = np.random.choice(df[df['in_genDR'] == 0].index.values, df.shape[0]/100)
    subset_rows = np.concatenate((df[df['in_genDR'] == 1].index.values, subset_rows))
    df = df.iloc[subset_rows]
    sklearn.utils.shuffle(df)

    y = df['in_genDR']
    corpus = df['GeneRIF text']

    return corpus, y

def get_pipeline(model):
    pipeline = sklearn.pipeline.make_pipeline(
                TfidfVectorizer(stop_words='english'),
                model
            )
    return pipeline
