import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_gendr_rifs_df():
    df = pd.read_csv('data/gendr_rifs.csv')
    return df

def get_tfidf():
    df = get_gendr_rifs_df()
    vectorizer = TfidfVectorizer()
    corpus = df['GeneRIF text']
    return vectorizer.fit_transform(corpus)
