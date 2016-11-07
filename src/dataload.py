import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
    return vectorizer, vectorizer.transform(corpus).todense()
