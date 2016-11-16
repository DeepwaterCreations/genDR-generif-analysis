import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.pipeline
import sklearn.utils

def get_gendr_rifs_df():
    """Returns a data frame of GeneRIFs and related data for genes that are part 
    of the GenDR database.
    """
    df = get_labeled_df()
    return df[df['in_genDR'] == 1]

def get_labeled_df():
    """Returns a data frame with GeneRIF data labeled according to membership
    in the GenDR database.
    """
    df = pd.read_csv('data/gendr_rifs.csv')
    return df

def get_aws_df():
    """Loads a labeled data frame from an AWS S3 bucket."""
    df = pd.read_csv("https://s3-us-west-2.amazonaws.com/dc.galvanize.capstone.generifdata/gendr_rifs.csv")
    return df

def get_tfidf(corpus=None):
    """Returns a TfIdf vectorization of the provided corpus and the model used to generate it.
    If no corpus is provided, uses the GeneRIF data from GenDR genes.
    """
    if corpus is None:
        data = get_gendr_rifs_df()
        corpus = data['GeneRIF text']
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.transform(corpus)

def get_subset_data():
    """Combines the GeneRIF data for GenDR genes with a random sampling of the 
    GeneRIF data for genes not part of GenDR, shuffles the data,
    and returns the RIF and the labels separately.
    """
    df = get_labeled_df()
    #I have orders of magnitude more geneRIFs that aren't for genDR genes than ones that are,
    #so I want to take a subset of the 'false' values and only train/test on those.
    subset_rows = np.random.choice(df[df['in_genDR'] == 0].index.values, df.shape[0]/500)
    subset_rows = np.concatenate((df[df['in_genDR'] == 1].index.values, subset_rows))
    df = df.iloc[subset_rows]
    sklearn.utils.shuffle(df)

    y = df['in_genDR']
    corpus = df['GeneRIF text']

    return corpus, y

def get_pipeline(model):
    """Returns a pipeline object that combines the 
    provided model with a TfIdf vectorizer.
    """
    pipeline = sklearn.pipeline.make_pipeline(
                TfidfVectorizer(stop_words='english'),
                model
            )
    return pipeline
