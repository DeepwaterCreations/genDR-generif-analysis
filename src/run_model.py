import cPickle

import numpy as np

import dataload
from build_model import MODEL_FILEPATH

PREDICTIONS_FILEPATH = "data/predictions.csv"

def get_top_words(model, n=50):
    """Return a list of the top n most important words found by the model"""
    coefs = model.named_steps['multinomialnb'].coef_
    max_indices = coefs[0].argsort()
    featurenames = model.named_steps['tfidfvectorizer'].get_feature_names()
    return np.array(featurenames)[max_indices][:-(n+1):-1]

if __name__ == "__main__":
    with open(MODEL_FILEPATH) as model_file:
        model = cPickle.load(model_file)
    df = dataload.get_aws_df()
    df = df[df['in_genDR'] == 0]
    
    corpus = df["GeneRIF text"]

    predictions = model.predict(corpus)

    df['dr_relevant'] = predictions

    df.to_csv(PREDICTIONS_FILEPATH)
