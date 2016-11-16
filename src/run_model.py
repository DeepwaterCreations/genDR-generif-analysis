import cPickle

import numpy as np
from wordcloud import WordCloud

import dataload
from build_model import MODEL_FILEPATH

PREDICTIONS_FILEPATH = "data/predictions.csv"

def get_top_words(model, n=50):
    """Return a list of the top n most important words found by the model"""
    coefs = model.named_steps['multinomialnb'].coef_
    max_indices = coefs[0].argsort()
    featurenames = model.named_steps['tfidfvectorizer'].get_feature_names()
    return np.array(featurenames)[max_indices][:-(n+1):-1]

def get_top_words_cloud(featurenames, coefficients, num_words=100):
    weights = zip(featurenames, coefficients)
    weights.sort(key=lambda x:x[1])
    wc = WordCloud(background_color="white", 
            ranks_only=True,
            max_font_size=44,
            prefer_horizontal=1.0,
            width=800,
            height=400,
            max_words=100)
    wc.fit_words(weights[:-101:-1])
    wc.to_file('topwords.png')

if __name__ == "__main__":
    with open(MODEL_FILEPATH) as model_file:
        model = cPickle.load(model_file)
    df = dataload.get_aws_df()
    df = df[df['in_genDR'] == 0]
    
    corpus = df["GeneRIF text"]

    predictions = model.predict(corpus)

    df['dr_relevant'] = predictions

    df.to_csv(PREDICTIONS_FILEPATH)
