import cPickle
import sys
import os.path

import numpy as np
from wordcloud import WordCloud

import dataload

PREDICTIONS_FILEPATH = "data/predictions.csv"
WORDCLOUD_FILEPATH = "data/topwords.png"

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
    if len(sys.argv) < 2:
        sys.exit("USAGE: %s filepath-to-pickled-model" % sys.argv[0])

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        sys.exit("Error: File '%s' not found" % filepath)

    with open(filepath) as model_file:
        model = cPickle.load(model_file)

    df = dataload.get_labeled_df()
    df = df[df['in_genDR'] == 0]
    corpus = df["GeneRIF text"]

    predictions = model.predict(corpus)
    df['dr_relevant'] = predictions

    print "Top Words:"
    top_words = get_top_words(model, n=100)
    print ", ".join(top_words) 
    
    save_wordcloud = raw_input("Save wordcloud? (y/N)").lower()
    if save_wordcloud == 'y':
        coefs = model.named_steps['multinomialnb'].coef_[0]
        featurenames = model.named_steps['tfidfvectorizer'].get_feature_names()
        get_top_words_cloud(featurenames, coefs)
        print "Saved as {0}".format(WORDCLOUD_FILEPATH)
    save_csv = raw_input("Save csv? (y/N)").lower()
    if save_csv == 'y':
        df.to_csv(PREDICTIONS_FILEPATH)
        print "Saved as {0}".format(PREDICTIONS_FILEPATH)
    
