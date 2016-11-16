import cPickle
import os.path

import pandas as pd
import sklearn.model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

import dataload

MODEL_FILEPATH = "data/predictive_model{0}.pckl"
SCORES_FILEPATH = "data/model_score_list.txt"

def do_gridsearch(X_train, y_train):
    """Runs a gridsearch over a pipeline containing a TfIdfVectorizer
    and a Multinomial Naive Bayes model
    """
    pipeline = dataload.get_pipeline(MultinomialNB())
    gridsearch = GridSearchCV(pipeline, {
        'tfidfvectorizer__max_df':(0.01, 0.1, 0.2, 0.5, 0.8, 1.0),
        'tfidfvectorizer__min_df':(0.0, 0.001),
        'tfidfvectorizer__max_features':(None, 20, 50, 100, 500, 1000),
        'tfidfvectorizer__ngram_range':((1,1), (1,2), (2,2)),
        'tfidfvectorizer__lowercase':(False, True),
        'tfidfvectorizer__norm':('l1', 'l2'),
        'tfidfvectorizer__sublinear_tf':(False, True),
        'multinomialnb__alpha':(0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
        }, n_jobs=7, verbose=1)
    gridsearch.fit(X_train, y_train)
    return gridsearch

def save_model(gridsearch, score):
    """Save the generated model to a new file, incrementing the filename
    suffix if necessary, and write the accuracy score and parameters to a text file.
    """
    ver_num = 0
    while os.path.isfile(MODEL_FILEPATH.format(ver_num)):
        ver_num += 1
    filename = MODEL_FILEPATH.format(ver_num)
    with open(filename, 'w') as model_file:
        cPickle.dump(gridsearch.best_estimator_, model_file)
    with open(SCORES_FILEPATH, 'a') as scores_file:
        scores_file.write("\n{0} | {1} | {2}".format(filename,
                                                score,
                                                gridsearch.best_params_))
        

if __name__ == "__main__":
    X, y = dataload.get_subset_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            stratify = y)
    gridsearch = do_gridsearch(X_train, y_train)
    print "SCORE:{0}".format(gridsearch.best_estimator_.score(X_test, y_test))
    print "Best Estimator:{0}".format(gridsearch.best_estimator_)
    print "Best Params:{0}".format(gridsearch.best_params_)

    with open(MODEL_FILEPATH, 'w') as model_file:
        cPickle.dump(gridsearch.best_estimator_, model_file)
