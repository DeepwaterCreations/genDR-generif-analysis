import cPickle

import pandas as pd
import sklearn.model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

import dataload

if __name__ == "__main__":
    pipeline = dataload.get_pipeline(MultinomialNB())
    X, y = dataload.get_subset_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            stratify = y)
    gridsearch = GridSearchCV(pipeline, {
        'tfidfvectorizer__max_df':(0.01, 0.1, 0.2, 0.5, 0.8, 1.0),
        # 'tfidfvectorizer__min_df':(0.0, 0.05, 0.1),
        'tfidfvectorizer__ngram_range':((1,1), (1,2)),
        'tfidfvectorizer__lowercase':(False, True),
        'tfidfvectorizer__norm':('l1', 'l2'),
        'multinomialnb__alpha':(0.0, 0.001, 0.01, 0.25, 0.5, 0.75, 1.0)
        }, n_jobs=7, verbose=1)
    gridsearch.fit(X_train, y_train)
    print "SCORE:{0}".format(gridsearch.score(X_test, y_test))
    print "Best Estimator:{0}".format(gridsearch.best_estimator_)
    print "Best Params:{0}".format(gridsearch.best_params_)
    with open("data/gridsearch.pckl", 'w') as p_file:
        cPickle.dump(gridsearch, p_file)