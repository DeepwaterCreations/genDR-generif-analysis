import cPickle
import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

import dataload

MODEL_FILEPATH = "data/predictive_model{0}.pckl"
SCORES_FILEPATH = "data/model_score_list.txt"
ROC_FILEPATH = "data/roc_curve{0}.svg"

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
    Returns the filename suffix.
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
    return ver_num
        
def gen_roc(y_test, model):
    """Generates a ROC graph for the given model and data labels"""
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1], pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
        
    sns.set_style("dark")
    sns.set_context("talk")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], color="black", linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Area under curve: %0.2f)" % auc)


if __name__ == "__main__":
    X, y = dataload.get_subset_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, 
                                                                            stratify = y)
    gridsearch = do_gridsearch(X_train, y_train)
    score = gridsearch.best_estimator_.score(X_test, y_test)
    print "SCORE:{0}".format(score)
    print "Best Params:{0}".format(gridsearch.best_params_)
    print "Stopwords:"
    print gridsearch.best_estimator_.named_steps['tfidfvectorizer'].stop_words_
    gen_roc(y_test, gridsearch.best_estimator_)
    print "Displaying ROC graph (close to continue)"
    plt.show()

    save = raw_input("Save this model? (Y/n)").lower()
    if save != 'n':
        suffix = save_model(gridsearch, score)
        roc_filepath = ROC_FILEPATH.format(suffix)
        plt.savefig(roc_filepath, bbox_inches="tight")
        savepath = MODEL_FILEPATH.format(suffix)
        print "Saved to {0}".format(savepath)
    else:
        print "Model not saved"
