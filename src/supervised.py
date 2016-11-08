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
    pipeline.fit(X_train, y_train)
    print pipeline.score(X_test, y_test)
