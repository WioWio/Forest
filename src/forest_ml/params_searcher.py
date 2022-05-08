from sklearn.model_selection import RandomizedSearchCV

from .pipeline import create_pipeline


def search_best_model(X, y, classifier):
    space = dict()
    if classifier == 'K-Neighbors':
        space['classifier__n_neighbors'] = [2, 3, 5]
    else:
        space['classifier__logreg_c'] = [0, 0.1, 0.2, 0.5, 1]
    estim = create_pipeline(classifier=classifier)
    searcher = RandomizedSearchCV(estim, space, scoring='accuracy')
    searcher.fit(X, y)
    return searcher.best_estimator_