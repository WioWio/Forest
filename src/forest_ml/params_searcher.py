from sklearn.model_selection import RandomizedSearchCV

from .pipeline import create_pipeline


def search_best_model(X, y, classifier, selector, random_state):
    space = dict()
    if classifier == 'K-Neighbors':
        space['classifier__n_neighbors'] = range(1, 40)
        space['classifier__weights'] = ['distance', 'uniform']
    else:
        space['classifier__logreg_c'] = range(0, 1, 0.1)
        space['classifier__max_iter'] = [100,500,1000]
    if selector == 'PCA':
        space['selector__n_components'] = range(1,30)
    estim = create_pipeline(classifier, selector, random_state=random_state)
    searcher = RandomizedSearchCV(estim, space, scoring='accuracy', 
                                  random_state=random_state)
    searcher.fit(X, y)
    return searcher.best_estimator_