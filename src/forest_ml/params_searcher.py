from typing import Any
from pandas import array
from sklearn.model_selection import RandomizedSearchCV

from .pipeline import create_pipeline


def search_best_model(
        X: array,
        y: array,
        classifier: str,
        selector: str,
        random_state: int
) -> Any:
    space = create_space(classifier, selector)
    estim = create_pipeline(classifier, selector, random_state=random_state)
    searcher = RandomizedSearchCV(
        estim, space, scoring="accuracy", random_state=random_state
    )
    searcher.fit(X, y)
    return searcher.best_estimator_


def create_space(
        classifier: str,
        selector: str
) -> dict[str,list[Any]]:
    space = dict()
    if classifier == "K-Neighbors":
        space["classifier__n_neighbors"] = [n for n in range(1, 40, 1)]
        space["classifier__weights"] = ["distance", "uniform"]
    else:
        space["classifier__logreg_c"] = [0.1*n for n in range(1,10)]
        space["classifier__max_iter"] = [100, 500, 1000]
    if selector == "PCA":
        space["selector__n_components"] = [n for n in range(1, 30, 1)]
    return space
