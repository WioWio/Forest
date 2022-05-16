from typing import Any
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV

from .pipeline import create_pipeline


def search_best_model(
        X: pd.DataFrame,
        y: pd.Series,
        classifier: str,
        selector: str,
        random_state: int
) -> Any:
    space = create_space(classifier, selector)
    estim = create_pipeline(classifier, selector, use_scaler=True, random_state=random_state)
    searcher = RandomizedSearchCV(
        estim, space, scoring="accuracy", random_state=random_state, cv=5
    )
    searcher.fit(X, y)
    return searcher.best_estimator_, searcher.best_params_


def create_space(
        classifier: str,
        selector: str
) -> dict[str,list[Any]]:
    space = dict()
    if classifier == "K-Neighbors":
        space["classifier__n_neighbors"] = [n for n in range(1, 40, 1)]
    elif classifier == "LogReg":
        space["classifier__C"] = [0.1*n for n in range(1,10)]
    elif classifier == "Forest":
        space["classifier__max_depth"] = [n for n in range(2, 15, 2)]
        space["classifier__n_estimators"] = [n for n in range(100, 600, 100)]
        space["classifier__min_samples_split"] = [n for n in range(2, 15, 2)]
    elif classifier == "Trees":
        space["classifier__n_estimators"] = [n for n in range(100, 600, 100)]
        space["classifier__criterion"] = ["gini", "entropy"]
        space["classifier__max_depth"] = [n for n in range(2, 15, 2)]
    if selector == "PCA":
        space["selector__n_components"] = [n for n in range(1, 30, 1)]
    if selector == "Lasso":
        space["selector__alpha"] = [0.1*n for n in range(1,10)]
    return space
