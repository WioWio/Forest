from typing import Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from boruta import BorutaPy
from sklearn.linear_model import Lasso

import pandas as pd


def select_features(
       selector: str, 
       pca_components: int,
       alpha: float
) -> Any:
    if selector == "Boruta":
        return BorutaPy(
            RandomForestClassifier(max_depth=10),
            n_estimators="auto",
            verbose=0,
            max_iter=10,
            random_state=42,
        )
    elif selector == "PCA":
        return PCA(n_components=pca_components)
    elif selector == "Trees":
        return SelectFromModel(ExtraTreesClassifier(n_estimators=50))
    elif selector == "Lasso":
        return SelectFromModel(Lasso(alpha=alpha))
    return False


def custom_select(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        uniq_count = X[col].value_counts(ascending=True)
        if len(uniq_count) == 1:
            X = X.drop(col, axis=1)
        elif len(uniq_count) == 2:
            rariest = uniq_count.iloc[0]
            if rariest/uniq_count.iloc[1] < 0.01:
                X = X.drop(col, axis=1)
    return X
