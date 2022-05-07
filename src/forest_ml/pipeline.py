from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forest_ml.feature_engin import select_features


def create_pipeline(
    classifier: str,
    selector: str,
    use_scaler: bool,
    logreg_C: float = None,
    max_iter: int = None,
    n_neighbors: int = None,
) -> Pipeline:
    steps = []
    if selector is not None:
        steps.append(("selector", select_features(selector)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == "K-Neighbors":
        steps.append(("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)))
    elif classifier == "LogReg":
        steps.append(("classifier", LogisticRegression(C=logreg_C, max_iter=max_iter)))
    return Pipeline(steps=steps)
