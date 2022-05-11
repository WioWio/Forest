from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engin import select_features


def create_pipeline(
    classifier: str,
    selector: str = '',
    pca_components: int = 0,
    use_scaler: bool = True,
    logreg_C: float = 1,
    max_iter: int = 0,
    n_neighbors: int = 0,
    random_state: int = 42,
) -> Pipeline:
    steps = []
    if selector != '':
        steps.append(("selector", select_features(selector, pca_components)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == "K-Neighbors":
        steps.append(("classifier", 
                      KNeighborsClassifier(n_neighbors=n_neighbors)))
    elif classifier == "LogReg":
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    C=logreg_C, max_iter=max_iter, random_state=random_state
                ),
            )
        )
    return Pipeline(steps=steps)
