from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engin import select_features


def create_pipeline(
    classifier: str,
    selector: str = None,
    pca_components: int = None,
    use_scaler: bool = True,
    logreg_C: float = None,
    max_iter: int = None,
    n_neighbors: int = None,
    random_state: int = None,
) -> Pipeline:
    steps = []
    if selector is not None:
        steps.append(("selector", select_features(selector, pca_components)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == "K-Neighbors":
        steps.append(("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)))
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
