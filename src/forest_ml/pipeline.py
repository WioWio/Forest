from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engin import select_features


def create_pipeline(
    classifier: str,
    selector: str = '',
    pca_components: int = 0,
    alpha: float = 1,
    use_scaler: bool = True,
    logreg_C: float = 1.0,
    max_iter: int = 1000,
    n_neighbors: int = 0,
    max_depth: int = 10,
    random_state: int = 42,
) -> Pipeline:
    steps = []
    if selector != '':
        steps.append(("selector", select_features(selector, pca_components, alpha)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == "K-Neighbors":
        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     weights='distance')
    elif classifier == "LogReg":
        model = LogisticRegression(C=logreg_C,
                                   max_iter=max_iter,
                                   random_state=random_state
                )
    elif classifier == "Forest":
        model = RandomForestClassifier(max_depth=max_depth)
    elif classifier == "Trees":
        model = ExtraTreesClassifier(random_state=random_state)
    steps.append(("classifier", model))
    return Pipeline(steps=steps)
