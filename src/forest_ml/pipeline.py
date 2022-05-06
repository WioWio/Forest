from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forest_ml.feature_engin import select_features


def create_pipeline(classifier: str, selector: str, use_scaler: bool, logreg_C: float = None, max_iter: int = None, n_clusters: int= None) -> Pipeline:
    steps = []
    if selector != 'None':
        steps.append(("selector", select_features(selector)))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == 'K-Means':
        steps.append(("classifier", KMeans(n_clusters=n_clusters, max_iter=max_iter)))
    elif classifier == 'LogReg':
        steps.append(("classifier", LogisticRegression(C=logreg_C, max_iter=max_iter)))
    return Pipeline(steps=steps)
