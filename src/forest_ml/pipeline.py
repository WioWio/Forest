from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(classifier: str, use_scaler: bool, logreg_C: float = None, max_iter: int = None, n_clusters: int= None) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if classifier == 'K-Means':
        steps.append(("classifier", KMeans(n_clusters=n_clusters, max_iter=max_iter)))
    elif classifier == 'LogReg':
        steps.append(("classifier", LogisticRegression(C=logreg_C, max_iter=max_iter)))
    return Pipeline(steps=steps)
