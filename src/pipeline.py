from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline_k_means(
    use_scaler:bool,n_clusters:int
)->Pipeline:
    steps = []
    if use_scaler:
        steps.append(('scaler',StandardScaler()))
    steps.append(('classifier',KMeans(n_clusters=n_clusters)))
    return Pipeline(steps=steps)

def create_pipeline_log_reg(
    use_scaler:bool, logreg_C:float, max_iter:int
)->Pipeline:
    steps = []
    if use_scaler:
        steps.append(('scaler',StandardScaler()))
    steps.append(('classifier',LogisticRegression(C=logreg_C,max_iter=max_iter)))
    return Pipeline(steps=steps)