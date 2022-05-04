#from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    use_scaler:bool,logreg_C:float
)->Pipeline:
    steps = []
    if use_scaler:
        steps.append(('scaler',StandardScaler()))
    #steps.append(('classifier',KMeans(n_clusters=n_clusters)))
    steps.append(('classifier',LogisticRegression(C=logreg_C)))
    return Pipeline(steps=steps)