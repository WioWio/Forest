from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    use_scaler:bool
)->Pipeline:
    steps = []
    if use_scaler:
        steps.append(('scaler',StandardScaler()))
    return Pipeline(steps=steps)