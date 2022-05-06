from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


def select_features(selector: str):
    if selector == 'Boruta':
        return BorutaPy(RandomForestClassifier(max_depth=10), 
                         n_estimators='auto', 
                         verbose=0, 
                         max_iter=100,
                         random_state=42)
    if selector == 'PCA':
        return PCA(n_components=2)
    return False