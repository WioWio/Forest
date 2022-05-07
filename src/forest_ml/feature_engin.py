from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from boruta import BorutaPy
from sklearn.linear_model import Lasso


def select_features(selector: str, pca_components: int):
    if selector == "Boruta":
        return BorutaPy(
            RandomForestClassifier(max_depth=10),
            n_estimators="auto",
            verbose=0,
            max_iter=100,
            random_state=42,
        )
    if selector == "PCA":
        return PCA(n_components=pca_components)
    if selector == "Trees":
        return SelectFromModel(ExtraTreesClassifier(n_estimators=50))
    if selector == "Lasso":
        return SelectFromModel(Lasso())
    return False
