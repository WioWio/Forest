from email.policy import default
from pathlib import Path
from joblib import dump

import click

import mlflow
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    v_measure_score,
    mean_squared_error,
)
from sklearn.model_selection import KFold

from forest_ml.params_searcher import search_params

from .pipeline import create_pipeline
from .data import get_dataset


def get_metrics(model, X, y, n_splits: int) -> list:
    X = X.to_numpy()
    accuracy, mse, loss, v_score = [], [], [], []
    splits = KFold(n_splits=n_splits)
    for train_i, test_i in splits.split(X):
        fitted_model = model.fit(X[train_i], y[train_i])
        y_pred = fitted_model.predict(X[test_i])
        y_prob = fitted_model.predict_proba(X[test_i])
        y_true = y[test_i]
        accuracy.append(accuracy_score(y_true, y_pred))
        mse.append(mean_squared_error(y_true, y_pred))
        loss.append(log_loss(y_true, y_prob))
        v_score.append(v_measure_score(y[test_i], y_pred))
    accuracy, mse, loss, v_score = np.mean(accuracy), np.mean(mse), np.mean(loss), np.mean(v_score)
    metrics = [
        ("Accuracy", accuracy),
        ("Mean Squared Error", mse),
        ("Log Loss", loss),
        ("V-Score", v_score),
    ]
    return metrics


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/forest_data.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--classifier",
    default="K-Neighbors",
    type=str,
    show_default=True,
)
@click.option(
    "--selector",
    type=str,
)
@click.option(
    "--pca-components",
    default=2,
    type=int,
    show_default=True,
)
@click.option(
    "--use-search-cv",
    default=True,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    # test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_neighbors: int,
    classifier: str,
    selector: str,
    pca_components: int,
    use_search_cv: bool
) -> None:
    features, target = get_dataset(dataset_path)
    with mlflow.start_run():
        mlflow.log_param('model', classifier)
        if use_search_cv:
            params, best_score = search_params(features, target, classifier)
            for param, value in params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric('accuracy', best_score)    
        else:
            pipeline = create_pipeline(
                classifier, selector, pca_components, 
                use_scaler, logreg_c, max_iter, n_neighbors
            )
            metrics = get_metrics(pipeline, features, target, n_splits=5)
            if selector:
                mlflow.log_param("selector", selector)
                if selector == "PCA":
                    mlflow.log_param("n_components", pca_components)
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("max_iter", max_iter)
            if classifier == "LogReg":
                mlflow.log_param("logreg_c", logreg_c)
            if classifier == "K-Neighbors":
                mlflow.log_param("n_neighbors", n_neighbors)
            for metric in metrics:
                mlflow.log_metric(metric[0], metric[1])
                click.echo(f"{metric[0]}: {metric[1]}")
            dump(pipeline, save_model_path)
            click.echo(f"Model is saved to {save_model_path}.")
