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

from .params_searcher import search_best_model
from .pipeline import create_pipeline
from .data import get_dataset


def get_metrics(pipeline, classifier, X, y, n_splits: int) -> list:
    X = X.to_numpy()
    accuracy, mse, loss, v_score = [], [], [], []
    splits = KFold(n_splits=n_splits)
    for train_i, test_i in splits.split(X):
        X_train,y_train = X[train_i], y[train_i]
        X_test,y_test = X[test_i],  y[test_i]
        if pipeline is None:
            fitted_model = search_best_model(X_train, y_train, classifier)
        else:
            fitted_model = pipeline.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)
        y_prob = fitted_model.predict_proba(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        loss.append(log_loss(y_test, y_prob))
        v_score.append(v_measure_score(y_test, y_pred))
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
    "--use-nested-cv",
    default=True,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_neighbors: int,
    classifier: str,
    selector: str,
    pca_components: int,
    use_nested_cv: bool
) -> None:
    features, target = get_dataset(dataset_path)
    with mlflow.start_run():
        if use_nested_cv:
            metrics = get_metrics(None, classifier, features, target, n_splits=5)
            model = search_best_model(features, target, classifier)
        else:
            model = create_pipeline(
                classifier, selector, pca_components, 
                use_scaler, logreg_c, max_iter, n_neighbors
            )
            metrics = get_metrics(model, classifier, features, target, n_splits=5)
        mlflow.log_param('model', classifier)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_nested_cv", use_nested_cv)
        if selector:
             mlflow.log_param("selector", selector)
             if selector == "PCA":
                mlflow.log_param("n_components", pca_components)       
        if not use_nested_cv:
            mlflow.log_param("max_iter", max_iter)
            if classifier == "LogReg":
                mlflow.log_param("logreg_c", logreg_c)
            if classifier == "K-Neighbors":
                mlflow.log_param("n_neighbors", n_neighbors)
        for metric in metrics:
            mlflow.log_metric(metric[0], metric[1])
            click.echo(f"{metric[0]}: {metric[1]}")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")


train()
