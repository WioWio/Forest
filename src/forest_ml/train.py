from pathlib import Path
from typing import Any, Tuple
from joblib import dump

import click

import mlflow
import numpy as np
import pandas as pd
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
from .feature_engin import custom_select


def get_cv_metrics(
    pipeline: Any,
    classifier: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    selector: str = '',
    random_state: int = 42
) -> list[Any]:
    X = X.to_numpy()
    accuracy, mse, loss, v_score = [], [], [], []
    splits = KFold(n_splits=n_splits)
    for train_i, test_i in splits.split(X):
        X_train, y_train = X[train_i], y[train_i]
        X_test, y_test = X[test_i], y[test_i]
        if pipeline is None:
            fitted_model, _ = search_best_model(
                X_train, y_train, classifier, selector, random_state
            )
        else:
            fitted_model = pipeline.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)
        y_prob = fitted_model.predict_proba(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        loss.append(log_loss(y_test, y_prob))
        v_score.append(v_measure_score(y_test, y_pred))
    metrics = get_mean_metrics(accuracy, mse, loss, v_score)
    return metrics


def get_mean_metrics(
        accuracy: list[float], 
        mse:list[float], 
        loss: list[float], 
        v_score: list[float]
    ) -> list[Any]:
    accuracy, mse, loss, v_score = (
        np.mean(accuracy),
        np.mean(mse),
        np.mean(loss),
        np.mean(v_score),
    )
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
    default=Path("data/forest_data.csv"),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default=Path("src/forest_ml/model/model.joblib"),
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
    type=click.FloatRange(0, 1.0),
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
    default="Trees",
    type=click.Choice(["K-Neighbors", "LogReg", "Forest", "Trees"]),
    show_default=True,
)
@click.option(
    "--selector",
    default='Custom',
    type=click.Choice(["", "PCA", "Boruta", "Trees", "Lasso", "Custom"]),
    show_default=True,
)
@click.option(
    "--pca-components",
    default=2,
    type=int,
    show_default=True,
)
@click.option(
    "--alpha",
    default=1,
    type=float,
    show_default=True,
)
@click.option(
    "--use-nested-cv",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-splits",
    default=5,
    type=int,
    show_default=True,
)
def train(
    dataset_path: str,
    save_model_path: str,
    random_state: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_neighbors: int,
    classifier: str,
    selector: str,
    pca_components: int,
    alpha: float,
    use_nested_cv: bool,
    n_splits: int
) -> None:
    features, target = get_dataset(dataset_path)
    if selector == "Custom":
        selector = ''
        features = custom_select(features)
    with mlflow.start_run(run_name=classifier):
        if use_nested_cv:
            metrics = get_cv_metrics(
                None, classifier, features, target,
                n_splits, selector, random_state
            )
            model, params = search_best_model(
                features, target, classifier, selector, random_state
            )
            click.echo(get_cv_metrics(model, classifier,features, target, n_splits))
            click.echo(f"Best parms: {params}")
        else:
            model = create_pipeline(
                classifier,
                selector,
                pca_components,
                alpha,
                use_scaler,
                logreg_c,
                max_iter,
                n_neighbors,
                random_state,
            )
            metrics = get_cv_metrics(model, classifier,
                                  features, target, n_splits)
        mlflow.log_param("model", classifier)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_nested_cv", use_nested_cv)
        mlflow.log_param("selector", selector)
        if not use_nested_cv:
            if selector == "PCA":
                mlflow.log_param("n_components", pca_components)
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
