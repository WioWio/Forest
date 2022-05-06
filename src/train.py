from pathlib import Path
from joblib import dump

import click
#import mlflow
#import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, v_measure_score, mean_squared_error
from sklearn.model_selection import KFold

from pipeline import create_pipeline_k_means, create_pipeline_log_reg
from data import get_dataset

def get_metrics(
    classifier:str, model, X, y, n_splits:int, random_state:int
    ) -> list:
    X = X.to_numpy()
    accuracy, mse, v_score = [],[],[]
    splits = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_i, test_i in splits.split(X):
        y_pred = model.fit(X[train_i],y[train_i]).predict(X[test_i])
        if classifier=="K-Means":
            y_pred+=1 #because k-means give labels starting from 0, not 1 like in dataset
        accuracy.append(accuracy_score(y[test_i],y_pred))
        mse.append(mean_squared_error(y[test_i],y_pred))
        v_score.append(v_measure_score(y[test_i],y_pred))
    accuracy, mse, v_score = np.mean(accuracy), np.mean(mse), np.mean(v_score)
    metrics = [("Accuracy",accuracy),("Mean Squared Error",mse),("V-Score",v_score)]
    return metrics


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="../data/forest_data.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="../data/model.joblib",
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
    default=100,
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
    "--n-clusters",
    default=7,
    type=int,
    show_default=True,
)
@click.option(
    "--classifier",
    default="K-Means",
    type=str,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    #test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_clusters: int,
    classifier:str
) -> None:
    features, target = get_dataset(dataset_path)
    if classifier=="K-Means":
        pipeline = create_pipeline_k_means(use_scaler,n_clusters)
    else:
        pipeline = create_pipeline_log_reg(use_scaler,logreg_c,max_iter)
    metrics = get_metrics(classifier, pipeline, features, target, 5, random_state)
    for metric in metrics:
        click.echo(f"{metric[0]}: {metric[1]}")
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")

train()