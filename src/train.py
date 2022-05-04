from pathlib import Path
from joblib import dump

import click
#import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from pipeline import create_pipeline_k_means, create_pipeline_log_reg
from data import get_dataset


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
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
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
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_clusters: int,
    classifier:str
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    if classifier=="K-Means":
        pipeline = create_pipeline_k_means(use_scaler,n_clusters)
    else:
        pipeline = create_pipeline_log_reg(use_scaler,logreg_c,max_iter)
    pipeline.fit(features_train, target_train)
    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")

train()