from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
        csv_path: str
        ) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(["Cover_Type", "Id"], axis=1)
    target = dataset["Cover_Type"]
    return features, target


def get_features(
        csv_path: Path
) -> pd.DataFrame:
    features = pd.read_csv(csv_path)
    if 'Cover_Type' in features.columns:
        click.echo("Dataset has target values")
        features = features.drop("Cover_Type", axis=1)
    click.echo(f"Dataset shape: {features.shape}.")
    return features


def split_id(
        df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop('Id', axis=1), df['Id']
