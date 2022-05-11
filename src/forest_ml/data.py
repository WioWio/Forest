from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
        csv_path: Path = None,
        dataset: pd.DataFrame = None
        ) -> Tuple[pd.DataFrame, pd.Series]:
    if csv_path:
        dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(["Id", "Cover_Type"], axis=1)
    target = dataset["Cover_Type"]
    return features, target