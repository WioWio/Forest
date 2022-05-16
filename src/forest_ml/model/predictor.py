from pathlib import Path
from typing import Any
import joblib
import pandas as pd
import click

from ..data import get_features, split_id


@click.command()
@click.option(
    "-m",
    "--model-path",
    default = "src/forest_ml/model/model.joblib",
    type = Path,
    show_default = True
)
@click.option(
    "-t",
    "--test-path",
    default = "data/test.csv",
    #default = "tests/datasamples.csv",
    type = Path,
    show_default = True
)
@click.option(
    "-s",
    "--save-path",
    default = "data/predictions.csv",
    type = Path,
    show_default = True
)
def predict(
        model_path: Path,
        test_path: Path,
        save_path: Path
) -> None:
    model = joblib.load(model_path)
    features = get_features(test_path)
    features, ids = split_id(features)
    predictions = model.predict(features)
    predictions = pd.Series(predictions, index=ids, name='Cover_Type')
    predictions.to_csv(save_path)
    click.echo(f"Predictions saved to {save_path}")
