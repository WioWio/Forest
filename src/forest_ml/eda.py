from pandas_profiling import ProfileReport
import pandas as pd

from pathlib import Path
from joblib import dump

import click


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
    "--save-path",
    default="data/eda_report.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def generate_eda(dataset_path: Path, save_path: Path):
    dataset = pd.read_csv(dataset_path)
    report = ProfileReport(dataset, title="Forest Pandas Profiling Report")
    report.to_file(save_path)
    click.echo(f"Eda report is saved to {save_path}.")
