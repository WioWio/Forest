import os
from pathlib import Path
from typing import Any
import click
import joblib
import pytest
import pandas as pd
from click.testing import CliRunner

from forest_ml.model.predictor import predict


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


@pytest.fixture
def save_path() -> Path:
    return "predictions.csv"


@pytest.fixture
def model() -> Any: 
    return joblib.load("src/forest_ml/model/model.joblib")


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.read_csv("tests/datasamples.csv",index_col='Id')


@pytest.fixture
def test_path() -> str:
    return "test.csv"


@pytest.fixture
def model_path() -> str:
    return "model.joblib"

def test_predictor(
        runner: CliRunner,
        save_path: Path,
        model:Any,
        test_data: pd.DataFrame,
        test_path: str,
        model_path: str
) -> None:
    with runner.isolated_filesystem():
        test_data.to_csv(test_path)
        assert test_data.shape[1] == 55
        joblib.dump(model, model_path)
        assert not os.path.exists(save_path)
        result = runner.invoke(predict, ["-s", save_path, "-t", test_path, "-m", model_path])
        assert result.exit_code == 0
        assert 'saved' in result.output
        assert os.path.exists(save_path)
        preds = pd.read_csv(save_path, index_col='Id')
        assert preds.shape[1] == 1
        assert preds.shape[0] == test_data.shape[0]
        assert preds.values.all() in range(1,7)
        assert preds.index.all() == test_data.index.all()
        