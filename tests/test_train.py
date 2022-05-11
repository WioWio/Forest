import os
from pathlib import Path
import pandas as pd
import pytest

from click.testing import CliRunner

from forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


@pytest.fixture
def save_path() -> Path:
    return 'model.joblib'


@pytest.fixture
def load_path() -> Path:
    return 'datasamples.csv'


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.read_csv('tests/datasamples.csv')

def test_logreg_c(
    runner: CliRunner,
    data: pd.DataFrame,
    load_path: Path,
    save_path: Path
) -> None:
    """It fails when log_reg_c is not btw. 0 and 1."""
    with runner.isolated_filesystem():
        data.to_csv(load_path)
        result = runner.invoke(train, ['-s', save_path,
                                       '-d', load_path,
                                       "--logreg-c", 6])
        assert result.exit_code == 2
        assert "Invalid value for '--logreg-c'" in result.output

        result = runner.invoke(train, ['-s', save_path,
                                       '-d', load_path,
                                       "--logreg-c", 0])
        assert result.exit_code == 0
        assert "Model is saved" in result.output

def test_classifier(
    runner: CliRunner,
    data: pd.DataFrame,
    load_path: Path,
    save_path: Path
) -> None:
    with runner.isolated_filesystem(): 
        data.to_csv(load_path)
        result = runner.invoke(train, ['-s', save_path,
                                       '-d', load_path,
                                       "--classifier", 'Forest']) 
        assert result.exit_code == 2
        assert "Invalid value for '--classifier'" in result.output

        result = runner.invoke(train, ['-s', save_path,
                                       '-d', load_path,
                                       "--classifier", 'LogReg'])
        assert result.exit_code == 0
        assert "Model is saved" in result.output





def test_create_model(
        runner: CliRunner,
        data: pd.DataFrame,
        load_path: Path,
        save_path: Path
        ):
    with runner.isolated_filesystem():
        data.to_csv(load_path)
        assert not os.path.exists(save_path)
        runner.invoke(train,
                      ['-s', save_path,
                       '-d', load_path])
        assert os.path.exists(save_path)