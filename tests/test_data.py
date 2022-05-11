import pytest
from pathlib import Path
import pandas as pd

from forest_ml.data import get_dataset


@pytest.fixture
def path() -> Path:
    return Path("tests\datasamples.csv")


def test_get_dataset(path: Path) -> None:
    features, target = get_dataset(path)

    assert features.shape[1] == 54
    assert type(target) == pd.Series
    assert features.shape[0] == target.shape[0]
    assert target.values.all() in range(1, 7)
    assert target.name == "Cover_Type"
