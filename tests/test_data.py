import pytest
from pathlib import Path
import pandas as pd

from forest_ml.data import get_dataset, get_features, split_id


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


def test_split_fuse_id(path: Path) -> None:
    features = get_features(path)
    assert 'Cover_Type' not in features.columns
    assert features.shape[1] == 55
    features, ids = split_id(features)
    assert type(features) == pd.DataFrame
    assert type(ids) == pd.Series
    assert features.shape[0] == ids.shape[0]
    assert features.shape[1] == 54

    #assert type(df) == pd.DataFrame
    #assert df.shape[1] == 55
