from click.testing import CliRunner
import pytest

from forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_log_reg_c(
    runner: CliRunner
) -> None:
    """It fails when log_reg_c is not btw. 0 and 1."""
    result = runner.invoke(train, ["--logreg-c", 6])
    assert result.exit_code == 2
    assert "Invalid value for '--logreg-c'" in result.output

    result = runner.invoke(train, ["--logreg-c", 0])
    assert result.exit_code == 0
    assert "Model is saved" in result.output

def test_classifier(
    runner: CliRunner
) -> None:
    result = runner.invoke(train, ["--classifier", 'Forest']) 
    assert result.exit_code == 2
    assert "Invalid value for '--classifier'" in result.output

    result = runner.invoke(train, ["--classifier", 'LogReg'])
    assert result.exit_code == 0
    assert "Model is saved" in result.output