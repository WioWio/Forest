from click.testing import CliRunner
import pytest

from forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_log_reg_c(
    runner: CliRunner
) -> None:
    """It fails when log_reg_c is not btw. 0 and 1."""
    result = runner.invoke( train, ["--logreg-c", 6])
    assert result.exit_code == 2
    assert "Invalid value for '--logreg-c'" in result.output