from unittest.mock import patch

import pytest
from hydra.core.hydra_config import HydraConfig


@pytest.fixture
def hydra_config(monkeypatch: pytest.MonkeyPatch):
    class MockJob:
        name = "test_job"

    class MockHydraConfig:
        job = MockJob()

    monkeypatch.setattr(HydraConfig, "get", lambda: MockHydraConfig())


def test_set_experiment(hydra_config):
    from hydraflow.mlflow import set_experiment

    with patch("mlflow.set_experiment") as mock_set_experiment:
        set_experiment(prefix="prefix_", suffix="_suffix")
        mock_set_experiment.assert_called_once_with("prefix_test_job_suffix")


def test_set_experiment_with_uri(hydra_config):
    from hydraflow.mlflow import set_experiment

    with (
        patch("mlflow.set_experiment") as mock_set_experiment,
        patch("mlflow.set_tracking_uri") as mock_set_tracking_uri,
    ):
        set_experiment(prefix="prefix_", suffix="_suffix", uri="http://example.com")
        mock_set_tracking_uri.assert_called_once_with("http://example.com")
        mock_set_experiment.assert_called_once_with("prefix_test_job_suffix")
