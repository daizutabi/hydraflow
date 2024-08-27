from unittest.mock import MagicMock, patch

import pytest

from hydraflow.context import log_run, watch


def test_log_run_error_handling():
    config = MagicMock()
    config.some_param = "value"

    with (
        patch("hydraflow.context.log_params") as mock_log_params,
        patch("hydraflow.context.HydraConfig.get") as mock_hydra_config,
        patch("hydraflow.context.mlflow.log_artifacts") as mock_log_artifacts,
    ):
        mock_log_params.side_effect = Exception("Test exception")
        mock_hydra_config.return_value.runtime.output_dir = "/tmp"
        mock_log_artifacts.return_value = None

        with pytest.raises(Exception, match="Test exception"):
            with log_run(config):
                pass


def test_watch_error_handling():
    func = MagicMock()
    dir = "/tmp"

    with patch("hydraflow.context.Observer") as mock_observer:
        mock_observer_instance = mock_observer.return_value
        mock_observer_instance.start.side_effect = Exception("Test exception")

        with pytest.raises(Exception, match="Test exception"):
            with watch(func, dir):
                pass
