from pathlib import Path

import hydra
import pytest

from train import validate_training_protocol


CONFIG_DIR = str(Path(__file__).parents[1] / "configs")


def _compose(*overrides: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return hydra.compose(
            config_name="train.yaml",
            overrides=["+experiment=pcann_reimpl-mc10", *overrides],
        )


def test_original_pcann_protocol_is_valid():
    validate_training_protocol(_compose())


def test_protocol_rejects_one_off_batch_override():
    with pytest.raises(ValueError, match="batch_size"):
        validate_training_protocol(_compose("datamodule.batch_size=64"))
