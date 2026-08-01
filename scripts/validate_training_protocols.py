"""Compose every experiment and verify the shared training protocol."""

from pathlib import Path
import sys

import hydra


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import validate_training_protocol  # noqa: E402


def compose(experiment: str, *overrides: str):
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        return hydra.compose(
            config_name="train.yaml",
            overrides=[f"+experiment={experiment}", *overrides],
        )


def main() -> None:
    experiment_dir = ROOT / "configs" / "experiment"
    checked = 0
    skipped = []
    for path in sorted(experiment_dir.glob("*.yaml")):
        cfg = compose(path.stem)
        protocol = cfg.get("training_protocol")
        if protocol and bool(protocol.get("enforce", False)):
            validate_training_protocol(cfg)
            checked += 1
        else:
            skipped.append(path.stem)

    try:
        validate_training_protocol(
            compose("pcann_reimpl-mc10", "datamodule.batch_size=64")
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Protocol guard accepted an inconsistent batch size")

    print(f"Validated {checked} comparable experiments")
    if skipped:
        print("Explicitly non-comparable: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
