"""Rebuild caches, train every current experiment, test, and summarize results.

This is deliberately a single sequential pipeline: a failed preprocessing or
training command stops all downstream work instead of producing a mixed summary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from affex.data.model_sources import (  # noqa: E402
    ESM2_MODEL,
    ESM3_WEIGHTS,
    ESMC_LIBRARY,
    ESMC_WEIGHTS,
    PLM_INTERACT_CHECKPOINT,
    PLM_INTERACT_REPO,
)


PDB_DIR = Path("data/raw/ppb-affinity/pdb")
TRAIN_CSV = Path("data/train/pcann-plus-trainval.csv")
TESTS = {
    "testAB": Path("data/test/testAB-clean.csv"),
    "test_fabs": Path("data/test/test-fabs.csv"),
}
ALL_CSVS = (TRAIN_CSV, *TESTS.values())
FOLDS = ",".join(str(fold) for fold in range(25))
STATUS_PATH = ROOT / "pipeline_status.json"


@dataclass(frozen=True)
class Experiment:
    config: str
    group: str
    approach: str


EXPERIMENTS = (
    Experiment("pcann_reimpl-mc10", "GPU-pcann-42", "pcann"),
    Experiment("pcann_reimpl-rbf32-mc10", "GPU-rbf32-r05-42", "pcann_esm2_rbf32"),
    Experiment("pcann_reimpl-rbf64-mc10", "GPU-rbf64-r05-42", "pcann_esm2_rbf64"),
    Experiment(
        "pcann_reimpl-esm2-no-edge-features-mc10",
        "GPU-pcann_esm2_no_edge_features-42",
        "pcann_esm2_no_edge_features",
    ),
    Experiment("pcann_reimpl-esm3", "GPU-pcann_esm3-42", "pcann_esm3"),
    Experiment("pcann_reimpl-plm-interact-mc10", "GPU-plm_interact-42", "plm_interact"),
    Experiment("pcann_reimpl-esm2-paired-mc10", "GPU-esm2_paired-42", "esm2_paired"),
    Experiment(
        "pcann_reimpl-esm2-paired-bidir-mc10",
        "GPU-esm2_paired_bidir-42",
        "esm2_paired_bidir",
    ),
    Experiment(
        "pcann_reimpl-esm2-paired-linker-distance-gggs-mc10",
        "GPU-esm2_paired_linker_distance_gggs-42",
        "esm2_paired_linker_distance_gggs",
    ),
    Experiment(
        "pcann_reimpl-esm2-paired-linker-all-boundaries-mc10",
        "GPU-esm2_paired_linker_distance_gggs_all_boundaries-42",
        "esm2_paired_linker_distance_gggs_all_boundaries",
    ),
    Experiment(
        "pcann_reimpl-esm2-paired-linker-all-boundaries-rbf64-mc10",
        "GPU-esm2_paired_linker_distance_gggs_all_boundaries_rbf64-42",
        "esm2_paired_linker_distance_gggs_all_boundaries_rbf64",
    ),
    Experiment(
        "pcann_reimpl-esm2-all-structure-chains-x-mc10",
        "GPU-esm2_all_structure_chains_x-42",
        "esm2_all_structure_chains_x",
    ),
    Experiment(
        "pcann_reimpl-esm2-all-structure-chains-distance-gggs-mc10",
        "GPU-esm2_all_structure_chains_distance_gggs-42",
        "esm2_all_structure_chains_distance_gggs",
    ),
    Experiment("pcann_reimpl-esmc600", "GPU-esmc600-42", "pcann_esmc600"),
    Experiment(
        "pcann_reimpl-esmc600-paired-native-break-mc10",
        "GPU-esmc600_paired_native_break-42",
        "esmc600_paired_native_break",
    ),
    # Must run after GPU-esm2_paired-42 because it initializes PCANN from it.
    Experiment(
        "pcann_reimpl-esm2-paired-lora-mc10",
        "GPU-esm2_paired_lora-42",
        "esm2_paired_lora",
    ),
)


LINKER_DIRS = {
    "esm2_paired_linker_distance_gggs": Path(
        "data/raw/ppb-affinity/esm2_paired_linker_distance_gggs"
    ),
    "esm2_paired_linker_distance_gggs_all_boundaries": Path(
        "data/raw/ppb-affinity/esm2_paired_linker_distance_gggs_all_boundaries"
    ),
    "esm2_paired_linker_distance_gggs_all_boundaries_rbf64": Path(
        "data/raw/ppb-affinity/esm2_paired_linker_distance_gggs_all_boundaries"
    ),
    "esm2_all_structure_chains_distance_gggs": Path(
        "data/raw/ppb-affinity/esm2_all_structure_chains_distance_gggs"
    ),
}

PREPROCESSING_STAGES_BY_APPROACH = {
    "pcann": {"extract/esm2_per_chain"},
    "pcann_esm2_rbf32": {"extract/esm2_per_chain"},
    "pcann_esm2_rbf64": {"extract/esm2_per_chain"},
    "pcann_esm2_no_edge_features": {"extract/esm2_per_chain"},
    "pcann_esm3": {"extract/esm3_per_chain"},
    "plm_interact": {"extract/plm_interact"},
    "esm2_paired": {"extract/esm2_paired"},
    "esm2_paired_bidir": {"extract/esm2_paired_bidir"},
    "esm2_paired_linker_distance_gggs": {"extract/esm2_paired_linker_distance_gggs"},
    "esm2_paired_linker_distance_gggs_all_boundaries": {
        "extract/esm2_paired_linker_distance_gggs_all_boundaries"
    },
    "esm2_paired_linker_distance_gggs_all_boundaries_rbf64": {
        "extract/esm2_paired_linker_distance_gggs_all_boundaries"
    },
    "esm2_all_structure_chains_x": {"extract/esm2_all_structure_chains_x"},
    "esm2_all_structure_chains_distance_gggs": {
        "extract/esm2_all_structure_chains_distance_gggs"
    },
    "pcann_esmc600": {"extract/esmc600_per_chain"},
    "esmc600_paired_native_break": {"extract/esmc600_paired_native_break"},
    "esm2_paired_lora": {"extract/esm2_lora_tail_cache"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: str, stage: str, **extra: object) -> None:
    payload = {"state": state, "stage": stage, "updated_at": now(), **extra}
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def command_text(command: Sequence[object]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run(command: Sequence[object], stage: str) -> None:
    command = [str(part) for part in command]
    write_status("running", stage, command=command_text(command))
    print(f"\n[{now()}] {stage}\n$ {command_text(command)}", flush=True)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def csv_arguments() -> list[str]:
    arguments: list[str] = []
    for csv_path in ALL_CSVS:
        arguments.extend(("--csv", str(csv_path)))
    return arguments


def preprocessing_commands(python: str) -> list[tuple[str, list[str]]]:
    common_pair = [
        python,
        "scripts/data/run_plm_interact_extraction.py",
        str(PDB_DIR),
        *csv_arguments(),
        "--model-name",
        str(ESM2_MODEL),
        "--checkpoint-repo",
        "",
        "--skip-too-long",
        "--device",
        "auto",
    ]
    return [
        (
            "extract/esm2_per_chain",
            [
                python,
                "scripts/data/run_esm_extraction_inprocess.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/esm2_hf_per_chain",
                "--model-name",
                str(ESM2_MODEL),
                "--device",
                "auto",
            ],
        ),
        (
            "extract/plm_interact",
            [
                python,
                "scripts/data/run_plm_interact_extraction.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/plm_interact",
                "--model-name",
                str(ESM2_MODEL),
                "--skip-too-long",
                "--device",
                "auto",
            ],
        ),
        (
            "extract/esm2_paired",
            [*common_pair, "--savedir", "data/raw/ppb-affinity/esm2_paired"],
        ),
        (
            "extract/esm2_paired_bidir",
            [
                *common_pair,
                "--savedir",
                "data/raw/ppb-affinity/esm2_paired_bidir",
                "--bidirectional-average",
            ],
        ),
        (
            "extract/esm2_paired_linker_distance_gggs",
            [
                *common_pair,
                "--savedir",
                "data/raw/ppb-affinity/esm2_paired_linker_distance_gggs",
                "--distance-aware-linker",
            ],
        ),
        (
            "extract/esm2_paired_linker_distance_gggs_all_boundaries",
            [
                *common_pair,
                "--savedir",
                "data/raw/ppb-affinity/esm2_paired_linker_distance_gggs_all_boundaries",
                "--distance-aware-linker",
                "--inter-protein-distance-aware-linker",
            ],
        ),
        (
            "extract/esm2_all_structure_chains_x",
            [
                *common_pair,
                "--savedir",
                "data/raw/ppb-affinity/esm2_all_structure_chains_x",
                "--all-structure-chains",
                "--chain-separator",
                "X",
            ],
        ),
        (
            "extract/esm2_all_structure_chains_distance_gggs",
            [
                *common_pair,
                "--savedir",
                "data/raw/ppb-affinity/esm2_all_structure_chains_distance_gggs",
                "--all-structure-chains",
                "--distance-aware-linker",
            ],
        ),
        (
            "extract/esm2_lora_tail_cache",
            [
                python,
                "scripts/data/run_esm_tail_cache_extraction.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/esm2_pair_tail_cache_l1",
                "--model-name",
                str(ESM2_MODEL),
                "--tail-layers",
                "1",
                "--skip-too-long",
                "--device",
                "auto",
            ],
        ),
        (
            "extract/esm3_per_chain",
            [
                python,
                "scripts/data/run_esm3_extraction.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/esm3_per_chain",
                "--weights",
                str(ESM3_WEIGHTS),
                "--esm-library-dir",
                str(ESMC_LIBRARY),
                "--max-length",
                "2048",
                "--skip-too-long",
                "--device",
                "auto",
            ],
        ),
        (
            "extract/esmc600_per_chain",
            [
                python,
                "scripts/data/run_esmc_extraction.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/esmc600",
                "--weights",
                str(ESMC_WEIGHTS),
                "--esmc-library-dir",
                str(ESMC_LIBRARY),
                "--skip-too-long",
                "--device",
                "auto",
            ],
        ),
        (
            "extract/esmc600_paired_native_break",
            [
                python,
                "scripts/data/run_esmc_pair_extraction.py",
                str(PDB_DIR),
                *csv_arguments(),
                "--savedir",
                "data/raw/ppb-affinity/esmc600_paired_native_break",
                "--weights",
                str(ESMC_WEIGHTS),
                "--esmc-library-dir",
                str(ESMC_LIBRARY),
                "--skip-too-long",
                "--device",
                "auto",
            ],
        ),
    ]


def validate_pipeline(python: str) -> None:
    required = [PDB_DIR, *ALL_CSVS, ESM2_MODEL, ESM3_WEIGHTS, ESMC_WEIGHTS, ESMC_LIBRARY]
    missing = [str(path) for path in required if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError("Missing pipeline inputs: " + ", ".join(missing))

    config_dir = ROOT / "configs" / "experiment"
    discovered = {path.stem for path in config_dir.glob("*.yaml")}
    declared = {experiment.config for experiment in EXPERIMENTS}
    if discovered != declared:
        raise RuntimeError(
            f"Experiment registry mismatch; undeclared={sorted(discovered - declared)}, "
            f"missing={sorted(declared - discovered)}"
        )

    approaches = {experiment.approach for experiment in EXPERIMENTS}
    if set(PREPROCESSING_STAGES_BY_APPROACH) != approaches:
        raise RuntimeError("Preprocessing registry does not cover every experiment approach")

    forbidden = {
        "pcann_reimpl-esm2-paired-linker-mc10",
        "pcann_reimpl-esm2-paired-linker-bidir-mc10",
        "pcann_reimpl-esm2-paired-linker-interface-mc10",
    }
    if forbidden & discovered:
        raise RuntimeError("Obsolete experiments are still present: " + ", ".join(sorted(forbidden & discovered)))

    lora_config = (config_dir / "pcann_reimpl-esm2-paired-lora-mc10.yaml").read_text(encoding="utf-8")
    if f"model_name: {ESM2_MODEL.as_posix()}" not in lora_config.replace("\\", "/"):
        raise RuntimeError("LoRA is not configured to use the canonical ESM-2 source")

    subprocess.run(
        [python, "scripts/validate_training_protocols.py"],
        cwd=ROOT,
        check=True,
    )


def write_model_source_manifest() -> None:
    with (ROOT / "experiment_model_sources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("family", "source", "use"))
        writer.writeheader()
        writer.writerow(
            {
                "family": "ESM-2",
                "source": str(ESM2_MODEL),
                "use": "all ESM-2 per-chain, paired, linker, all-chain, PLM-interact-base, and LoRA runs",
            }
        )
        writer.writerow(
            {
                "family": "ESM-3",
                "source": str(ESM3_WEIGHTS),
                "use": "sequence-only independent per-chain ESM3-sm-open-v1 embeddings",
            }
        )
        writer.writerow(
            {
                "family": "ESM-C",
                "source": str(ESMC_WEIGHTS),
                "use": "all ESM-C per-chain and native-break paired runs",
            }
        )
        writer.writerow(
            {
                "family": "PLM-interact adapter",
                "source": f"{PLM_INTERACT_REPO} -> {PLM_INTERACT_CHECKPOINT}",
                "use": f"fine-tuned weights over canonical base {ESM2_MODEL}",
            }
        )


def train_all(python: str, experiments: Sequence[Experiment] = EXPERIMENTS) -> None:
    for experiment in experiments:
        run(
            [
                python,
                "src/train.py",
                "--multirun",
                f"+experiment={experiment.config}",
                f"datamodule.val_fold={FOLDS}",
                "seed=42",
                f"group={experiment.group}",
                "quiet=true",
                "+skip_postfit_evaluation=true",
                "~callbacks.save_predictions",
            ],
            f"train/{experiment.approach}",
        )


def test_all(python: str, experiments: Sequence[Experiment] = EXPERIMENTS) -> None:
    for experiment in experiments:
        checkpoint_dir = Path("logs/multiruns") / experiment.group
        for test_name, test_csv in TESTS.items():
            output = Path(f"predictions_{experiment.approach}_{test_name}.csv")
            run(
                [
                    python,
                    "src/predict.py",
                    "--checkpoints-dir",
                    str(checkpoint_dir),
                    "--test-csv",
                    str(test_csv),
                    "--output",
                    str(output),
                    "--device",
                    "auto",
                ],
                f"test/{experiment.approach}/{test_name}",
            )

            linker_dir = LINKER_DIRS.get(experiment.approach)
            if linker_dir is not None:
                run(
                    [
                        python,
                        "scripts/data/export_linker_report.py",
                        str(linker_dir),
                        "--predictions-csv",
                        str(output),
                        "--output",
                        f"predictions_{experiment.approach}_{test_name}_linkers.csv",
                    ],
                    f"linkers/{experiment.approach}/{test_name}",
                )


def summarize(python: str) -> None:
    run(
        [
            python,
            "scripts/summarize_predictions.py",
            "predictions_*.csv",
            "--include-common-subsets",
            "--output",
            "prediction_results_summary.csv",
        ],
        "summary",
    )
    run(
        [
            python,
            "scripts/data/add_membrane_errors.py",
            "predictions_*.csv",
        ],
        "summary/membrane_errors",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Reuse existing embedding caches and resume from 25-fold training",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="APPROACH",
        help="Train and test only the named registered approaches",
    )
    args = parser.parse_args()
    python = sys.executable

    try:
        write_status("running", "validation")
        validate_pipeline(python)
        if args.validate_only:
            write_status("validated", "validation", experiments=len(EXPERIMENTS))
            print(f"Pipeline valid: {len(EXPERIMENTS)} experiments, 25 folds each")
            return

        experiments = EXPERIMENTS
        if args.only:
            requested = set(args.only)
            experiments = tuple(experiment for experiment in EXPERIMENTS if experiment.approach in requested)
            unknown = requested - {experiment.approach for experiment in experiments}
            if unknown:
                raise ValueError("Unknown approaches requested with --only: " + ", ".join(sorted(unknown)))

        write_model_source_manifest()
        if not args.skip_preprocessing:
            commands = preprocessing_commands(python)
            if args.only:
                required_stages = set().union(
                    *(PREPROCESSING_STAGES_BY_APPROACH[experiment.approach] for experiment in experiments)
                )
                commands = [(stage, command) for stage, command in commands if stage in required_stages]
            for stage, command in commands:
                run(command, stage)
        train_all(python, experiments)
        test_all(python, experiments)
        summarize(python)
        write_status("complete", "complete", experiments=len(experiments))
        print(f"[{now()}] Full pipeline complete", flush=True)
    except BaseException as error:
        write_status("failed", "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
