#!/usr/bin/env python3
"""Standalone ensemble inference for the affex PCANN release.

Discovers the ``fold*_seed*/`` checkpoint directories under ``--checkpoints-dir``,
runs each model on the test set, averages predictions per UID, writes a
``uid,target,pred`` CSV, and prints ensemble MAE / Pearson / Spearman.

Usage:
    uv run python src/predict.py \
        --checkpoints-dir logs/multiruns/EXP-043 \
        --test-csv data/test/testAB-clean.csv \
        --output predictions_testAB.csv

    # Fast subset (e.g. a 2-fold smoke test):
    uv run python src/predict.py --checkpoints-dir logs/multiruns/EXP-043 \
        --test-csv data/test/test-fabs.csv --folds 0,1

Each ``fold*_seed*/`` directory must contain ``checkpoints/epoch_*.ckpt`` and
``.hydra/config.yaml`` — the model is rebuilt from that config before loading weights.
The shipped test CSVs are already leak-filtered, so no filtering happens here.
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef

from affex.data.measurement import RT, ExactMeasurement
from affex.model.lightning import Alpine


def discover_runs(checkpoints_dir: Path) -> list[dict]:
    """Find all fold*_seed* subdirs and parse fold/seed from names."""
    runs = []
    for d in sorted(checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"fold(\d+)_seed(\d+)", d.name)
        if m:
            runs.append({"path": d, "fold": int(m.group(1)), "seed": int(m.group(2))})
    return runs


def find_best_checkpoint(run_dir: Path) -> Path:
    """Find the best (non-last) checkpoint in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    candidates = [p for p in ckpt_dir.glob("epoch_*.ckpt") if p.name != "last.ckpt"]
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoint found in {ckpt_dir}")
    if len(candidates) == 1:
        return candidates[0]
    # If multiple, pick the highest epoch number (shouldn't happen with save_top_k=1).
    return max(candidates, key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else 0)


def build_test_dataloader(run_dir: Path, test_csv: Path | None = None):
    """Instantiate the datamodule from the run's hydra config and return its test dataloader."""
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    if test_csv:
        datamodule.test_csv = test_csv
    datamodule.setup("test")
    return datamodule.test_dataloader()


def item_key(item) -> str:
    """Unique key: uid + chains, so the same PDB with different chain assignments is preserved."""
    rec = "".join(item.receptor_chains)
    lig = "".join(item.ligand_chains)
    return f"{item.uid}_{rec}_{lig}"


def run_inference(run_dir: Path, test_dataloader) -> tuple[dict[str, float], dict[str, float]]:
    """Load model from checkpoint, run test inference.

    Returns (predictions, targets) mapping key -> value. Only ExactMeasurement samples included.
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    ckpt_path = find_best_checkpoint(run_dir)
    model = Alpine.load_from_checkpoint(
        ckpt_path,
        model=hydra.utils.instantiate(cfg.lightning.model),
        optimizer_fn=hydra.utils.instantiate(cfg.lightning.optimizer_fn),
        lr_scheduler_fn=hydra.utils.instantiate(cfg.lightning.lr_scheduler_fn),
    )
    model.eval()

    predictions: dict[str, float] = {}
    targets: dict[str, float] = {}

    with torch.no_grad():
        for batch in test_dataloader:
            graphs, descs = batch
            preds = model.model.forward(graphs).cpu().flatten()
            for i, item in enumerate(descs):
                if isinstance(item.affinity, ExactMeasurement):
                    key = item_key(item)
                    predictions[key] = preds[i].item()
                    targets[key] = RT * math.log(item.affinity.value)

    return predictions, targets


def compute_metrics(preds: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    """Compute Pearson, Spearman, MAE from key-keyed dicts."""
    keys = sorted(preds.keys())
    pred_t = torch.tensor([preds[k] for k in keys])
    tgt_t = torch.tensor([targets[k] for k in keys])

    return {
        "MAE": MeanAbsoluteError()(pred_t, tgt_t).item(),
        "Pearson": PearsonCorrCoef()(pred_t, tgt_t).item(),
        "Spearman": SpearmanCorrCoef()(pred_t, tgt_t).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference over a directory of fold*_seed* checkpoints")
    parser.add_argument(
        "--checkpoints-dir", required=True, help="Directory containing fold*_seed*/ subdirectories"
    )
    parser.add_argument("--test-csv", required=True, help="Test set CSV (already leak-filtered)")
    parser.add_argument("--output", default=None, help="Path to write the uid,target,pred ensemble CSV")
    parser.add_argument("--folds", default=None, help="Comma-separated fold indices to include (e.g. '0,1')")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.is_dir():
        raise SystemExit(f"Checkpoints directory not found: {checkpoints_dir}")

    runs = discover_runs(checkpoints_dir)
    if not runs:
        raise SystemExit(f"No fold*_seed* directories found in {checkpoints_dir}")

    if args.folds is not None:
        selected = {int(f) for f in args.folds.split(",")}
        runs = [r for r in runs if r["fold"] in selected]
        if not runs:
            raise SystemExit(f"No runs found for folds {selected}")

    folds = sorted({r["fold"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    print(f"Found {len(runs)} runs ({len(folds)} folds x {len(seeds)} seeds)")

    # Build the test dataloader once from the first run's config.
    test_csv = Path(args.test_csv)
    test_dataloader = build_test_dataloader(runs[0]["path"], test_csv)

    # Collect each run's predictions, then average per UID across all runs.
    preds_per_key: dict[str, list[float]] = defaultdict(list)
    all_targets: dict[str, float] = {}
    for run in runs:
        print(f"  Loading {run['path'].name}...", end=" ", flush=True)
        preds, targets = run_inference(run["path"], test_dataloader)
        print(f"{len(preds)} samples")
        for key, value in preds.items():
            preds_per_key[key].append(value)
        all_targets.update(targets)

    ensemble = {key: sum(values) / len(values) for key, values in preds_per_key.items()}
    metrics = compute_metrics(ensemble, all_targets)

    print("─" * 48)
    print(f"Ensemble over {len(runs)} models on {test_csv.name} (N={len(ensemble)})")
    print(f"  MAE      = {metrics['MAE']:.4f}")
    print(f"  Pearson  = {metrics['Pearson']:.4f}")
    print(f"  Spearman = {metrics['Spearman']:.4f}")
    print("─" * 48)

    if args.output:
        rows = [
            {"uid": key, "target": all_targets[key], "pred": ensemble[key]}
            for key in sorted(ensemble.keys())
        ]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
