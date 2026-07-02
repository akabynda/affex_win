#!/usr/bin/env python3
"""Summarize metrics from saved prediction CSV files.

The expected input format is the output produced by ``src/predict.py``:
``uid,target,pred``.  Column aliases such as ``prediction`` and ``y_true`` are
also accepted to make the script useful for older ad-hoc outputs.
"""

from __future__ import annotations

import argparse
import math
from glob import glob
from pathlib import Path

import pandas as pd


TARGET_COLUMNS = ("target", "y_true", "true", "label", "affinity")
PRED_COLUMNS = ("pred", "prediction", "y_pred", "score")


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            paths.append(path)
            continue

        matched = sorted(Path(match) for match in glob(pattern))
        if matched:
            paths.extend(p for p in matched if p.is_file())
            continue

        raise SystemExit(f"No files matched: {pattern}")

    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return sorted(unique)


def find_column(columns: list[str], candidates: tuple[str, ...], path: Path) -> str:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise SystemExit(f"{path}: none of the expected columns found: {', '.join(candidates)}")


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    target_col = find_column(list(df.columns), TARGET_COLUMNS, path)
    pred_col = find_column(list(df.columns), PRED_COLUMNS, path)

    uid_col = next((column for column in df.columns if column.lower() == "uid"), None)
    if uid_col is None:
        uid = pd.Series([str(i) for i in range(len(df))], name="uid")
    else:
        uid = df[uid_col].astype(str)

    out = pd.DataFrame(
        {
            "uid": uid,
            "target": pd.to_numeric(df[target_col], errors="coerce"),
            "pred": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["target", "pred"])
    out = out.drop_duplicates(subset=["uid"], keep="last")
    return out


def infer_evaluation_set(path: Path) -> str:
    stem = path.stem.lower().replace("-", "_")
    if "testab" in stem:
        return "testAB-clean"
    if "test_fabs" in stem or "fabs" in stem:
        return "test-fabs"
    return "unknown"


def infer_approach(path: Path) -> str:
    name = path.stem
    lower = name.lower()
    if lower.startswith("predictions_"):
        name = name[len("predictions_") :]

    suffixes = (
        "_test_fabs",
        "_test-fabs",
        "_testAB",
        "_testab",
        "_test",
    )
    for suffix in suffixes:
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)]
            break

    return name


def pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return math.nan
    value = x.corr(y, method="pearson")
    return float(value) if pd.notna(value) else math.nan


def spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return math.nan
    value = x.corr(y, method="spearman")
    return float(value) if pd.notna(value) else math.nan


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    err = df["pred"] - df["target"]
    abs_err = err.abs()
    return {
        "n": int(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "median_ae": float(abs_err.median()),
        "bias": float(err.mean()),
        "pearson": pearson(df["pred"], df["target"]),
        "spearman": spearman(df["pred"], df["target"]),
    }


def summarize_one(path: Path, df: pd.DataFrame, subset: str) -> dict[str, object]:
    return {
        "approach": infer_approach(path),
        "evaluation_set": infer_evaluation_set(path),
        "subset": subset,
        "source": str(path),
        **compute_metrics(df),
    }


def add_common_subset_rows(rows: list[dict[str, object]], loaded: dict[Path, pd.DataFrame]) -> None:
    by_eval: dict[str, list[Path]] = {}
    for path in loaded:
        by_eval.setdefault(infer_evaluation_set(path), []).append(path)

    for evaluation_set, paths in sorted(by_eval.items()):
        if len(paths) < 2:
            continue

        common_uids: set[str] | None = None
        for path in paths:
            uids = set(loaded[path]["uid"])
            common_uids = uids if common_uids is None else common_uids & uids

        if not common_uids:
            continue

        for path in paths:
            df = loaded[path]
            common_df = df[df["uid"].isin(common_uids)].copy()
            row = summarize_one(path, common_df, f"common_{evaluation_set}")
            row["source"] = str(path)
            rows.append(row)


def format_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("mae", "rmse", "median_ae", "bias", "pearson", "spearman"):
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAE/Pearson/Spearman summary for prediction CSV files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["predictions_*.csv"],
        help="Prediction CSV paths or glob patterns. Default: predictions_*.csv",
    )
    parser.add_argument(
        "--output",
        default="prediction_results_summary.csv",
        help="Where to write the summary CSV. Use '-' to skip writing. Default: prediction_results_summary.csv",
    )
    parser.add_argument(
        "--include-common-subsets",
        action="store_true",
        help="Also add metrics on the UID intersection for each inferred evaluation set.",
    )
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    loaded = {path: read_predictions(path) for path in paths}

    rows = [summarize_one(path, df, "available") for path, df in loaded.items()]
    if args.include_common_subsets:
        add_common_subset_rows(rows, loaded)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["evaluation_set", "subset", "approach", "source"]).reset_index(drop=True)

    if args.output != "-":
        summary.to_csv(args.output, index=False)
        print(f"Saved {len(summary)} rows to {args.output}")

    print(format_float_columns(summary).to_string(index=False))


if __name__ == "__main__":
    main()
