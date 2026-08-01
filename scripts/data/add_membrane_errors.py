"""Create a separate error table for membrane proteins in test sets."""

from __future__ import annotations

import argparse
import csv
from glob import glob
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBRANE_CSV = ROOT / "data/membrane_proteins.csv"
DEFAULT_OUTPUT = ROOT / "data/test/membrane_proteins_with_errors.csv"
DEFAULT_PREDICTION_GLOB = "predictions_*.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def infer_test_set(path: Path) -> str | None:
    stem = path.stem.lower()
    if stem.endswith("_testab"):
        return "testAB"
    if stem.endswith("_test_fabs") or stem.endswith("_test-fabs"):
        return "test_fabs"
    return None


def infer_approach(path: Path, test_set: str) -> str:
    stem = path.stem
    prefix = "predictions_"
    if stem.lower().startswith(prefix):
        stem = stem[len(prefix):]
    suffix = "_testAB" if test_set == "testAB" else "_test_fabs"
    return stem[: -len(suffix)]


def find_prediction_files(patterns: list[str]) -> list[Path]:
    paths: dict[Path, None] = {}
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            matches = [candidate]
        else:
            matches = [Path(match) for match in glob(pattern)]
        for path in matches:
            if path.is_file() and not path.stem.lower().endswith("_linkers"):
                paths[path.resolve()] = None
    return sorted(paths)


def collect_test_rows(
    membrane: dict[str, dict[str, str]], prediction_files: list[Path]
) -> list[dict[str, str | float]]:
    output: list[dict[str, str | float]] = []
    for path in prediction_files:
        source = infer_test_set(path)
        if source is None:
            continue
        approach = infer_approach(path, source)
        for row in read_rows(path):
            pdb_id = row["uid"].split("_", 1)[0].lower()
            if pdb_id not in membrane:
                continue
            error = abs(float(row["pred"]) - float(row["target"]))
            output.append({
                "uid": row["uid"],
                "pdb_id": pdb_id,
                "test_set": source,
                "approach": approach,
                "membrane_annotations": membrane[pdb_id]["membrane_annotations"],
                "target": float(row["target"]),
                "pred": float(row["pred"]),
                "absolute_error": error,
            })
    output.sort(key=lambda row: (str(row["test_set"]), str(row["approach"]), str(row["uid"])))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membrane-csv", type=Path, default=DEFAULT_MEMBRANE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "predictions",
        nargs="*",
        default=[str(ROOT / DEFAULT_PREDICTION_GLOB)],
        help="Prediction CSV paths or glob patterns (default: predictions_*.csv in the repo root)",
    )
    args = parser.parse_args()

    membrane = {row["uid"].lower(): row for row in read_rows(args.membrane_csv)}
    prediction_files = find_prediction_files(args.predictions)
    if not prediction_files:
        raise SystemExit("No prediction CSV files found")
    output_rows = collect_test_rows(membrane, prediction_files)
    fieldnames = [
        "uid", "pdb_id", "test_set", "approach", "membrane_annotations",
        "target", "pred", "absolute_error",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
