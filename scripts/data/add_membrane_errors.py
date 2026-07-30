"""Create a separate error table for membrane proteins in test sets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBRANE_CSV = ROOT / "data/membrane_proteins.csv"
DEFAULT_OUTPUT = ROOT / "data/test/membrane_proteins_with_errors.csv"
TEST_PREDICTIONS = {
    "testAB": ROOT / "predictions_testAB.csv",
    "test_fabs": ROOT / "predictions_test_fabs.csv",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def collect_test_rows(membrane: dict[str, dict[str, str]]) -> list[dict[str, str | float]]:
    output: list[dict[str, str | float]] = []
    for source, path in TEST_PREDICTIONS.items():
        for row in read_rows(path):
            pdb_id = row["uid"].split("_", 1)[0].lower()
            if pdb_id not in membrane:
                continue
            error = abs(float(row["pred"]) - float(row["target"]))
            output.append({
                "uid": row["uid"],
                "pdb_id": pdb_id,
                "test_set": source,
                "membrane_annotations": membrane[pdb_id]["membrane_annotations"],
                "target": float(row["target"]),
                "pred": float(row["pred"]),
                "absolute_error": error,
            })
    output.sort(key=lambda row: float(row["absolute_error"]), reverse=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membrane-csv", type=Path, default=DEFAULT_MEMBRANE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    membrane = {row["uid"].lower(): row for row in read_rows(args.membrane_csv)}
    output_rows = collect_test_rows(membrane)
    fieldnames = [
        "uid", "pdb_id", "test_set", "membrane_annotations",
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
