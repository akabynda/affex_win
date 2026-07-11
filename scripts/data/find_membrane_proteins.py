"""Find membrane-protein PDB entries used anywhere in the PCANN dataset.

An entry is considered membrane-associated when RCSB annotates at least one
of its polymer entities with OPM, PDBTM, MemProtMD, or mpstruc.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ANNOTATIONS = ("OPM", "PDBTM", "MemProtMD", "mpstruc")
DEFAULT_INPUTS = (
    ("trainval", Path("data/train/pcann-plus-trainval.csv")),
    ("testAB", Path("data/test/testAB-clean.csv")),
    ("test_fabs", Path("data/test/test-fabs.csv")),
)


def read_dataset_entries(inputs: list[tuple[str, Path]]) -> dict[str, set[str]]:
    """Return normalized PDB ID -> dataset source names."""
    entries: dict[str, set[str]] = defaultdict(set)
    for source, path in inputs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "uid" not in reader.fieldnames:
                raise ValueError(f"{path} has no 'uid' column")
            for row in reader:
                pdb_id = row["uid"].strip().upper()
                if pdb_id:
                    entries[pdb_id].add(source)
    return dict(entries)


def query_annotation(annotation: str, retries: int = 4) -> set[str]:
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_annotation.type",
                "operator": "exact_match",
                "value": annotation,
            },
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry",
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "PCANN-membrane-scan/1.0"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            return {item["identifier"].upper() for item in result.get("result_set", [])}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/membrane_proteins.csv"),
        help="output CSV (default: data/membrane_proteins.csv)",
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("SOURCE", "CSV"),
        help="replace defaults with one or more SOURCE CSV pairs",
    )
    args = parser.parse_args()

    inputs = (
        [(source, Path(path)) for source, path in args.input]
        if args.input
        else list(DEFAULT_INPUTS)
    )
    dataset_entries = read_dataset_entries(inputs)
    annotation_entries = {name: query_annotation(name) for name in ANNOTATIONS}

    rows = []
    for pdb_id, sources in dataset_entries.items():
        labels = [name for name in ANNOTATIONS if pdb_id in annotation_entries[name]]
        if labels:
            rows.append((pdb_id.lower(), ";".join(sorted(sources)), ";".join(labels)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("uid", "dataset_sources", "membrane_annotations"))
        writer.writerows(sorted(rows))

    source_counts = {
        source: sum(source in sources and any(pdb_id in ids for ids in annotation_entries.values())
                    for pdb_id, sources in dataset_entries.items())
        for source, _ in inputs
    }
    print(f"Scanned {len(dataset_entries)} unique PDB entries")
    print(f"Found {len(rows)} membrane entries: " + ", ".join(f"{k}={v}" for k, v in source_counts.items()))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
