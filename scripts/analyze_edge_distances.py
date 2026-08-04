"""Report and save classic PCANN kNN edge-distance distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from affex.data.ppb_dataset import PPBAffinityDataset
from affex.data.transform.graph_builder import ResidueInterfaceEsmGraphBuilder


COORDINATE_MODES = ("ca", "heavy_atom_centroid")
QUANTILES = (
    ("min_angstrom", 0.00),
    ("p01_angstrom", 0.01),
    ("p05_angstrom", 0.05),
    ("p25_angstrom", 0.25),
    ("median_angstrom", 0.50),
    ("p75_angstrom", 0.75),
    ("p95_angstrom", 0.95),
    ("p99_angstrom", 0.99),
    ("max_angstrom", 1.00),
)


def analyze(args: argparse.Namespace, coordinate_mode: str) -> dict[str, object]:
    dataset = PPBAffinityDataset(
        args.pdb_dir,
        pd.read_csv(args.csv),
        ResidueInterfaceEsmGraphBuilder(
            radius=args.interface_radius,
            esm_dir=args.esm_dir,
            coordinate_mode=coordinate_mode,
        ),
        num_workers=args.workers,
    )
    distances = torch.cat([graph.distances for graph, _ in dataset.data]).float()
    levels = torch.tensor([level for _, level in QUANTILES])
    values = torch.quantile(distances, levels).tolist()
    row: dict[str, object] = {
        "coordinate_mode": coordinate_mode,
        "source_csv": str(args.csv),
        "interface_contact_radius_angstrom": args.interface_radius,
        "knn_k": 50,
        "graphs": len(dataset),
        "edges": distances.numel(),
    }
    row.update({name: value for (name, _), value in zip(QUANTILES, values)})
    for cutoff in (5.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        row[f"fraction_gt_{cutoff:g}A"] = float((distances > cutoff).float().mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/train/pcann-plus-trainval.csv"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("data/raw/ppb-affinity/pdb"))
    parser.add_argument("--esm-dir", type=Path, default=Path("data/raw/ppb-affinity/esm2_hf_per_chain"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--coordinate-mode",
        action="append",
        choices=COORDINATE_MODES,
        help="Mode to analyze; repeat as needed. Default: both modes",
    )
    parser.add_argument("--interface-radius", type=float, default=5.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/edge_distance_distribution.csv"),
    )
    args = parser.parse_args()

    modes = args.coordinate_mode or list(COORDINATE_MODES)
    rows = [analyze(args, mode) for mode in modes]
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))
    print(f"Saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
