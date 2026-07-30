"""Export distance-aware linker metadata from paired ESM embedding files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one CSV row per linker used during ESM extraction")
    parser.add_argument("embeddings_dir", type=Path, help="Directory containing paired ESM .pt files")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        help="Optional predictions CSV used to restrict the report and attach target/pred columns",
    )
    args = parser.parse_args()

    predictions: dict[str, dict[str, object]] | None = None
    if args.predictions_csv is not None:
        prediction_frame = pd.read_csv(args.predictions_csv)
        predictions = {
            str(row["uid"]).lower(): {
                "uid": row["uid"],
                "target": row.get("target"),
                "pred": row.get("pred"),
            }
            for _, row in prediction_frame.iterrows()
        }

    rows: list[dict[str, object]] = []
    for embedding_path in sorted(args.embeddings_dir.glob("*.pt")):
        prediction = predictions.get(embedding_path.stem.lower()) if predictions is not None else None
        if predictions is not None and prediction is None:
            continue
        metadata = torch.load(embedding_path, map_location="cpu", weights_only=False).get("metadata", {})
        linker_groups = (
            ("receptor", "receptor_linkers"),
            ("ligand", "ligand_linkers"),
            ("inter_protein", "inter_protein_linkers"),
            ("reverse_inter_protein", "reverse_inter_protein_linkers"),
        )
        for side, metadata_key in linker_groups:
            for linker in metadata.get(metadata_key, []):
                rows.append(
                    {
                        "uid": prediction["uid"] if prediction is not None else embedding_path.stem,
                        "item_key": embedding_path.stem,
                        "target": prediction["target"] if prediction is not None else None,
                        "pred": prediction["pred"] if prediction is not None else None,
                        "side": side,
                        "source_chain": linker["source_chain"],
                        "target_chain": linker["target_chain"],
                        "terminal_distance_angstrom": linker["terminal_distance_angstrom"],
                        "target_length_angstrom": linker["target_length_angstrom"],
                        "gggs_repeats": linker["repeats"],
                        "linker_residues": len(linker["sequence"]),
                        "chosen_contour_length_angstrom": len(linker["sequence"])
                        * metadata["residue_contour_length_angstrom"],
                        "linker_sequence": linker["sequence"],
                    }
                )

    columns = [
        "uid",
        "item_key",
        "target",
        "pred",
        "side",
        "source_chain",
        "target_chain",
        "terminal_distance_angstrom",
        "target_length_angstrom",
        "gggs_repeats",
        "linker_residues",
        "chosen_contour_length_angstrom",
        "linker_sequence",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} linker rows from {args.embeddings_dir} to {args.output}")


if __name__ == "__main__":
    main()
