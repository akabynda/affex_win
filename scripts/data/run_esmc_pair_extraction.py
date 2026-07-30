"""Extract paired ESM-C 600M residue embeddings for PCANN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.esmc import EsmcPairEncoder
from affex.data.measurement import ExactMeasurement
from affex.data.plm_interact import encode_complex_embeddings
from affex.data.transform.graph_builder import item_embedding_key, read_structure
from affex.data.types import DataItem

DEFAULT_WEIGHTS = Path("models/esmc_600m_2024_12/esmc_600m_2024_12_v0.pth")
DEFAULT_LIBRARY_DIR = Path("models/biohub_esm_lib")


def chain_list(value: object) -> list[str]:
    return list(str(value))


def items_from_csvs(csv_paths: list[Path], pdb_dir: Path) -> list[DataItem]:
    items_by_key: dict[str, DataItem] = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            uid = str(row["uid"])
            item = DataItem(
                uid=uid,
                pdb=pdb_dir / f"{uid}.pdb",
                receptor_chains=chain_list(row["receptor_chains"]),
                ligand_chains=chain_list(row["ligand_chains"]),
                affinity=ExactMeasurement(0.0),
            )
            items_by_key[item_embedding_key(item)] = item
    return list(items_by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paired Biohub ESM-C 600M residue embeddings")
    parser.add_argument("pdb_dir", type=Path)
    parser.add_argument("--csv", type=Path, action="append", required=True)
    parser.add_argument("--savedir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--esmc-library-dir", type=Path, default=DEFAULT_LIBRARY_DIR)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-too-long", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")

    args.savedir.mkdir(parents=True, exist_ok=True)
    items = [item for item in items_from_csvs(args.csv, args.pdb_dir) if item.pdb.is_file()]
    pending = [item for item in items if not (args.savedir / f"{item_embedding_key(item)}.pt").exists()]
    total_pending = len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Using device: {device}")
    print(
        f"Selected {len(items)} complexes, {len(items) - total_pending} already processed, "
        f"{total_pending} pending, {len(pending)} to run"
    )
    print("Base model: biohub/esmc-600m-2024-12")
    print(f"Maximum token length: {args.max_length}")
    print("All chain boundaries use the native ESM-C chain-break token: '|'")

    encoder = EsmcPairEncoder(args.weights, args.esmc_library_dir, device)
    errors: list[tuple[str, str]] = []
    for item in tqdm(pending):
        out_path = args.savedir / f"{item_embedding_key(item)}.pt"
        try:
            structure = read_structure(item.pdb)
            embeddings = encode_complex_embeddings(
                item=item,
                structure=structure,
                encoder=encoder,
                max_length=args.max_length,
                chain_separator="|",
            )
            torch.save(embeddings, out_path)
        except ValueError as err:
            if args.skip_too_long and "exceeds max_length" in str(err):
                errors.append((item_embedding_key(item), str(err)))
                continue
            raise
        except Exception as err:
            errors.append((item_embedding_key(item), str(err)))
            print(f"{item_embedding_key(item)}: {err}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if errors:
        print(f"Finished with {len(errors)} skipped/failed complexes")
        for key, message in errors[:10]:
            print(f"  {key}: {message}")
    print("Finished ESM-C paired extraction")


if __name__ == "__main__":
    main()
