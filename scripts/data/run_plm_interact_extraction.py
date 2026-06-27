"""Extract pair-aware PLM-interact embeddings for affex/PCANN.

The output format matches ResidueInterfacePlmInteractGraphBuilder: one file per
complex/chains assignment, keyed as uid_receptorchains_ligandchains.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.measurement import ExactMeasurement
from affex.data.plm_interact import PlmInteractPairEncoder, encode_complex_embeddings
from affex.data.transform.graph_builder import item_embedding_key, read_structure
from affex.data.types import DataItem

DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_CHECKPOINT_REPO = "danliu1226/PLM-interact-650M-Leakage-Free-Dataset"
DEFAULT_SAVEDIR = "data/raw/ppb-affinity/plm_interact"


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
    parser = argparse.ArgumentParser(description="Extract PLM-interact pair-aware residue embeddings")
    parser.add_argument("pdb_dir", type=Path, help="Directory containing .pdb files")
    parser.add_argument("--csv", type=Path, action="append", required=True, help="CSV containing uid/receptor/ligand")
    parser.add_argument("--savedir", type=Path, default=Path(DEFAULT_SAVEDIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base ESM-2 Hugging Face model")
    parser.add_argument("--embedding-size", type=int, default=1280)
    parser.add_argument(
        "--checkpoint-repo",
        default=DEFAULT_CHECKPOINT_REPO,
        help="Hugging Face repo with PLM-interact pytorch_model.bin; pass '' to use base ESM-2 only",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local PLM-interact pytorch_model.bin")
    parser.add_argument("--max-length", type=int, default=1603)
    parser.add_argument("--chain-separator", default="X", help="Residues inserted between chains on the same side")
    parser.add_argument("--bidirectional-average", action="store_true", help="Average rec-lig and lig-rec encodings")
    parser.add_argument(
        "--chain-policy",
        choices=["all", "interface"],
        default="all",
        help="Encode all requested chains, or only chains with cross-side interface contacts",
    )
    parser.add_argument("--interface-radius", type=float, default=5.0, help="Contact radius for --chain-policy interface")
    parser.add_argument("--skip-too-long", action="store_true", help="Skip complexes exceeding --max-length")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or any torch device string")
    parser.add_argument("--limit", type=int, default=None, help="Debug: process at most N pending complexes")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")

    checkpoint_repo = args.checkpoint_repo or None
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
    print(f"Base model: {args.model_name}")
    print(f"PLM-interact checkpoint repo: {checkpoint_repo or '<none>'}")
    print(f"Chain separator: {args.chain_separator!r}")
    print(f"Bidirectional average: {args.bidirectional_average}")
    print(f"Chain policy: {args.chain_policy}")
    if args.chain_policy == "interface":
        print(f"Interface radius: {args.interface_radius}")

    encoder = PlmInteractPairEncoder(
        model_name=args.model_name,
        embedding_size=args.embedding_size,
        device=device,
        checkpoint=args.checkpoint,
        checkpoint_repo=checkpoint_repo,
    )

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
                chain_separator=args.chain_separator,
                bidirectional_average=args.bidirectional_average,
                chain_policy=args.chain_policy,
                interface_radius=args.interface_radius,
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

    if errors:
        print(f"Finished with {len(errors)} skipped/failed complexes")
        for key, message in errors[:10]:
            print(f"  {key}: {message}")
    print("Finished PLM-interact extraction")


if __name__ == "__main__":
    main()
