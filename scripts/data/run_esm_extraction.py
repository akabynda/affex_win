"""Extract ESM2 per-residue embeddings for PDB structures.

Usage:
    uv run python scripts/data/run_esm_extraction.py <pdb_dir> [--savedir <esm_dir>] [--workers 2]

Outputs one .pt file per PDB into the save directory (default: data/raw/ppb-affinity/esm).
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from affex.data.esm2 import EsmRunner

DEFAULT_SAVEDIR = "data/raw/ppb-affinity/esm"


def resolve_esm_model_dir(cli_value: str | None) -> str:
    """ESM model dir from --esm-model-dir, else $ESM_MODEL_DIR; error if neither is set."""
    model_dir = cli_value or os.environ.get("ESM_MODEL_DIR")
    if not model_dir:
        raise SystemExit(
            "ESM model directory not set. Pass --esm-model-dir or set the ESM_MODEL_DIR "
            "environment variable to a clone of facebookresearch/esm."
        )
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="Extract ESM2 embeddings from PDB files")
    parser.add_argument("pdb_dir", help="Directory containing .pdb files")
    parser.add_argument("--savedir", default=DEFAULT_SAVEDIR, help="Output directory for .pt files")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument(
        "--esm-model-dir",
        default=None,
        help="Path to a facebookresearch/esm clone (falls back to $ESM_MODEL_DIR).",
    )
    args = parser.parse_args()

    esm_model_dir = resolve_esm_model_dir(args.esm_model_dir)
    datadir = Path(args.pdb_dir)
    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    esm = EsmRunner(esm_model_dir, str(savedir))
    all_pdbs = sorted(datadir.glob("*.pdb"))
    pdb_list = [p for p in all_pdbs if not (savedir / f"{p.stem}.pt").exists()]
    print(f"Found {len(all_pdbs)} PDB files, {len(all_pdbs) - len(pdb_list)} already processed, {len(pdb_list)} to run")

    with Pool(args.workers) as pool:
        list(tqdm(pool.imap_unordered(esm.process_one, pdb_list), total=len(pdb_list)))


if __name__ == "__main__":
    main()
