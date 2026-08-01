"""Extract independent single-chain ESM-C embeddings for classic PCANN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.esm2 import embed_sequences
from affex.data.esmc import EsmcPairEncoder
from affex.data.model_sources import ESMC_LIBRARY, ESMC_WEIGHTS
from affex.data.transform.graph_builder import read_structure

DEFAULT_WEIGHTS = ESMC_WEIGHTS
DEFAULT_ESMC_LIBRARY = ESMC_LIBRARY
DEFAULT_SAVEDIR = Path("data/raw/ppb-affinity/esmc600")


def uid_to_pdb_stems(uid: str) -> list[str]:
    return [part.split("_", 1)[0].lower() for part in str(uid).split("-")]


def stems_from_csvs(csv_paths: list[Path]) -> set[str]:
    stems: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        stems.update(stem for uid in frame["uid"] for stem in uid_to_pdb_stems(uid))
    return stems


class ClassicEsmcModel:
    def __init__(self, encoder: EsmcPairEncoder, max_length: int) -> None:
        self.encoder = encoder
        self.max_length = max_length

    def predict(self, sequence: str) -> torch.Tensor:
        return self.encoder.encode_sequence(sequence, self.max_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract independent ESM-C chain embeddings for classic PCANN")
    parser.add_argument("pdb_dir", type=Path, help="Directory containing .pdb files")
    parser.add_argument("--savedir", type=Path, default=DEFAULT_SAVEDIR)
    parser.add_argument("--csv", type=Path, action="append", default=[], help="CSV containing a uid column")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--esmc-library-dir", type=Path, default=DEFAULT_ESMC_LIBRARY)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-too-long", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or any torch device string")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")

    args.savedir.mkdir(parents=True, exist_ok=True)
    selected_stems = stems_from_csvs(args.csv) if args.csv else None
    pdbs = sorted(args.pdb_dir.glob("*.pdb"))
    if selected_stems is not None:
        pdbs = [path for path in pdbs if path.stem.lower() in selected_stems]
    pending = [path for path in pdbs if not (args.savedir / f"{path.stem.lower()}.pt").exists()]

    print(f"Using device: {device}")
    print(
        f"Found {len(pdbs)} selected PDB files, {len(pdbs) - len(pending)} already processed, "
        f"{len(pending)} to run"
    )
    print("Encoding every unique chain independently; no chain separators or paired context")

    encoder = EsmcPairEncoder(args.weights, args.esmc_library_dir, device)
    model = ClassicEsmcModel(encoder, args.max_length)
    errors: list[tuple[str, str]] = []
    fatal_errors: list[tuple[str, str]] = []
    for pdb_path in tqdm(pending):
        try:
            structure = read_structure(pdb_path)
            embeddings = embed_sequences(structure, model)
            embeddings["metadata"] = {
                "model_name": encoder.model_name,
                "embedding_size": encoder.embedding_size,
                "encoding": "independent_single_chain",
                "max_length": args.max_length,
                "format": "esmc-classic-single-chain-v1",
            }
            torch.save(embeddings, args.savedir / f"{pdb_path.stem.lower()}.pt")
        except ValueError as err:
            if args.skip_too_long and "exceeds max_length" in str(err):
                errors.append((pdb_path.name, str(err)))
                continue
            fatal_errors.append((pdb_path.name, str(err)))
        except Exception as err:
            fatal_errors.append((pdb_path.name, str(err)))

    if errors:
        print(f"Skipped {len(errors)} structures exceeding max_length")
        for name, message in errors[:10]:
            print(f"  {name}: {message}")
    if fatal_errors:
        print(f"Finished with {len(fatal_errors)} errors")
        for name, message in fatal_errors[:10]:
            print(f"  {name}: {message}")
        raise SystemExit(1)
    print("Finished classic ESM-C extraction")


if __name__ == "__main__":
    main()
