"""Extract independent single-chain ESM-3 embeddings for classic PCANN."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.esm2 import embed_sequences
from affex.data.esm3 import Esm3SequenceEncoder
from affex.data.model_sources import ESM3_WEIGHTS, ESMC_LIBRARY
from affex.data.transform.graph_builder import read_structure

DEFAULT_SAVEDIR = Path("data/raw/ppb-affinity/esm3_per_chain")


def uid_to_pdb_stems(uid: str) -> list[str]:
    return [part.split("_", 1)[0].lower() for part in str(uid).split("-")]


def stems_from_csvs(csv_paths: list[Path]) -> set[str]:
    stems: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        stems.update(stem for uid in frame["uid"] for stem in uid_to_pdb_stems(uid))
    return stems


class ClassicEsm3Model:
    def __init__(self, encoder: Esm3SequenceEncoder, max_length: int) -> None:
        self.encoder = encoder
        self.max_length = max_length

    def predict(self, sequence: str) -> torch.Tensor:
        return self.encoder.predict(sequence, self.max_length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_dir", type=Path)
    parser.add_argument("--savedir", type=Path, default=DEFAULT_SAVEDIR)
    parser.add_argument("--csv", type=Path, action="append", default=[])
    parser.add_argument("--weights", type=Path, default=ESM3_WEIGHTS)
    parser.add_argument("--esm-library-dir", type=Path, default=ESMC_LIBRARY)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--skip-too-long", action="store_true")
    parser.add_argument("--device", default="auto")
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
    print(f"Found {len(pdbs)} selected PDB files, {len(pdbs) - len(pending)} already processed, {len(pending)} to run")
    print("Encoding every unique chain independently with sequence-only ESM-3")

    encoder = Esm3SequenceEncoder(args.weights, args.esm_library_dir, device)
    model = ClassicEsm3Model(encoder, args.max_length)
    skipped: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for pdb_path in tqdm(pending):
        try:
            embeddings = embed_sequences(read_structure(pdb_path), model)
            embeddings["metadata"] = {
                "model_name": encoder.model_name,
                "embedding_size": encoder.embedding_size,
                "encoding": "independent_single_chain",
                "tracks": "sequence_only",
                "max_length": args.max_length,
                "format": encoder.output_format,
            }
            torch.save(embeddings, args.savedir / f"{pdb_path.stem.lower()}.pt")
        except ValueError as err:
            if args.skip_too_long and "exceeds max_length" in str(err):
                skipped.append((pdb_path.name, str(err)))
            else:
                failures.append((pdb_path.name, str(err)))
        except Exception as err:
            failures.append((pdb_path.name, str(err)))

    if skipped:
        print(f"Skipped {len(skipped)} structures exceeding max_length")
    if failures:
        print(f"Finished with {len(failures)} errors")
        for name, message in failures[:10]:
            print(f"  {name}: {message}")
        raise SystemExit(1)
    print("Finished classic ESM-3 extraction")


if __name__ == "__main__":
    main()
