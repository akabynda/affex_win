"""Extract ESM2 embeddings while keeping the model loaded in one process.

This is useful on Windows/GPU where spawning the upstream ESM extractor for
every chain reloads the 650M model repeatedly.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.esm2 import embed_sequences
from affex.data.model_sources import ESM2_MODEL
from affex.data.transform.graph_builder import read_structure


def uid_to_pdb_stems(uid: str) -> list[str]:
    stems = []
    for part in str(uid).split("-"):
        stems.append(part.split("_", 1)[0].lower())
    return stems


def stems_from_csvs(csv_paths: list[Path]) -> set[str]:
    stems: set[str] = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        stems.update(stem for uid in df["uid"] for stem in uid_to_pdb_stems(uid))
    return stems


class InProcessHuggingFaceEsmModel:
    def __init__(self, model_name: str, device: torch.device, truncation_seq_length: int = 1022) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.truncation_seq_length = truncation_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)

    def predict(self, sequence: str) -> torch.Tensor:
        truncated = sequence[: self.truncation_seq_length]
        features = self.tokenizer(
            truncated,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        features = {name: value.to(self.device) for name, value in features.items()}
        with torch.inference_mode():
            output = self.model(**features, return_dict=True)
        return output.last_hidden_state[0, 1 : len(truncated) + 1].detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ESM2 embeddings for selected PDBs")
    parser.add_argument("pdb_dir", type=Path, help="Directory containing .pdb files")
    parser.add_argument(
        "--savedir",
        type=Path,
        default=Path("data/raw/ppb-affinity/esm2_hf_per_chain"),
    )
    parser.add_argument("--csv", type=Path, action="append", default=[], help="CSV containing a uid column")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or any torch device string")
    parser.add_argument("--model-name", default=str(ESM2_MODEL), help="Canonical local ESM-2 model")
    parser.add_argument("--truncation-seq-length", type=int, default=1022)
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
        pdbs = [p for p in pdbs if p.stem.lower() in selected_stems]
    pending = [p for p in pdbs if not (args.savedir / f"{p.stem.lower()}.pt").exists()]

    print(f"Using device: {device}")
    print(f"Found {len(pdbs)} selected PDB files, {len(pdbs) - len(pending)} already processed, {len(pending)} to run")
    print(f"Hugging Face model: {args.model_name}")
    model = InProcessHuggingFaceEsmModel(
        args.model_name,
        device,
        truncation_seq_length=args.truncation_seq_length,
    )

    errors: list[tuple[str, str]] = []
    for pdb_path in tqdm(pending):
        try:
            structure = read_structure(pdb_path)
            embeddings = embed_sequences(structure, model)
            torch.save(embeddings, args.savedir / f"{pdb_path.stem.lower()}.pt")
        except Exception as err:
            errors.append((pdb_path.name, str(err)))
            print(f"{pdb_path}: {err}")

    if errors:
        print(f"Finished with {len(errors)} errors")
        raise SystemExit(1)
    print("Finished ESM extraction")


if __name__ == "__main__":
    main()
