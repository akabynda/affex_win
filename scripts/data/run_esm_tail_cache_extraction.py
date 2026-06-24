"""Extract cached pair ESM states before tail layers/adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from affex.data.esm2 import get_alignment, get_alignment_indices, get_full_sequences, get_sequences
from affex.data.measurement import ExactMeasurement
from affex.data.plm_interact import build_side_sequence
from affex.data.transform.graph_builder import item_embedding_key, read_structure
from affex.data.types import DataItem

DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_SAVEDIR = "data/raw/ppb-affinity/esm2_pair_tail_cache"


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


def encode_tail_cache(
    item: DataItem,
    tokenizer: Any,
    model: torch.nn.Module,
    tail_layers: int,
    max_length: int,
    device: torch.device,
    chain_separator: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    structure = read_structure(item.pdb)
    full_sequences = get_full_sequences(structure)
    sequences = get_sequences(structure)
    alignment = get_alignment(structure)
    indices = get_alignment_indices(sequences, alignment)

    missing = [chain for chain in item.receptor_chains + item.ligand_chains if chain not in full_sequences]
    if missing:
        raise KeyError(f"chains missing from SEQRES/full sequences for {item.uid}: {missing}")

    receptor_sequence, receptor_spans = build_side_sequence(full_sequences, item.receptor_chains, chain_separator)
    ligand_sequence, ligand_spans = build_side_sequence(full_sequences, item.ligand_chains, chain_separator)
    expected_length = len(receptor_sequence) + len(ligand_sequence) + 3
    if expected_length > max_length:
        raise ValueError(
            f"paired sequence length {expected_length} exceeds max_length={max_length}; "
            "increase --max-length or skip this item"
        )

    tokenized = tokenizer(
        receptor_sequence,
        ligand_sequence,
        padding=False,
        truncation="longest_first",
        return_tensors="pt",
        max_length=max_length,
    )
    token_count = int(tokenized["input_ids"].shape[1])
    if token_count != expected_length:
        raise ValueError(f"tokenized length {token_count} does not match expected pair length {expected_length}")

    features = {name: value.to(device) for name, value in tokenized.items()}
    with torch.inference_mode():
        output = model(**features, output_hidden_states=True, return_dict=True)
    hidden_states = output.hidden_states
    cache_index = -tail_layers - 1
    if abs(cache_index) > len(hidden_states):
        raise ValueError(f"tail_layers={tail_layers} exceeds hidden state count={len(hidden_states)}")

    receptor_start = 1
    ligand_start = len(receptor_sequence) + 2
    token_offsets = {
        **{span.chain_id: receptor_start + span.start for span in receptor_spans},
        **{span.chain_id: ligand_start + span.start for span in ligand_spans},
    }

    chain_ids = item.receptor_chains + item.ligand_chains
    return {
        "hidden_states": hidden_states[cache_index][0].detach().to(dtype=dtype).cpu(),
        "attention_mask": tokenized["attention_mask"][0].to(dtype=torch.long).cpu(),
        "token_offsets": token_offsets,
        "indices": {chain_id: indices[chain_id] for chain_id in chain_ids},
        "sequences": {chain_id: full_sequences[chain_id] for chain_id in chain_ids},
        "metadata": {
            "model_name": getattr(model, "name_or_path", None),
            "tail_layers": tail_layers,
            "chain_separator": chain_separator,
            "format": "paired-esm-tail-cache-v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cached pair ESM states before tail layers/adapters")
    parser.add_argument("pdb_dir", type=Path, help="Directory containing .pdb files")
    parser.add_argument("--csv", type=Path, action="append", required=True, help="CSV containing uid/receptor/ligand")
    parser.add_argument("--savedir", type=Path, default=Path(DEFAULT_SAVEDIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tail-layers", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1603)
    parser.add_argument("--chain-separator", default="X")
    parser.add_argument("--skip-too-long", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or any torch device string")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=None, help="Debug: process at most N pending complexes")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
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
    print(f"Tail layers: {args.tail_layers}")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, add_pooling_layer=False).eval().to(device)

    errors: list[tuple[str, str]] = []
    for item in tqdm(pending):
        out_path = args.savedir / f"{item_embedding_key(item)}.pt"
        try:
            cache = encode_tail_cache(
                item=item,
                tokenizer=tokenizer,
                model=model,
                tail_layers=args.tail_layers,
                max_length=args.max_length,
                device=device,
                chain_separator=args.chain_separator,
                dtype=dtype,
            )
            torch.save(cache, out_path)
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
    print("Finished ESM tail cache extraction")


if __name__ == "__main__":
    main()
