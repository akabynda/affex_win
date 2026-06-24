from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import torch
from torch import Tensor, nn

from affex.data.esm2 import get_alignment, get_alignment_indices, get_full_sequences, get_sequences
from affex.data.types import DataItem


@dataclass(frozen=True)
class ChainSpan:
    chain_id: str
    start: int
    stop: int


class PlmInteractWrapper(nn.Module):
    def __init__(self, model_name: str, embedding_size: int) -> None:
        super().__init__()
        from transformers import AutoModelForMaskedLM

        self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name)
        self.classifier = nn.Linear(embedding_size, 1)


class PlmInteractPairEncoder:
    def __init__(
        self,
        model_name: str,
        embedding_size: int,
        device: torch.device,
        checkpoint: Path | None = None,
        checkpoint_repo: str | None = None,
    ) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.embedding_size = embedding_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = PlmInteractWrapper(model_name, embedding_size)

        checkpoint_path = checkpoint
        if checkpoint_path is None and checkpoint_repo:
            checkpoint_path = Path(hf_hub_download(repo_id=checkpoint_repo, filename="pytorch_model.bin"))
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            self._load_compatible_state_dict(state)

        self.model.eval().to(device)

    def _load_compatible_state_dict(self, state: dict[str, Tensor]) -> None:
        model_state = self.model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state.items():
            clean_key = key.removeprefix("module.")
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                compatible[clean_key] = value
            else:
                skipped.append(key)

        self.model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"Skipped {len(skipped)} checkpoint tensors with missing keys or incompatible shapes")

    def encode_pair(self, sequence_a: str, sequence_b: str, max_length: int) -> tuple[Tensor, Tensor]:
        expected_length = len(sequence_a) + len(sequence_b) + 3
        if expected_length > max_length:
            raise ValueError(
                f"paired sequence length {expected_length} exceeds max_length={max_length}; "
                "increase --max-length or skip this item"
            )

        tokenized = self.tokenizer(
            sequence_a,
            sequence_b,
            padding=False,
            truncation="longest_first",
            return_tensors="pt",
            max_length=max_length,
        )
        token_count = int(tokenized["input_ids"].shape[1])
        if token_count != expected_length:
            raise ValueError(
                f"tokenized length {token_count} does not match expected PLM-interact length {expected_length}"
            )

        features = {name: value.to(self.device) for name, value in tokenized.items()}
        with torch.inference_mode():
            output = self.model.esm_mask.base_model(**features, return_dict=True)
        hidden = output.last_hidden_state[0].detach().cpu()

        first_start = 1
        first_stop = first_start + len(sequence_a)
        second_start = first_stop + 1
        second_stop = second_start + len(sequence_b)
        return hidden[first_start:first_stop], hidden[second_start:second_stop]


def build_side_sequence(
    full_sequences: dict[str, str],
    chain_ids: list[str],
    chain_separator: str,
) -> tuple[str, list[ChainSpan]]:
    parts: list[str] = []
    spans: list[ChainSpan] = []
    cursor = 0
    for index, chain_id in enumerate(chain_ids):
        if index > 0 and chain_separator:
            parts.append(chain_separator)
            cursor += len(chain_separator)

        sequence = full_sequences[chain_id]
        start = cursor
        stop = start + len(sequence)
        spans.append(ChainSpan(chain_id=chain_id, start=start, stop=stop))
        parts.append(sequence)
        cursor = stop

    return "".join(parts), spans


def split_side_embeddings(side_embeddings: Tensor, spans: list[ChainSpan]) -> dict[str, Tensor]:
    return {span.chain_id: side_embeddings[span.start : span.stop].clone() for span in spans}


def average_chain_embeddings(first: dict[str, Tensor], second: dict[str, Tensor]) -> dict[str, Tensor]:
    return {chain_id: (first[chain_id] + second[chain_id]) / 2 for chain_id in first}


def encode_complex_embeddings(
    item: DataItem,
    structure: gemmi.Structure,
    encoder: PlmInteractPairEncoder,
    max_length: int,
    chain_separator: str = "X",
    bidirectional_average: bool = False,
) -> dict[str, Any]:
    full_sequences = get_full_sequences(structure)
    sequences = get_sequences(structure)
    alignment = get_alignment(structure)
    indices = get_alignment_indices(sequences, alignment)

    missing = [chain for chain in item.receptor_chains + item.ligand_chains if chain not in full_sequences]
    if missing:
        raise KeyError(f"chains missing from SEQRES/full sequences for {item.uid}: {missing}")

    receptor_sequence, receptor_spans = build_side_sequence(full_sequences, item.receptor_chains, chain_separator)
    ligand_sequence, ligand_spans = build_side_sequence(full_sequences, item.ligand_chains, chain_separator)

    receptor_embeddings, ligand_embeddings = encoder.encode_pair(receptor_sequence, ligand_sequence, max_length)
    chain_embeddings = {
        **split_side_embeddings(receptor_embeddings, receptor_spans),
        **split_side_embeddings(ligand_embeddings, ligand_spans),
    }

    if bidirectional_average:
        ligand_embeddings_rev, receptor_embeddings_rev = encoder.encode_pair(
            ligand_sequence,
            receptor_sequence,
            max_length,
        )
        reverse_chain_embeddings = {
            **split_side_embeddings(receptor_embeddings_rev, receptor_spans),
            **split_side_embeddings(ligand_embeddings_rev, ligand_spans),
        }
        chain_embeddings = average_chain_embeddings(chain_embeddings, reverse_chain_embeddings)

    return {
        "sequences": {chain_id: full_sequences[chain_id] for chain_id in chain_embeddings},
        "embeddings": chain_embeddings,
        "indices": {chain_id: indices[chain_id] for chain_id in chain_embeddings},
        "metadata": {
            "model_name": encoder.model_name,
            "embedding_size": encoder.embedding_size,
            "chain_separator": chain_separator,
            "bidirectional_average": bidirectional_average,
            "format": "plm-interact-pair-v1",
        },
    }
