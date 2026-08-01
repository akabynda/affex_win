from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import gemmi
import torch
from torch import Tensor, nn

from affex.data.esm2 import get_alignment, get_alignment_indices, get_full_sequences, get_sequences
from affex.data.model_sources import PLM_INTERACT_CHECKPOINT
from affex.data.types import DataItem


@dataclass(frozen=True)
class ChainSpan:
    chain_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class ChainLinker:
    source_chain: str
    target_chain: str
    terminal_distance_angstrom: float
    target_length_angstrom: float
    repeats: int
    sequence: str


LINKER_LENGTH_SCALE = math.pi / 2
DEFAULT_LINKER_REPEAT = "GGGS"
DEFAULT_RESIDUE_CONTOUR_LENGTH_ANGSTROM = 3.8


class PairEncoder(Protocol):
    model_name: str
    embedding_size: int

    def encode_pair(self, sequence_a: str, sequence_b: str, max_length: int) -> tuple[Tensor, Tensor]: ...

    def encode_sequence(self, sequence: str, max_length: int) -> Tensor: ...


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
            PLM_INTERACT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=checkpoint_repo,
                    filename=PLM_INTERACT_CHECKPOINT.name,
                    local_dir=PLM_INTERACT_CHECKPOINT.parent,
                )
            )
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

    def encode_sequence(self, sequence: str, max_length: int) -> Tensor:
        """Encode one residue sequence and return embeddings without BOS/EOS tokens."""
        expected_length = len(sequence) + 2
        if expected_length > max_length:
            raise ValueError(
                f"single sequence length {expected_length} exceeds max_length={max_length}; "
                "increase --max-length or skip this item"
            )

        tokenized = self.tokenizer(
            sequence,
            padding=False,
            truncation=False,
            return_tensors="pt",
            max_length=max_length,
        )
        token_count = int(tokenized["input_ids"].shape[1])
        if token_count != expected_length:
            raise ValueError(
                f"tokenized length {token_count} does not match expected single-sequence ESM2 length "
                f"{expected_length}"
            )

        features = {name: value.to(self.device) for name, value in tokenized.items()}
        with torch.inference_mode():
            output = self.model.esm_mask.base_model(**features, return_dict=True)
        return output.last_hidden_state[0, 1:-1].detach().cpu()

    def encode_linked_pair(
        self,
        sequence_a: str,
        sequence_b: str,
        linker: str,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        """Encode both proteins as one ESM2 sequence with a residue linker between them."""
        combined_sequence = f"{sequence_a}{linker}{sequence_b}"
        hidden = self.encode_sequence(combined_sequence, max_length)
        first_stop = len(sequence_a)
        second_start = first_stop + len(linker)
        second_stop = second_start + len(sequence_b)
        return hidden[:first_stop], hidden[second_start:second_stop]


def build_side_sequence(
    full_sequences: dict[str, str],
    chain_ids: list[str],
    chain_separator: str | Sequence[str],
) -> tuple[str, list[ChainSpan]]:
    if isinstance(chain_separator, str):
        separators = [chain_separator] * max(0, len(chain_ids) - 1)
    else:
        separators = list(chain_separator)
        expected = max(0, len(chain_ids) - 1)
        if len(separators) != expected:
            raise ValueError(f"expected {expected} chain separators for {len(chain_ids)} chains, got {len(separators)}")

    parts: list[str] = []
    spans: list[ChainSpan] = []
    cursor = 0
    for index, chain_id in enumerate(chain_ids):
        if index > 0 and separators[index - 1]:
            separator = separators[index - 1]
            parts.append(separator)
            cursor += len(separator)

        sequence = full_sequences[chain_id]
        start = cursor
        stop = start + len(sequence)
        spans.append(ChainSpan(chain_id=chain_id, start=start, stop=stop))
        parts.append(sequence)
        cursor = stop

    return "".join(parts), spans


def _terminal_ca_position(
    structure: gemmi.Structure,
    chain_id: str,
    terminus: Literal["start", "end"],
) -> gemmi.Position:
    chain = structure[0].find_chain(chain_id)
    if chain is None:
        raise KeyError(f"chain {chain_id!r} is missing from the structure model")

    residues = list(chain.get_polymer())
    if not residues:
        # This fallback keeps the helper usable for minimal structures without
        # entity/polymer annotations while still requiring an alpha carbon.
        residues = list(chain)
    ordered_residues = residues if terminus == "start" else reversed(residues)
    for residue in ordered_residues:
        ca = residue.find_atom("CA", "*")
        if ca is not None:
            return ca.pos

    raise ValueError(f"chain {chain_id!r} has no polymer residue with a CA atom")


def build_distance_aware_linkers(
    structure: gemmi.Structure,
    chain_ids: list[str],
    repeat: str = DEFAULT_LINKER_REPEAT,
    residue_contour_length_angstrom: float = DEFAULT_RESIDUE_CONTOUR_LENGTH_ANGSTROM,
) -> list[ChainLinker]:
    if not repeat:
        raise ValueError("linker repeat must not be empty")
    if residue_contour_length_angstrom <= 0:
        raise ValueError("residue contour length must be positive")

    repeat_length = len(repeat) * residue_contour_length_angstrom
    linkers: list[ChainLinker] = []
    for source_chain, target_chain in zip(chain_ids, chain_ids[1:], strict=False):
        source_end = _terminal_ca_position(structure, source_chain, "end")
        target_start = _terminal_ca_position(structure, target_chain, "start")
        distance = source_end.dist(target_start)
        target_length = distance * LINKER_LENGTH_SCALE
        repeats = math.ceil(target_length / repeat_length)
        linkers.append(
            ChainLinker(
                source_chain=source_chain,
                target_chain=target_chain,
                terminal_distance_angstrom=distance,
                target_length_angstrom=target_length,
                repeats=repeats,
                sequence=repeat * repeats,
            )
        )
    return linkers


def split_side_embeddings(side_embeddings: Tensor, spans: list[ChainSpan]) -> dict[str, Tensor]:
    return {span.chain_id: side_embeddings[span.start : span.stop].clone() for span in spans}


def ordered_peptide_chain_ids(structure: gemmi.Structure, full_sequences: dict[str, str]) -> list[str]:
    """Return peptide chain IDs in their order in the first structure model."""
    return [chain.name for chain in structure[0] if chain.name in full_sequences]


def average_chain_embeddings(first: dict[str, Tensor], second: dict[str, Tensor]) -> dict[str, Tensor]:
    return {chain_id: (first[chain_id] + second[chain_id]) / 2 for chain_id in first}


def encode_complex_embeddings(
    item: DataItem,
    structure: gemmi.Structure,
    encoder: PairEncoder,
    max_length: int,
    chain_separator: str = "X",
    bidirectional_average: bool = False,
    distance_aware_linker: bool = False,
    inter_protein_distance_aware_linker: bool = False,
    linker_repeat: str = DEFAULT_LINKER_REPEAT,
    residue_contour_length_angstrom: float = DEFAULT_RESIDUE_CONTOUR_LENGTH_ANGSTROM,
    all_structure_chains: bool = False,
) -> dict[str, Any]:
    if inter_protein_distance_aware_linker and not distance_aware_linker:
        raise ValueError("inter-protein distance-aware linker requires distance_aware_linker=True")
    if all_structure_chains and inter_protein_distance_aware_linker:
        raise ValueError("all-structure-chain encoding has no separate inter-protein boundary")
    if all_structure_chains and bidirectional_average:
        raise ValueError("bidirectional averaging is not defined for all-structure-chain encoding")

    full_sequences = get_full_sequences(structure)
    sequences = get_sequences(structure)
    alignment = get_alignment(structure)
    indices = get_alignment_indices(sequences, alignment)

    missing = [chain for chain in item.receptor_chains + item.ligand_chains if chain not in full_sequences]
    if missing:
        raise KeyError(f"chains missing from SEQRES/full sequences for {item.uid}: {missing}")

    if all_structure_chains:
        structure_chains = ordered_peptide_chain_ids(structure, full_sequences)
        if not structure_chains:
            raise ValueError(f"no peptide chains found for {item.uid}")

        structure_linkers = (
            build_distance_aware_linkers(
                structure,
                structure_chains,
                linker_repeat,
                residue_contour_length_angstrom,
            )
            if distance_aware_linker
            else []
        )
        separators: str | list[str] = (
            [linker.sequence for linker in structure_linkers]
            if distance_aware_linker
            else chain_separator
        )
        combined_sequence, chain_spans = build_side_sequence(
            full_sequences,
            structure_chains,
            separators,
        )
        combined_embeddings = encoder.encode_sequence(combined_sequence, max_length)
        chain_embeddings = split_side_embeddings(combined_embeddings, chain_spans)

        return {
            "sequences": {chain_id: full_sequences[chain_id] for chain_id in structure_chains},
            "embeddings": chain_embeddings,
            "indices": {chain_id: indices[chain_id] for chain_id in structure_chains},
            "metadata": {
                "model_name": encoder.model_name,
                "embedding_size": encoder.embedding_size,
                "all_structure_chains": True,
                "chain_separator": None if distance_aware_linker else chain_separator,
                "distance_aware_linker": distance_aware_linker,
                "pair_boundary": None,
                "linker_repeat": linker_repeat if distance_aware_linker else None,
                "linker_length_scale": LINKER_LENGTH_SCALE if distance_aware_linker else None,
                "residue_contour_length_angstrom": (
                    residue_contour_length_angstrom if distance_aware_linker else None
                ),
                "structure_linkers": [linker.__dict__ for linker in structure_linkers],
                "original_receptor_chains": item.receptor_chains,
                "original_ligand_chains": item.ligand_chains,
                "encoded_structure_chains": structure_chains,
                "combined_sequence_length": len(combined_sequence),
                "format": "esm2-all-structure-chains-v1",
            },
        }

    receptor_chains = item.receptor_chains
    ligand_chains = item.ligand_chains

    receptor_linkers: list[ChainLinker] = []
    ligand_linkers: list[ChainLinker] = []
    inter_protein_linkers: list[ChainLinker] = []
    reverse_inter_protein_linkers: list[ChainLinker] = []
    receptor_separators: str | list[str] = chain_separator
    ligand_separators: str | list[str] = chain_separator
    if distance_aware_linker:
        receptor_linkers = build_distance_aware_linkers(
            structure, receptor_chains, linker_repeat, residue_contour_length_angstrom
        )
        ligand_linkers = build_distance_aware_linkers(
            structure, ligand_chains, linker_repeat, residue_contour_length_angstrom
        )
        receptor_separators = [linker.sequence for linker in receptor_linkers]
        ligand_separators = [linker.sequence for linker in ligand_linkers]
        if inter_protein_distance_aware_linker:
            inter_protein_linkers = build_distance_aware_linkers(
                structure,
                [receptor_chains[-1], ligand_chains[0]],
                linker_repeat,
                residue_contour_length_angstrom,
            )

    receptor_sequence, receptor_spans = build_side_sequence(full_sequences, receptor_chains, receptor_separators)
    ligand_sequence, ligand_spans = build_side_sequence(full_sequences, ligand_chains, ligand_separators)

    if inter_protein_distance_aware_linker:
        encode_linked_pair = getattr(encoder, "encode_linked_pair", None)
        if encode_linked_pair is None:
            raise TypeError(f"{type(encoder).__name__} does not support residue-linked pair encoding")
        receptor_embeddings, ligand_embeddings = encode_linked_pair(
            receptor_sequence,
            ligand_sequence,
            inter_protein_linkers[0].sequence,
            max_length,
        )
    else:
        receptor_embeddings, ligand_embeddings = encoder.encode_pair(receptor_sequence, ligand_sequence, max_length)
    chain_embeddings = {
        **split_side_embeddings(receptor_embeddings, receptor_spans),
        **split_side_embeddings(ligand_embeddings, ligand_spans),
    }

    if bidirectional_average:
        if inter_protein_distance_aware_linker:
            reverse_inter_protein_linkers = build_distance_aware_linkers(
                structure,
                [ligand_chains[-1], receptor_chains[0]],
                linker_repeat,
                residue_contour_length_angstrom,
            )
            ligand_embeddings_rev, receptor_embeddings_rev = encode_linked_pair(
                ligand_sequence,
                receptor_sequence,
                reverse_inter_protein_linkers[0].sequence,
                max_length,
            )
        else:
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
            "chain_separator": None if distance_aware_linker else chain_separator,
            "distance_aware_linker": distance_aware_linker,
            "inter_protein_distance_aware_linker": inter_protein_distance_aware_linker,
            "pair_boundary": "distance_aware_linker" if inter_protein_distance_aware_linker else "eos",
            "linker_repeat": linker_repeat if distance_aware_linker else None,
            "linker_length_scale": LINKER_LENGTH_SCALE if distance_aware_linker else None,
            "residue_contour_length_angstrom": (
                residue_contour_length_angstrom if distance_aware_linker else None
            ),
            "receptor_linkers": [linker.__dict__ for linker in receptor_linkers],
            "ligand_linkers": [linker.__dict__ for linker in ligand_linkers],
            "inter_protein_linkers": [linker.__dict__ for linker in inter_protein_linkers],
            "reverse_inter_protein_linkers": [
                linker.__dict__ for linker in reverse_inter_protein_linkers
            ],
            "bidirectional_average": bidirectional_average,
            "all_structure_chains": False,
            "original_receptor_chains": item.receptor_chains,
            "original_ligand_chains": item.ligand_chains,
            "encoded_receptor_chains": receptor_chains,
            "encoded_ligand_chains": ligand_chains,
            "format": (
                "plm-interact-linked-single-v1"
                if inter_protein_distance_aware_linker
                else getattr(encoder, "output_format", "plm-interact-pair-v1")
            ),
        },
    }
