from collections import defaultdict
from typing import Protocol

import gemmi
import numpy as np
from numpy import typing as npt


class EmbeddingModel(Protocol):
    def predict(self, sequence: str) -> npt.NDArray: ...


def get_alignment(st: gemmi.Structure) -> dict[str, gemmi.AlignmentResult]:
    alignment = {}
    for entity in st.entities:
        if entity.polymer_type.name.startswith("Peptide"):
            alignment[entity.name] = gemmi.align_sequence_to_polymer(
                entity.full_sequence,
                st[0][entity.name].get_polymer(),
                entity.polymer_type,
                gemmi.AlignmentScoring(),
            )
    return alignment


def get_sequences(st: gemmi.Structure) -> dict[str, str]:
    sequences = {}
    for chain in st[0]:
        three_letter_seq = [residue.name for residue in chain if residue.name]
        sequences[chain.name] = gemmi.one_letter_code(three_letter_seq)
    return sequences


def get_full_sequences(st: gemmi.Structure):
    sequences = {}
    entites_dict = {
        entity.name: entity.full_sequence
        for entity in st.entities
        if entity.polymer_type.name.startswith("Peptide")
    }
    for chain_id, full_sequence in entites_dict.items():
        chain_sequence = gemmi.one_letter_code(full_sequence)
        sequences[chain_id] = chain_sequence
    return sequences


def get_alignment_indices(
    sequences: dict[str, str], alignment: dict[str, gemmi.AlignmentResult]
) -> dict[str, npt.NDArray]:
    indexes_by_chains = {}
    for chain_id, result in alignment.items():
        seq_with_gaps = result.add_gaps(sequences[chain_id], 2)
        indexes = np.array([ind for ind, resname in enumerate(seq_with_gaps) if resname != "-"])
        indexes_by_chains[chain_id] = indexes
    return indexes_by_chains


def embed_sequences(
    st: gemmi.Structure,
    pretrained_model: EmbeddingModel,
) -> dict[str, dict[str, npt.NDArray | str]]:
    pretrained_embeddings = {
        "sequences": {},
        "embeddings": {},
        "indices": {},
    }
    full_sequences = get_full_sequences(st)
    sequences = get_sequences(st)
    alignment = get_alignment(st)
    pretrained_embeddings["indices"] = get_alignment_indices(sequences, alignment)

    # unique sequences embeddings
    seq_to_chains = defaultdict(list)
    for chain_id, seq in full_sequences.items():
        seq_to_chains[seq].append(chain_id)

    # save full sequences and embeddings
    for sequence, chain_ids in seq_to_chains.items():
        embeddings = pretrained_model.predict(sequence)
        # save embeds
        chains_key = "|".join(chain_ids)
        pretrained_embeddings["sequences"][chains_key] = sequence
        pretrained_embeddings["embeddings"][chains_key] = embeddings

    return pretrained_embeddings
