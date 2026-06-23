import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import gemmi
import numpy as np
import torch
from numpy import typing as npt
from torch import Tensor

from affex.data.transform.graph_builder import read_structure


class EmbeddingModel(Protocol):
    def predict(self, sequence: str) -> npt.NDArray: ...


def get_alignment(st: gemmi.Structure) -> dict[str, gemmi.AlignmentResult]:
    alignment = {}
    for entity in st.entities:
        if entity.polymer_type.name == "PeptideL":
            alignment[entity.name] = gemmi.align_sequence_to_polymer(
                entity.full_sequence,
                st[0][entity.name].get_polymer(),
                gemmi.PolymerType.PeptideL,
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
        entity.name: entity.full_sequence for entity in st.entities if entity.polymer_type.name == "PeptideL"
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


class EsmRunner:
    def __init__(self, modeldir: str, savedir: str, use_gpu: bool = True):
        self.modeldir = modeldir
        self.savedir = Path(savedir)
        self.use_gpu = use_gpu

    def predict(self, sequence: str) -> Tensor:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "tmp.fasta"), "w") as tmp_fasta_file:
                tmp_fasta_file.write(">tmp")
                tmp_fasta_file.write("\n")
                tmp_fasta_file.write(sequence)
                tmp_fasta_file.write("\n")

            # command
            os.putenv("PYTHONPATH", os.pathsep.join([os.getenv("PYTHONPATH", ""), self.modeldir]))

            fasta_file = os.path.join(tmp_dir, "tmp.fasta")
            command = [
                sys.executable,
                os.path.join(self.modeldir, "scripts", "extract.py"),
                "esm2_t33_650M_UR50D",
                fasta_file,
                tmp_dir,
                "--include",
                "per_tok",
            ]
            if not self.use_gpu:
                command.append("--nogpu")

            subprocess.check_call(command)
            repr_key = 33

            embeddings = torch.load(os.path.join(tmp_dir, "tmp.pt"))["representations"][repr_key]
        return embeddings

    def process_one(self, pdb: Path) -> None:
        savefile = self.savedir / f"{pdb.stem}.pt"
        if savefile.is_file():
            return
        st = read_structure(pdb)
        try:
            embeds = embed_sequences(st, self)  # type: ignore[arg-type]
            torch.save(embeds, self.savedir / f"{pdb.stem}.pt")
        except Exception as err:
            print(pdb, err)
