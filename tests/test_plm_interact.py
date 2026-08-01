import math
from types import SimpleNamespace

import gemmi
import pytest
import torch

from affex.data.plm_interact import PlmInteractPairEncoder, build_distance_aware_linkers, build_side_sequence


def _two_chain_structure(distance: float) -> gemmi.Structure:
    structure = gemmi.Structure()
    model = gemmi.Model("1")

    for chain_id, x in (("A", 0.0), ("B", distance)):
        chain = gemmi.Chain(chain_id)
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.seqid = gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(x, 0.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    return structure


def test_distance_aware_linker_uses_scaled_terminal_distance() -> None:
    structure = _two_chain_structure(20.0)

    linkers = build_distance_aware_linkers(structure, ["A", "B"])

    assert len(linkers) == 1
    assert linkers[0].terminal_distance_angstrom == pytest.approx(20.0)
    assert linkers[0].target_length_angstrom == pytest.approx(20.0 * math.pi / 2)
    assert linkers[0].repeats == 3
    assert linkers[0].sequence == "GGGSGGGSGGGS"


def test_side_sequence_accepts_one_separator_per_boundary() -> None:
    sequence, spans = build_side_sequence(
        {"A": "AA", "B": "BBB", "C": "C"},
        ["A", "B", "C"],
        ["GGGS", "GGGSGGGS"],
    )

    assert sequence == "AAGGGSBBBGGGSGGGSC"
    assert [(span.chain_id, span.start, span.stop) for span in spans] == [
        ("A", 0, 2),
        ("B", 6, 9),
        ("C", 17, 18),
    ]


def test_side_sequence_rejects_wrong_separator_count() -> None:
    with pytest.raises(ValueError, match="expected 2 chain separators"):
        build_side_sequence({"A": "A", "B": "B", "C": "C"}, ["A", "B", "C"], ["GGGS"])


class _FakeTokenizer:
    def __call__(self, sequence: str, **_: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.arange(len(sequence) + 2).unsqueeze(0)}


class _FakeBaseModel:
    def __call__(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        length = input_ids.shape[1]
        hidden = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_linked_pair_encoder_excludes_bos_linker_and_eos() -> None:
    encoder = PlmInteractPairEncoder.__new__(PlmInteractPairEncoder)
    encoder.device = torch.device("cpu")
    encoder.tokenizer = _FakeTokenizer()
    encoder.model = SimpleNamespace(esm_mask=SimpleNamespace(base_model=_FakeBaseModel()))

    first, second = encoder.encode_linked_pair("AA", "BBB", "GGGS", max_length=11)

    assert first.flatten().tolist() == [1.0, 2.0]
    assert second.flatten().tolist() == [7.0, 8.0, 9.0]


def test_single_sequence_encoder_excludes_bos_and_eos() -> None:
    encoder = PlmInteractPairEncoder.__new__(PlmInteractPairEncoder)
    encoder.device = torch.device("cpu")
    encoder.tokenizer = _FakeTokenizer()
    encoder.model = SimpleNamespace(esm_mask=SimpleNamespace(base_model=_FakeBaseModel()))

    embeddings = encoder.encode_sequence("AAXBBB", max_length=9)

    assert embeddings.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
