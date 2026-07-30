from types import SimpleNamespace

import torch

from affex.data.esmc import EsmcPairEncoder


class _FakeTokenizer:
    def encode(self, sequence: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens
        return list(range(len(sequence) + 2))


class _FakeModel:
    def __call__(self, sequence_tokens: torch.Tensor) -> SimpleNamespace:
        length = sequence_tokens.shape[1]
        embeddings = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
        return SimpleNamespace(embeddings=embeddings)


def test_esmc_pair_encoder_excludes_bos_chain_break_and_eos() -> None:
    encoder = EsmcPairEncoder.__new__(EsmcPairEncoder)
    encoder.device = torch.device("cpu")
    encoder.tokenizer = _FakeTokenizer()
    encoder.model = _FakeModel()

    first, second = encoder.encode_pair("AA", "BBB", max_length=8)

    assert first.flatten().tolist() == [1.0, 2.0]
    assert second.flatten().tolist() == [4.0, 5.0, 6.0]


def test_esmc_sequence_encoder_excludes_bos_and_eos() -> None:
    encoder = EsmcPairEncoder.__new__(EsmcPairEncoder)
    encoder.device = torch.device("cpu")
    encoder.tokenizer = _FakeTokenizer()
    encoder.model = _FakeModel()

    embeddings = encoder.encode_sequence("AAAA", max_length=6)

    assert embeddings.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
