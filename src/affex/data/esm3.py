"""Local sequence-only ESM-3 encoder used by classic PCANN."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import torch
from torch import Tensor


class _EmbeddingsOnlyHead(torch.nn.Module):
    """Avoid computing unused sequence, structure, and function logits."""

    def forward(self, x: Tensor, embed: Tensor, attentions=None):  # noqa: ARG002
        return SimpleNamespace(embeddings=embed)


class Esm3SequenceEncoder:
    """Extract final per-residue embeddings from ESM3-sm-open-v1."""

    model_name = "biohub/esm3-sm-open-v1"
    embedding_size = 1536
    output_format = "esm3-classic-single-chain-v1"

    def __init__(self, weights_path: Path, library_dir: Path, device: torch.device) -> None:
        self.device = device
        self._load_official_library(library_dir)

        from esm.models.esm3 import ESM3
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        self.tokenizer = EsmSequenceTokenizer()
        tokenizers = SimpleNamespace(sequence=self.tokenizer)

        def unavailable_component(_: torch.device | str) -> NoReturn:
            raise RuntimeError("sequence-only ESM-3 extraction does not load structure/function decoders")

        with torch.device("meta"):
            self.model = ESM3(
                d_model=self.embedding_size,
                n_heads=24,
                v_heads=256,
                n_layers=48,
                structure_encoder_fn=unavailable_component,
                structure_decoder_fn=unavailable_component,
                function_decoder_fn=unavailable_component,
                tokenizers=tokenizers,
            ).eval()
        # Materialize non-persistent buffers (for example rotary frequencies)
        # that are intentionally absent from the checkpoint state dict.
        self.model.to_empty(device="cpu")

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(self._unwrap_state_dict(state), assign=True, strict=True)
        del state
        self.model.output_heads = _EmbeddingsOnlyHead()
        self.model.to(device)
        if device.type == "cuda":
            self.model.to(torch.bfloat16)

    @staticmethod
    def _load_official_library(library_dir: Path) -> None:
        library_dir = library_dir.resolve()
        if not (library_dir / "esm" / "models" / "esm3.py").is_file():
            raise FileNotFoundError(f"official Biohub ESM library not found in {library_dir}")
        library_text = str(library_dir)
        if library_text not in sys.path:
            sys.path.insert(0, library_text)

        loaded_esm = sys.modules.get("esm")
        if loaded_esm is not None:
            loaded_path = Path(getattr(loaded_esm, "__file__", "")).resolve()
            if library_dir not in loaded_path.parents:
                raise RuntimeError(
                    f"a different esm package is already imported from {loaded_path}; "
                    "run ESM-3 extraction in a fresh Python process"
                )

    @staticmethod
    def _unwrap_state_dict(state: Any) -> dict[str, Tensor]:
        if not isinstance(state, dict):
            raise TypeError(f"expected an ESM-3 state dict, got {type(state).__name__}")
        for key in ("model", "state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        return state

    def predict(self, sequence: str, max_length: int = 2048) -> Tensor:
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
        expected_length = len(sequence) + 2
        if len(token_ids) != expected_length:
            raise ValueError(
                f"ESM-3 tokenized length {len(token_ids)} does not match single-chain length "
                f"{expected_length}"
            )
        if expected_length > max_length:
            raise ValueError(
                f"single-chain sequence length {expected_length} exceeds max_length={max_length}; "
                "increase --max-length or skip this structure"
            )

        sequence_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )
        with torch.inference_mode(), autocast_context:
            output = self.model(sequence_tokens=sequence_tokens)
        return output.embeddings[0, 1:-1].detach().float().cpu()
