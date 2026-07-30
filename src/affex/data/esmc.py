from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class EsmcPairEncoder:
    """Encode two protein sides in one ESM-C sequence separated by a chain break."""

    model_name = "biohub/esmc-600m-2024-12"
    embedding_size = 1152
    output_format = "esmc-chain-break-pair-v1"

    def __init__(self, weights_path: Path, library_dir: Path, device: torch.device) -> None:
        self.device = device
        self._load_official_esmc_library(library_dir)

        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers

        self.tokenizer = get_esmc_model_tokenizers()
        self.model = ESMC(
            d_model=self.embedding_size,
            n_heads=18,
            n_layers=36,
            tokenizer=self.tokenizer,
            use_flash_attn=False,
        ).eval()

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(self._unwrap_state_dict(state), strict=True)
        del state

        self.model.to(device)
        if device.type == "cuda":
            self.model.to(torch.bfloat16)

    @staticmethod
    def _load_official_esmc_library(library_dir: Path) -> None:
        library_dir = library_dir.resolve()
        if not (library_dir / "esm" / "models" / "esmc.py").is_file():
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
                    "run ESM-C extraction in a fresh Python process"
                )

    @staticmethod
    def _unwrap_state_dict(state: Any) -> dict[str, Tensor]:
        if not isinstance(state, dict):
            raise TypeError(f"expected an ESM-C state dict, got {type(state).__name__}")
        for key in ("model", "state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        return state

    def encode_pair(self, sequence_a: str, sequence_b: str, max_length: int) -> tuple[Tensor, Tensor]:
        combined_sequence = f"{sequence_a}|{sequence_b}"
        token_ids = self.tokenizer.encode(combined_sequence, add_special_tokens=True)
        expected_length = len(sequence_a) + len(sequence_b) + 3
        if len(token_ids) != expected_length:
            raise ValueError(
                f"ESM-C tokenized length {len(token_ids)} does not match expected paired length {expected_length}"
            )
        if expected_length > max_length:
            raise ValueError(
                f"paired sequence length {expected_length} exceeds max_length={max_length}; "
                "increase --max-length or skip this item"
            )

        sequence_tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.no_grad()
        )
        with torch.inference_mode(), autocast_context:
            output = self.model(sequence_tokens=sequence_tokens)
        hidden = output.embeddings[0].detach().float().cpu()

        first_start = 1
        first_stop = first_start + len(sequence_a)
        second_start = first_stop + 1
        second_stop = second_start + len(sequence_b)
        return hidden[first_start:first_stop], hidden[second_start:second_stop]

    def encode_sequence(self, sequence: str, max_length: int = 2048) -> Tensor:
        """Encode one protein chain independently for the classic PCANN path."""
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
        expected_length = len(sequence) + 2
        if len(token_ids) != expected_length:
            raise ValueError(
                f"ESM-C tokenized length {len(token_ids)} does not match expected single-chain length "
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
