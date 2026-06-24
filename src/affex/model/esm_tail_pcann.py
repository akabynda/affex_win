from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
from torch import Tensor, nn

from affex.data.types import AtomicInterfacePredictor, InterfaceGraph


class CachedEsmTailPcannModel(AtomicInterfacePredictor):
    def __init__(
        self,
        pcann: AtomicInterfacePredictor,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        tail_layers: int = 1,
        pcann_lr: float = 1e-3,
        esm_lr: float = 1e-5,
        pcann_checkpoint: str | None = None,
        pcann_checkpoint_dir: str | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        if tail_layers <= 0:
            raise ValueError("tail_layers must be positive")

        from transformers import AutoModel
        from transformers.models.esm.modeling_esm import create_bidirectional_mask

        base = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        if tail_layers > len(base.encoder.layer):
            raise ValueError(f"tail_layers={tail_layers} exceeds model depth={len(base.encoder.layer)}")

        self.pcann = pcann
        self.config = base.config
        self.rotary_embeddings = base.rotary_embeddings
        self.tail_layers = nn.ModuleList(list(base.encoder.layer[-tail_layers:]))
        self.emb_layer_norm_after = base.encoder.emb_layer_norm_after
        self.pcann_lr = pcann_lr
        self.esm_lr = esm_lr
        self._create_bidirectional_mask = create_bidirectional_mask
        self._load_pcann_checkpoint(pcann_checkpoint, pcann_checkpoint_dir)
        self.tail_layers.train()
        self.emb_layer_norm_after.train()

        del base
        gc.collect()

    def _load_pcann_checkpoint(self, checkpoint: str | None, checkpoint_dir: str | None) -> None:
        checkpoint_path = Path(checkpoint) if checkpoint else None
        if checkpoint_path is None and checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
            candidates = [p for p in ckpt_dir.glob("epoch_*.ckpt") if p.name != "last.ckpt"]
            if not candidates:
                raise FileNotFoundError(f"No epoch checkpoint found in {ckpt_dir}")
            checkpoint_path = max(
                candidates,
                key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else 0,
            )
        if checkpoint_path is None:
            return

        raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw_state["state_dict"] if isinstance(raw_state, dict) and "state_dict" in raw_state else raw_state
        model_state = self.pcann.state_dict()
        pcann_state = {}
        for key, value in state.items():
            clean_key = key.removeprefix("model.").removeprefix("pcann.")
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                pcann_state[clean_key] = value

        missing, unexpected = self.pcann.load_state_dict(pcann_state, strict=False)
        print(
            f"Loaded {len(pcann_state)} PCANN tensors from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def optimizer_parameters(self) -> list[dict[str, object]]:
        esm_params = list(self.tail_layers.parameters()) + list(self.emb_layer_norm_after.parameters())
        return [
            {"params": self.pcann.parameters(), "lr": self.pcann_lr},
            {"params": esm_params, "lr": self.esm_lr},
        ]

    def forward(self, graph: InterfaceGraph) -> Tensor:
        if graph.esm_hidden_states is None or graph.esm_attention_mask is None or graph.esm_token_positions is None:
            raise ValueError("graph must contain cached ESM hidden states, attention mask, and token positions")
        if graph.batch is None:
            raise ValueError("graph.batch is required for cached ESM tail model")

        dtype = next(self.tail_layers.parameters()).dtype
        hidden_states = graph.esm_hidden_states.to(dtype=dtype)
        attention_mask = graph.esm_attention_mask
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_embeddings(hidden_states, position_ids)
        extended_attention_mask = self._create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        for layer in self.tail_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.emb_layer_norm_after(hidden_states)

        graph.residue_features = hidden_states[graph.batch.long(), graph.esm_token_positions.long()]
        return self.pcann(graph)
