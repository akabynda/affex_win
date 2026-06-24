from __future__ import annotations

import gc
import math
import re
from pathlib import Path

import torch
from torch import Tensor, nn

from affex.data.types import AtomicInterfacePredictor, InterfaceGraph


class LoraLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class CachedEsmLoraPcannModel(AtomicInterfacePredictor):
    def __init__(
        self,
        pcann: AtomicInterfacePredictor,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        tail_layers: int = 1,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_lr: float = 1e-4,
        pcann_lr: float = 1e-5,
        train_pcann: bool = False,
        lora_targets: list[str] | None = None,
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

        for param in base.parameters():
            param.requires_grad = False

        self.pcann = pcann
        self.config = base.config
        self.rotary_embeddings = base.rotary_embeddings
        self.tail_layers = nn.ModuleList(list(base.encoder.layer[-tail_layers:]))
        self.emb_layer_norm_after = base.encoder.emb_layer_norm_after
        self.lora_lr = lora_lr
        self.pcann_lr = pcann_lr
        self.train_pcann = train_pcann
        self._create_bidirectional_mask = create_bidirectional_mask

        self._load_pcann_checkpoint(pcann_checkpoint, pcann_checkpoint_dir)
        for param in self.pcann.parameters():
            param.requires_grad = train_pcann
        if not train_pcann:
            self.pcann.eval()

        targets = lora_targets or ["query", "value"]
        self._inject_lora(targets, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        del base
        gc.collect()

    def _inject_lora(self, targets: list[str], rank: int, alpha: float, dropout: float) -> None:
        injected = 0
        for layer in self.tail_layers:
            self_attention = layer.attention.self
            for target in targets:
                if not hasattr(self_attention, target):
                    raise ValueError(f"Unsupported LoRA target for ESM self-attention: {target}")
                module = getattr(self_attention, target)
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at target {target}, got {type(module)}")
                setattr(self_attention, target, LoraLinear(module, rank=rank, alpha=alpha, dropout=dropout))
                injected += 1
        print(f"Injected LoRA into {injected} ESM attention projections")

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
        groups: list[dict[str, object]] = [
            {
                "params": [param for name, param in self.named_parameters() if "lora_" in name and param.requires_grad],
                "lr": self.lora_lr,
            }
        ]
        if self.train_pcann:
            groups.append({"params": self.pcann.parameters(), "lr": self.pcann_lr})
        return groups

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_pcann:
            self.pcann.eval()
        return self

    def forward(self, graph: InterfaceGraph) -> Tensor:
        if graph.esm_hidden_states is None or graph.esm_attention_mask is None or graph.esm_token_positions is None:
            raise ValueError("graph must contain cached ESM hidden states, attention mask, and token positions")
        if graph.batch is None:
            raise ValueError("graph.batch is required for cached ESM LoRA model")

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
