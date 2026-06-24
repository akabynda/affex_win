# pyright: reportUndefinedVariable=false
from __future__ import annotations

from typing import Protocol

from jaxtyping import Bool, Float32, Int
from torch import Tensor, nn

from affex.data.measurement import DataItem


class InterfaceGraph(Protocol):
    coordinates: Float32[Tensor, "n 3"]
    receptor_mask: Bool[Tensor, n]
    edge_index: Int[Tensor, "2 e"]
    distances: Float32[Tensor, e]
    atoms: Float32[Tensor, n] | None = None
    atom_features: Float32[Tensor, "n d"] | None = None
    residues: Float32[Tensor, n] | None = None
    residue_features: Float32[Tensor, "n d"] | None = None
    esm_hidden_states: Float32[Tensor, "b l d"] | None = None
    esm_attention_mask: Int[Tensor, "b l"] | None = None
    esm_token_positions: Int[Tensor, n] | None = None
    foldx_energy: Float32[Tensor, N] | None = None
    batch: Float32[Tensor, n] | None = None


BatchType = tuple[InterfaceGraph, list[DataItem]]


class AtomicInterfacePredictor(nn.Module):
    def forward(self, graph: InterfaceGraph) -> Tensor:
        raise NotImplementedError
