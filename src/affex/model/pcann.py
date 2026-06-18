from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm, GATConv, MetaLayer, global_mean_pool

from affex.data.types import InterfaceGraph


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"


class RadialBasisExpansion(nn.Module):
    offset: torch.Tensor

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 32,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EdgeConvLayer(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim_in,
        edge_hidden_dim,
        edge_feature_dim_out,
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + edge_feature_dim_in, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_feature_dim_out),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out


class KdModel_PoolEdges(BaseModel):
    def __init__(
        self,
        node_feature_dim: int | None,
        node_vocab_size: int | None,
        node_embedding_dim: int,
        edge_feature_dim: int,
        num_layers: int = 4,
        linear_layer_nodes: bool = False,
        linear_layer_edges: bool = False,
        batchnorm: bool = True,
        input_projection_dim: int | None = None,
        use_foldx: bool = False,
        foldx_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert (node_vocab_size is None) ^ (node_feature_dim is None)

        self.num_layers = num_layers
        if node_vocab_size:
            node_feature_dim = node_embedding_dim
            self.node_embed = nn.Embedding(node_vocab_size, node_feature_dim)
        elif (
            input_projection_dim is not None and node_feature_dim is not None and node_feature_dim != node_embedding_dim
        ):
            self.input_proj = Sequential(
                Linear(node_feature_dim, input_projection_dim),
                ReLU(),
                Linear(input_projection_dim, node_embedding_dim),
            )
            node_feature_dim = node_embedding_dim

        assert node_feature_dim is not None
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        if self.batchnorms is not None:
            self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.convs.append(
            MetaLayer(
                EdgeConvLayer(
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim_in=edge_feature_dim,
                    edge_hidden_dim=node_embedding_dim,
                    edge_feature_dim_out=node_embedding_dim,
                ),
                GATConv(
                    # in_channels=(node_feature_dim, node_feature_dim),
                    in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs,
                ),
            )
        )

        for _ in range(self.num_layers - 1):
            meta = MetaLayer(
                EdgeConvLayer(
                    node_feature_dim=node_embedding_dim,
                    edge_feature_dim_in=node_embedding_dim,
                    edge_hidden_dim=node_embedding_dim,
                    edge_feature_dim_out=node_embedding_dim,
                ),
                GATConv(
                    # in_channels=(node_embedding_dim, node_embedding_dim),
                    in_channels=node_embedding_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs,
                ),
            )

            self.convs.append(meta)
            if self.linear_layer_nodes is not None:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if self.linear_layer_edges is not None:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if self.batchnorms is not None:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        # Optional: RBF decomposition of FoldX interaction energy + FC layer
        self.use_foldx = use_foldx
        if use_foldx:
            layers: list[nn.Module] = [
                RadialBasisExpansion(start=-50, stop=50, num_gaussians=node_embedding_dim),
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.SiLU(inplace=True),
            ]
            if foldx_dropout > 0:
                layers.append(nn.Dropout(foldx_dropout))
            self.graph_embed = nn.Sequential(*layers)

        self.mlp = Sequential(
            Linear(node_embedding_dim, node_embedding_dim),
            ReLU(),
            Linear(node_embedding_dim, node_embedding_dim // 2),
            ReLU(),
            Linear(node_embedding_dim // 2, 1),
        )

    def forward(self, data: InterfaceGraph):
        edge_index, edge_attr, batch = (
            data.edge_index,
            data.distances.view(-1, 1).float(),
            data.batch,
        )
        if hasattr(self, "node_embed"):
            if hasattr(data, 'atoms') and data.atoms is not None:
                node_types = data.atoms
            elif hasattr(data, 'residues') and data.residues is not None:
                node_types = data.residues
            else:
                raise ValueError("data must have atoms or residues for node_embed")
            x = self.node_embed.forward(node_types)
        else:
            x = data.residue_features
            if hasattr(self, "input_proj"):
                x = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)

            if self.batchnorms is not None:
                h = F.relu(self.batchnorms[i](h))
            else:
                h = F.relu(h)

            if self.linear_layer_nodes is not None:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges is not None:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        assert batch is not None
        batch_edge = batch[edge_index[0]]
        edge_attr = global_mean_pool(edge_attr, batch_edge)
        if self.use_foldx:
            assert data.foldx_energy is not None
            edge_attr = edge_attr + self.graph_embed(data.foldx_energy.float())

        return self.mlp(edge_attr)
