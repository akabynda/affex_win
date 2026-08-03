import torch
from torch_geometric.data import Data

from affex.model.pcann import EdgeConvLayer, KdModel_PoolEdges, RadialBasisExpansion


def test_memory_efficient_edge_conv_matches_concatenated_linear() -> None:
    layer = EdgeConvLayer(3, 5, 7, 4)
    src = torch.randn(11, 3)
    dest = torch.randn(11, 3)
    edge_attr = torch.randn(11, 5)

    expected = layer.edge_mlp(torch.cat([src, dest, edge_attr], dim=1))
    actual = layer(src, dest, edge_attr)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_rbf_expansion_peaks_at_its_centers() -> None:
    rbf = RadialBasisExpansion(start=0.0, stop=4.0, num_gaussians=5)
    values = rbf(torch.arange(5, dtype=torch.float32))

    assert values.shape == (5, 5)
    assert torch.equal(values.argmax(dim=1), torch.arange(5))


def test_pcann_forward_accepts_rbf_edge_features() -> None:
    model = KdModel_PoolEdges(
        node_feature_dim=3,
        node_vocab_size=None,
        node_embedding_dim=4,
        edge_feature_dim=5,
        distance_rbf_num_gaussians=5,
        distance_rbf_start=0.0,
        distance_rbf_stop=4.0,
        num_layers=1,
        heads=1,
        add_self_loops=False,
        concat=False,
        bias=True,
        batchnorm=False,
    )
    graph = Data(
        residue_features=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]),
        distances=torch.tensor([1.0, 2.0, 1.0, 2.0]),
        batch=torch.zeros(4, dtype=torch.long),
    )

    assert model(graph).shape == (1, 1)


def test_pcann_forward_without_initial_edge_features() -> None:
    model = KdModel_PoolEdges(
        node_feature_dim=3,
        node_vocab_size=None,
        node_embedding_dim=4,
        edge_feature_dim=0,
        num_layers=1,
        heads=1,
        add_self_loops=False,
        concat=False,
        bias=True,
        batchnorm=False,
    )
    graph = Data(
        residue_features=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]),
        batch=torch.zeros(4, dtype=torch.long),
    )

    assert model(graph).shape == (1, 1)
