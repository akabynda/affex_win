import gemmi
import torch

from affex.data.transform.graph_builder import ResidueInterfaceEsmGraphBuilder


def _atom(name: str, element: str, xyz: tuple[float, float, float]) -> gemmi.Atom:
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*xyz)
    return atom


def test_heavy_atom_centroid_is_unweighted_mean_of_cnos_atoms() -> None:
    residue = gemmi.Residue()
    residue.name = "CYS"
    residue.seqid = gemmi.SeqId(1, " ")
    residue.add_atom(_atom("CA", "C", (0.0, 0.0, 0.0)))
    residue.add_atom(_atom("N", "N", (2.0, 0.0, 0.0)))
    residue.add_atom(_atom("O", "O", (0.0, 4.0, 0.0)))
    residue.add_atom(_atom("SG", "S", (0.0, 0.0, 6.0)))
    residue.add_atom(_atom("H", "H", (100.0, 100.0, 100.0)))

    builder = ResidueInterfaceEsmGraphBuilder(
        radius=5.0,
        esm_dir="unused",
        coordinate_mode="heavy_atom_centroid",
    )
    torch.testing.assert_close(
        torch.tensor(builder.residue_coordinate(residue)),
        torch.tensor([0.5, 1.0, 1.5]),
    )
