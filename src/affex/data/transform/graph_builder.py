from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi
import loguru
import torch
from jaxtyping import Bool, Float32, Int
from torch import Tensor
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected

from affex.data.constants import RESIDUE_INDICES
from affex.data.types import DataItem, InterfaceGraph


def read_structure(pdb: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(pdb))
    # setup entities, but do not remove duplicate chains
    # structure.setup_entities()
    structure.add_entity_types()
    structure.assign_subchains()
    structure.ensure_entities()
    # clean
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    structure.cell = gemmi.UnitCell()  # for correct contacts search
    return structure


def find_contacts(
    structure: gemmi.Structure, receptor_chains: list[str], ligand_chains: list[str], radius: float
) -> list[gemmi.ContactSearch.Result]:
    cs = gemmi.ContactSearch(radius)
    cs.ignore = gemmi.ContactSearch.Ignore.SameChain
    cs.twice = True

    # keep only necessary chains
    sel = gemmi.Selection(",".join(receptor_chains + ligand_chains))
    sel.remove_not_selected(structure)

    ns = gemmi.NeighborSearch(structure, radius).populate()
    contacts = cs.find_contacts(ns)
    contacts = [x for x in contacts if is_outer_contact(x, receptor_chains)]
    return contacts


def is_outer_contact(contact: gemmi.ContactSearch.Result, receptor_chains: list[str]) -> bool:
    src_is_receptor = contact.partner1.chain.name in receptor_chains
    dst_is_receptor = contact.partner2.chain.name in receptor_chains
    return src_is_receptor ^ dst_is_receptor


@dataclass
class _InterfaceGraph:
    coordinates: Float32[Tensor, "n 3"]
    receptor_mask: Bool[Tensor, "n"]
    edge_index: Int[Tensor, "2 e"]
    distances: Float32[Tensor, "e"]
    atoms: Float32[Tensor, "n"] | None = None
    atom_features: Float32[Tensor, "n d"] | None = None
    residues: Float32[Tensor, "n"] | None = None
    residue_features: Float32[Tensor, "n d"] | None = None
    foldx_energy: Float32[Tensor, "N"] | None = None
    batch: Float32[Tensor, "n"] | None = None


class InterfaceGraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, item: DataItem) -> InterfaceGraph | None: ...


# fmt: off
ATOM_NAMES = [
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2", "CH2",
    "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT",
    "SD", "SG",
]
# fmt: on
ATOMS_INDICES = {x: i for i, x in enumerate(ATOM_NAMES, start=1)}


class AtomicInterfaceGraphBuilder:
    def __init__(
        self,
        interface_distance: float,
        radius: float,
        max_neighbors: int = 30,
        keep_inner_edges: bool = False,
    ):
        super().__init__()
        self.interface_distance = interface_distance
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.keep_inner_edges = keep_inner_edges

    def build_graph(self, item: DataItem) -> InterfaceGraph | None:
        # prepare contacts
        structure = read_structure(item.pdb)
        contacts = find_contacts(structure, item.receptor_chains, item.ligand_chains, self.interface_distance)
        if len(contacts) == 0:
            loguru.logger.warning(f"Skipping {item.uid}: no contacts")
            return None
        # create mapping from atom to index
        atom_to_id: dict[tuple[gemmi.Atom, str, str], int] = {}
        for contact in contacts:
            key = (
                contact.partner1.atom,
                contact.partner1.residue.name,
                contact.partner1.chain.name,
            )
            if key not in atom_to_id:
                atom_to_id[key] = len(atom_to_id)

        # build graph
        atom_indices = torch.tensor([ATOMS_INDICES.get(atom.name, 0) for atom, _, _ in atom_to_id.keys()])
        residue_names = gemmi.one_letter_code([residue_name for _, residue_name, _ in atom_to_id.keys()])
        residue_indices = torch.tensor([RESIDUE_INDICES.get(x, 0) for x in residue_names])
        coordinates = torch.tensor([x.pos.tolist() for x, _, _ in atom_to_id.keys()], dtype=torch.float32)
        receptor_mask = torch.tensor([int(chain_id in item.receptor_chains) for _, _, chain_id in atom_to_id.keys()])

        edge_index = to_undirected(radius_graph(coordinates, max_num_neighbors=self.max_neighbors, r=self.radius))
        if not self.keep_inner_edges:
            src, dst = edge_index
            is_intermol = receptor_mask[src] != receptor_mask[dst]
            edge_index = edge_index[:, is_intermol]

        # distances between atoms
        src, dst = edge_index
        distances = (coordinates[src] - coordinates[dst]).norm(dim=1)

        return _InterfaceGraph(
            atoms=atom_indices,
            residues=residue_indices,
            coordinates=coordinates,
            receptor_mask=receptor_mask,
            edge_index=edge_index,
            distances=distances,
        )


class ResidueInterfaceGraphBuilder(InterfaceGraphBuilder):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def build_graph(self, item: DataItem) -> InterfaceGraph | None:
        structure = read_structure(item.pdb)
        contacts = find_contacts(structure, item.receptor_chains, item.ligand_chains, self.radius)
        # transform atomic contacts to residue contacts
        residue_contacts: dict[tuple[tuple[gemmi.Residue, str], tuple[gemmi.Residue, str]], float] = {}
        for contact in contacts:
            src = contact.partner1.residue, contact.partner1.chain.name
            dst = contact.partner2.residue, contact.partner2.chain.name
            if not contact.partner1.residue.get_ca() or not contact.partner2.residue.get_ca():
                continue
            prev_dist = residue_contacts.get((src, dst), contact.dist)
            residue_contacts[(src, dst)] = min(prev_dist, contact.dist)

        # create mapping from residue to index
        residue_to_id: dict[tuple[gemmi.Residue, str], int] = {}
        for src, _ in residue_contacts:
            if src not in residue_to_id:
                residue_to_id[src] = len(residue_to_id)

        # build graph
        residue_indices = [
            RESIDUE_INDICES.get(gemmi.one_letter_code([residue.name]), 0) for residue, _ in residue_to_id.keys()
        ]
        receptor_mask = [int(chain_id in item.receptor_chains) for _, chain_id in residue_to_id.keys()]
        coordinates = [res.get_ca().pos.tolist() for res, _ in residue_to_id.keys()]
        edge_index = [(residue_to_id[src], residue_to_id[dst]) for src, dst in residue_contacts]
        distances = [dist for _, dist in residue_contacts.items()]

        return _InterfaceGraph(
            residues=torch.tensor(residue_indices),
            coordinates=torch.tensor(coordinates, dtype=torch.float32),
            receptor_mask=torch.tensor(receptor_mask),
            edge_index=torch.tensor(edge_index).T,
            distances=torch.tensor(distances, dtype=torch.float32),
        )


class ResidueInterfaceEsmGraphBuilder(InterfaceGraphBuilder):
    def __init__(self, radius: float, esm_dir: Path) -> None:
        self.radius = radius
        self.esm_dir = Path(esm_dir)

    def build_graph(self, item: DataItem) -> InterfaceGraph | None:
        structure = read_structure(item.pdb)
        contacts = find_contacts(structure, item.receptor_chains, item.ligand_chains, self.radius)
        if len(contacts) == 0:
            loguru.logger.warning(f"Skipping {item.uid}: no contacts")
            return None
        # transform atomic contacts to residue contacts
        residue_contacts: dict[tuple[tuple[gemmi.Residue, str], tuple[gemmi.Residue, str]], float] = {}
        for contact in contacts:
            src = contact.partner1.residue, contact.partner1.chain.name
            dst = contact.partner2.residue, contact.partner2.chain.name
            if not contact.partner1.residue.get_ca() or not contact.partner2.residue.get_ca():
                continue
            prev_dist = residue_contacts.get((src, dst), contact.dist)
            residue_contacts[(src, dst)] = min(prev_dist, contact.dist)

        # create mapping from residue to index
        residue_to_id: dict[tuple[gemmi.Residue, str], int] = {}
        for src, _ in residue_contacts:
            if src not in residue_to_id:
                residue_to_id[src] = len(residue_to_id)

        # build graph
        residue_indices = [
            RESIDUE_INDICES.get(gemmi.one_letter_code([residue.name]), 0) for residue, _ in residue_to_id.keys()
        ]
        receptor_mask = torch.tensor([int(chain_id in item.receptor_chains) for _, chain_id in residue_to_id.keys()])
        coordinates = torch.tensor([res.get_ca().pos.tolist() for res, _ in residue_to_id.keys()], dtype=torch.float32)

        # NOTE: only contacts -> complete bipartite graph
        # edge_index = [(residue_to_id[src], residue_to_id[dst]) for src, dst in residue_contacts]
        # distances = [dist for _, dist in residue_contacts.items()]
        # rec = torch.nonzero(receptor_mask == 1).flatten()
        # lig = torch.nonzero(receptor_mask == 0).flatten()
        # edge_index = to_undirected(torch.tensor(list(itertools.product(rec, lig))).T)

        # build KNN graph
        edge_index = to_undirected(knn_graph(coordinates, k=50))  # better than fully-connected
        src, dst = edge_index
        is_intermol = receptor_mask[src] != receptor_mask[dst]
        edge_index = edge_index[:, is_intermol]

        src, dst = edge_index
        distances = (coordinates[src] - coordinates[dst]).norm(dim=1)

        # residue embeds
        try:
            interface_embeds = self.get_interface_embeds(item, structure, residue_to_id)
        except IndexError as err:
            loguru.logger.warning(f"Skipping {item.uid}: {err}")
            return None
        except Exception as err:
            raise err

        return _InterfaceGraph(
            residues=torch.tensor(residue_indices),
            residue_features=interface_embeds,
            coordinates=coordinates,
            receptor_mask=receptor_mask,
            edge_index=edge_index,
            distances=distances,
        )

    def get_interface_embeds(
        self,
        item: DataItem,
        structure: gemmi.Structure,
        residue_to_id: dict[tuple[gemmi.Residue, str], int],
    ) -> Tensor:
        # get residue indices
        chain_seqid_to_index = self.get_seqid_to_index_mapping(structure)
        embeddings_info = self.load_embeds(item.uid)
        chain_embeds = embeddings_info["embeddings"]
        # mapping from chain id to key in embeddings dict
        chain_id_to_sequence_key = {chain_id: key for key in chain_embeds.keys() for chain_id in key.split("|")}
        # TODO: check for synthetic complexes only
        # _a = {_id: len(s) for _id, s in embeddings_info["embeddings"].items()}
        # _b = {_id: len(s) for _id, s in chain_seqid_to_index.items()}
        # if _a != _b:
        #     raise IndexError(f"Lengths mismatch for {item.uid}: embeddings: {_a}, structure: {_b}")
        embeds: list[torch.Tensor] = []
        for residue, chain_id in residue_to_id.keys():
            # get residue index in chain according to structure
            structure_seq_index = chain_seqid_to_index[chain_id][str(residue.seqid)]
            # reindex according to full sequence
            # NOTE: this is necessary because embeddings are calculated for SEQRES sequences
            fullseq_index = embeddings_info["indices"][chain_id][structure_seq_index]
            chains_key = chain_id_to_sequence_key[chain_id]
            if fullseq_index >= (chain_len := len(chain_embeds[chains_key])):
                msg = f"fullseq index = {fullseq_index} exceeds embeddings length = {chain_len}"
                raise IndexError(msg)
            embedding = chain_embeds[chains_key][fullseq_index]
            # TODO: esm preprocessing step is modified to save tensors, not numpy arrays,
            embeds.append(torch.tensor(embedding))
        return torch.stack(embeds)

    def load_embeds(self, uid: str) -> dict[str, Any]:
        # TODO: embeds can belong to several files
        # keys: indices, sequences, embeddings
        # case 1: subunuts came from same structure
        if "-" not in uid:
            return torch.load(self.esm_dir / f"{uid.lower()}.pt", weights_only=False)
        # case 2: subunits come from different structures: 1a22_B-1iar_B
        else:
            # in structure, chains are in alphabetic order: A for 1a22_B and B for 1iar_B
            # NOTE: only dimers are supported!
            embed = {
                "sequences": {},
                "embeddings": {},
                "indices": {},
            }
            for _uid, chain_id in zip(uid.split("-"), "AB"):
                _uid, _chain_id = _uid.split("_")
                chain_embeds = torch.load(self.esm_dir / f"{_uid.lower()}.pt", weights_only=False)
                chain_id_to_sequence_key = {
                    chain_id: key for key in chain_embeds["sequences"].keys() for chain_id in key.split("|")
                }
                # find key for original chain
                key = chain_id_to_sequence_key[_chain_id]
                embed["sequences"][chain_id] = chain_embeds["sequences"][key]
                embed["embeddings"][chain_id] = chain_embeds["embeddings"][key]
                # indices for embeddings: whole sequences were used for cofolding
                embed["indices"][chain_id] = chain_embeds["indices"][_chain_id]
                # embed["indices"][chain_id] = torch.arange(0, len(embed["sequences"][chain_id])).numpy()
            return embed

    @staticmethod
    def get_seqid_to_index_mapping(structure: gemmi.Structure) -> dict[str, dict[str, int]]:
        seqid_to_index = {}
        for chain in structure[0]:
            seqid_to_index[chain.name] = {}
            for i, res in enumerate(chain):
                seqid_to_index[chain.name][str(res.seqid)] = i

        return seqid_to_index
