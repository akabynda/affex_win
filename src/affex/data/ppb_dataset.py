from collections.abc import Iterable
from multiprocessing import cpu_count
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from affex.data.measurement import RT, measurement_from_dict
from affex.data.transform.graph_builder import InterfaceGraphBuilder
from affex.data.types import BatchType, DataItem, InterfaceGraph


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    # kd_type,kd_value,kd_lower,kd_upper
    if "KD" in df.columns:
        df["kd_value"] = df["KD"]
    elif "kd_value" not in df.columns and "dG" in df.columns:
        # dG-only annotations (e.g. original pcann-trainval.csv): invert dG = RT·ln(KD)
        df["kd_value"] = np.exp(df["dG"] / RT)

    if "kd_type" not in df.columns:
        df["kd_type"] = "exact"

    if "kd_lower" not in df.columns:
        df["kd_lower"] = None

    if "kd_upper" not in df.columns:
        df["kd_upper"] = None

    return df


def parse_foldx_interaction_energy(fxout_path: Path) -> float:
    """Parse the Interaction Energy from a FoldX Interaction_*.fxout file."""
    lines = fxout_path.read_text().splitlines()
    if len(lines) < 10:
        raise ValueError(f"FoldX file {fxout_path} has fewer than 10 lines")
    headers = lines[8].split("\t")
    values = lines[9].split("\t")
    energy_idx = headers.index("Interaction Energy")
    return float(values[energy_idx])


def load_foldx_energies(foldx_dir: Path) -> dict[str, float]:
    """Load all FoldX interaction energies from a directory into a uid->energy dict."""
    energies: dict[str, float] = {}
    for fxout in foldx_dir.rglob("Interaction_*.fxout"):
        # Filename: Interaction_{uid}_{chains}.fxout
        stem = fxout.stem  # e.g. "Interaction_1a22_AC"
        parts = stem.split("_", 2)  # ["Interaction", "1a22", "AC"]
        uid = parts[1]
        energies[uid] = parse_foldx_interaction_energy(fxout)
    return energies


class PPBAffinityDataset(Dataset):
    def __init__(
        self,
        datadir: Path,
        subset_df: pd.DataFrame,
        graph_builder: InterfaceGraphBuilder,
        num_workers: int = cpu_count() - 1,
        foldx_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.graph_builder = graph_builder
        self.foldx_energies = load_foldx_energies(foldx_dir) if foldx_dir else {}
        self.data = self._preprocess(subset_df, datadir, num_workers)
        logger.info(f"Найдено структур: {len(self)}.")

    def _rows_to_items(self, subset_df: pd.DataFrame, datadir: Path) -> list[DataItem]:
        items = []
        for _, row in subset_df.iterrows():
            uid = row["uid"]
            pdb_path = datadir / f"{uid}.pdb"
            if not pdb_path.is_file():
                continue
            measurement = measurement_from_dict(row.to_dict())
            item = DataItem(str(uid), pdb_path, list(row["receptor_chains"]), list(row["ligand_chains"]), measurement)
            # item = DataItem(uid, pdb_path, row["ligand_chains"], row["receptor_chains"], measurement)
            items.append(item)
        return items

    def _preprocess_one(self, item: DataItem) -> tuple[InterfaceGraph, DataItem] | None:
        graph = self.graph_builder.build_graph(item)
        if graph is None:
            return None

        if self.foldx_energies:
            if item.uid not in self.foldx_energies:
                logger.warning(f"No FoldX energy for {item.uid}, using 0.0")
            graph.foldx_energy = torch.tensor(self.foldx_energies.get(item.uid, 0.0))

        graph = Data(
            **vars(graph),
            num_nodes=len(graph.coordinates),
        )
        return (cast(InterfaceGraph, graph), item)

    def _preprocess(
        self,
        subset_df: pd.DataFrame,
        datadir: Path,
        num_workers: int,
    ) -> list[tuple[InterfaceGraph, DataItem]]:
        # NOTE: subset_df: ["uid", "receptor_chains", "ligand_chains", "KD"]
        subset_df = add_missing_columns(subset_df.copy())
        # subset_df = subset_df.sort_values("kd_value").drop_duplicates("uid")
        items = self._rows_to_items(subset_df, datadir)
        logger.info(f"Preprocessing with {num_workers} cores")
        if num_workers == 0:
            data = tqdm([self._preprocess_one(x) for x in items], total=len(items))
        else:
            # Ensure spawn method is set for clean PyTorch process isolation
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
            with mp.Pool(num_workers) as pool:
                data = list(tqdm(pool.map(self._preprocess_one, items), total=len(items)))
                pool.close()
                pool.join()
        data = [x for x in data if x is not None]
        return data

    def __getitem__(self, index: int) -> tuple[InterfaceGraph, DataItem]:
        return self.data[index]

    @staticmethod
    def collate_fn(batch: list[tuple[InterfaceGraph, DataItem]]) -> BatchType:
        graph, descs = zip(*batch)
        descs = cast(Iterable[DataItem], descs)
        return (
            cast(InterfaceGraph, Batch.from_data_list(list(graph))),
            list(descs),
        )

    def __len__(self) -> int:
        return len(self.data)
