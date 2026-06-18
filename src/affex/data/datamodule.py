from pathlib import Path

import pandas as pd
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from affex.data.ppb_dataset import PPBAffinityDataset
from affex.data.transform.graph_builder import InterfaceGraphBuilder


class DataModule(LightningDataModule):
    datadir: Path
    annotation_csv: Path
    folds_csv: Path
    test_csv: Path | None
    batch_size: int

    train_dataset: PPBAffinityDataset
    val_dataset: PPBAffinityDataset
    test_dataset: PPBAffinityDataset

    def __init__(
        self,
        datadir: str,
        train_csv: str,
        folds_csv: str,
        val_fold: int,
        test_csv: str | None,
        batch_size: int,
        graph_builder: InterfaceGraphBuilder,
        sample: int = 0,
        foldx_dir: str | None = None,
        synthetic_csv: str | None = None,
        synthetic_datadir: str | None = None,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.datadir = Path(datadir)
        self.annotation_csv = Path(train_csv)
        self.folds_csv = Path(folds_csv)
        self.val_fold = val_fold
        self.test_csv = Path(test_csv) if test_csv is not None else None
        self.batch_size = batch_size
        self.graph_builder = graph_builder
        self.sample = sample
        self.foldx_dir = Path(foldx_dir) if foldx_dir is not None else None
        self.synthetic_csv = Path(synthetic_csv) if synthetic_csv is not None else None
        self.synthetic_datadir = Path(synthetic_datadir) if synthetic_datadir is not None else None
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        num_workers = self.num_workers
        if stage == "fit":
            train_df, val_df = self._load_train_val_splits()
            self.train_dataset = PPBAffinityDataset(
                self.datadir,
                train_df,
                self.graph_builder,
                num_workers=num_workers,
                foldx_dir=self.foldx_dir,
            )

            # Add synthetic data if provided
            if self.synthetic_csv is not None and self.synthetic_datadir is not None:
                synthetic_df = pd.read_csv(self.synthetic_csv)
                synthetic_dataset = PPBAffinityDataset(
                    self.synthetic_datadir,
                    synthetic_df,
                    self.graph_builder,
                    num_workers=num_workers,
                    foldx_dir=None,  # No FoldX features for synthetic data
                )
                self.train_dataset.data += synthetic_dataset.data

            self.val_dataset = PPBAffinityDataset(
                self.datadir,
                val_df,
                self.graph_builder,
                num_workers=num_workers,
                foldx_dir=self.foldx_dir,
            )
        elif stage == "validate":
            _, val_df = self._load_train_val_splits()
            self.val_dataset = PPBAffinityDataset(
                self.datadir,
                val_df,
                self.graph_builder,
                num_workers=num_workers,
                foldx_dir=self.foldx_dir,
            )
        elif stage == "test":
            assert self.test_csv is not None
            test_df = pd.read_csv(self.test_csv)
            self.test_dataset = PPBAffinityDataset(
                self.datadir,
                test_df,
                self.graph_builder,
                num_workers=num_workers,
                foldx_dir=self.foldx_dir,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _make_item_key(df: pd.DataFrame) -> pd.Series:
        """Create composite key: uid_receptorchains_ligandchains."""
        return df["uid"] + "_" + df["receptor_chains"] + "_" + df["ligand_chains"]

    def _load_train_val_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split train/val data based on fold assignments."""
        annotations = pd.read_csv(self.annotation_csv)
        folds = pd.read_csv(self.folds_csv)

        # Determine merge key: item_key (composite) or uid (legacy)
        if "item_key" in folds.columns:
            merge_key = "item_key"
            annotations[merge_key] = self._make_item_key(annotations)
        else:
            merge_key = "uid"

        # Validate key consistency
        ann_keys = set(annotations[merge_key])
        fold_keys = set(folds[merge_key])
        if ann_keys != fold_keys:
            missing_in_folds = ann_keys - fold_keys
            missing_in_ann = fold_keys - ann_keys
            error_msg = f"Key mismatch ({merge_key}) between annotation and folds CSV"
            if missing_in_folds:
                error_msg += f"\n  Missing in folds CSV ({len(missing_in_folds)}): {sorted(missing_in_folds)[:5]}..."
            if missing_in_ann:
                error_msg += f"\n  Missing in annotation CSV ({len(missing_in_ann)}): {sorted(missing_in_ann)[:5]}..."
            raise ValueError(error_msg)

        # Determine fold column name
        fold_col = f"split_{self.val_fold}"
        if fold_col not in folds.columns:
            available = [c for c in folds.columns if c.startswith("split_")]
            raise ValueError(
                f"Column '{fold_col}' not found in folds CSV. "
                f"Available columns: {available}"
            )

        # Merge and split
        merged = annotations.merge(folds[[merge_key, fold_col]], on=merge_key)
        assert isinstance(merged, pd.DataFrame)
        train_df = merged[merged[fold_col] == "train"].copy()
        val_df = merged[merged[fold_col] == "val"].copy()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)

        # Drop merge/fold columns as they're no longer needed
        drop_cols = [fold_col]
        if merge_key == "item_key":
            drop_cols.append("item_key")
        train_df = train_df.drop(columns=drop_cols)
        val_df = val_df.drop(columns=drop_cols)

        # Validate splits are not empty
        if len(train_df) == 0:
            raise ValueError(f"Train split is empty for fold {self.val_fold}")
        if len(val_df) == 0:
            raise ValueError(f"Validation split is empty for fold {self.val_fold}")

        return train_df, val_df

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=PPBAffinityDataset.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=PPBAffinityDataset.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=PPBAffinityDataset.collate_fn,
        )
