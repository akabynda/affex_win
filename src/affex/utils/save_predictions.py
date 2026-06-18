import math
from pathlib import Path

import lightning as L
import pandas as pd

from affex.data.measurement import ExactMeasurement, IntervalMeasurement, BoundedMeasurement

RT = 1.9872036e-3 * 298.15  # kcal/mol


def _target_dg(item) -> float:
    """Compute target dG from a DataItem, matching lightning.py logic."""
    if isinstance(item.affinity, ExactMeasurement):
        return RT * math.log(item.affinity.value)
    elif isinstance(item.affinity, IntervalMeasurement):
        return RT * math.log(item.affinity.estimate)
    elif isinstance(item.affinity, BoundedMeasurement):
        return RT * math.log(item.affinity.bound)
    else:
        raise ValueError(f"Unknown measurement type: {type(item.affinity)}")


class SavePredictions(L.Callback):
    def __init__(
        self,
        val_preds_path: Path | None = None,
        test_preds_path: Path | None = None,
        train_preds_path: Path | None = None,
    ):
        super().__init__()
        self.val_preds_path = val_preds_path
        self.test_preds_path = test_preds_path
        self.train_preds_path = train_preds_path
        self.val_scores: list[pd.DataFrame] = []
        self.test_scores: list[pd.DataFrame] = []
        self.train_scores: list[pd.DataFrame] = []

    def _collect_batch(self, outputs, batch) -> pd.DataFrame:
        assert isinstance(outputs, dict)
        _, descs = batch
        rows = []
        for item, pred in zip(descs, outputs["predictions"].cpu().flatten().tolist()):
            target = _target_dg(item)
            rows.append({
                "uid": item.uid,
                "target": target,
                "prediction": pred,
                "error": pred - target,
            })
        return pd.DataFrame(rows)

    # --- Training ---
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.train_preds_path:
            self.train_scores.append(self._collect_batch(outputs, batch))

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_preds_path and self.train_scores:
            pd.concat(self.train_scores).assign(epoch=trainer.current_epoch).to_csv(
                self.train_preds_path, mode="a", index=False,
                header=not Path(self.train_preds_path).exists() or Path(self.train_preds_path).stat().st_size == 0,
            )
            self.train_scores = []

    # --- Validation ---
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.val_preds_path:
            self.val_scores.append(self._collect_batch(outputs, batch))

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_preds_path and self.val_scores:
            pd.concat(self.val_scores).assign(epoch=trainer.current_epoch).to_csv(
                self.val_preds_path, mode="a", index=False,
                header=not Path(self.val_preds_path).exists() or Path(self.val_preds_path).stat().st_size == 0,
            )
            self.val_scores = []

    # --- Test ---
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.test_preds_path:
            self.test_scores.append(self._collect_batch(outputs, batch))

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_preds_path and self.test_scores:
            pd.concat(self.test_scores).assign(epoch=trainer.current_epoch).to_csv(
                self.test_preds_path, mode="a", index=False,
                header=not Path(self.test_preds_path).exists() or Path(self.test_preds_path).stat().st_size == 0,
            )
            self.test_scores = []
