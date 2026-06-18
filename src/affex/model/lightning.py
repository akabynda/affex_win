import math
from collections.abc import Callable, Iterator
from typing import Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import (
    MeanAbsoluteError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from affex.data.measurement import BoundedMeasurement, ExactMeasurement, IntervalMeasurement
from affex.data.types import AtomicInterfacePredictor, BatchType

RT = 1.987204258 * 0.001 * 298


def create_regression_metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            f"{prefix}/PearsonCorrCoef": PearsonCorrCoef(),
            f"{prefix}/SpearmanCorrCoef": SpearmanCorrCoef(),
            f"{prefix}/MeanAbsoluteError": MeanAbsoluteError(),
        }
    )


class Alpine(L.LightningModule):
    def __init__(
        self,
        model: AtomicInterfacePredictor,
        optimizer_fn: Callable[[Iterator[nn.Parameter]], Optimizer],
        lr_scheduler_fn: Callable[[Optimizer], LRScheduler],
        synthetic_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.model = model
        self._optimizer = optimizer_fn(self.model.parameters())
        self._scheduler = lr_scheduler_fn(self._optimizer)
        self.optimizer_fn = optimizer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.metrics = {stage: create_regression_metrics(prefix=stage) for stage in ["train", "val", "test"]}
        self.synthetic_weight = synthetic_weight

    def forward(self, batch: BatchType) -> Tensor:
        graphs, _ = batch
        return self.model.forward(graphs)

    def batch_step(self, batch: BatchType, stage: Literal["train", "val", "test"]) -> dict[str, Tensor]:
        graphs, descs = batch
        predictions = self.model.forward(graphs)
        # TODO: fix loss calculation

        # NOTE: model must predict log10(KD)
        # log10_kd = torch.tensor([math.log10(x.affinity.value) for x in descs]).view(-1, 1)
        # loss = F.mse_loss(predictions, log10_kd)
        # # convert to dG from log10(KD)
        # predicted_dg = RT * math.log(10) * predictions
        # target_dg = RT * math.log(10) * log10_kd

        # NOTE: model must predict dG
        # dg = torch.tensor([RT * math.log(x.affinity.value) for x in descs]).view(-1, 1)

        # flags: 0 = exact, 1 = bounded ">" (KD > X, weak binding, dG ≥ target),
        #        2 = bounded "<" (KD < X, strong binding, dG ≤ target)
        targets = []
        flags = []
        for item in descs:
            if isinstance(item.affinity, ExactMeasurement):
                targets.append(RT * math.log(item.affinity.value))
                flags.append(0)
            elif isinstance(item.affinity, IntervalMeasurement):
                targets.append(RT * math.log(item.affinity.estimate))
                flags.append(0)
            elif isinstance(item.affinity, BoundedMeasurement):
                targets.append(RT * math.log(item.affinity.bound))
                flags.append(1 if item.affinity.bound_type == ">" else 2)
            else:
                raise ValueError(f"Unknown measurement type: {type(item.affinity)}")

        targets = torch.tensor(targets, device=predictions.device).view(-1, 1)
        flags = torch.tensor(flags, device=predictions.device)

        assert graphs.batch is not None
        batch_size = int(graphs.batch.max().item()) + 1

        loss: Tensor = torch.tensor(0.0, device=predictions.device)
        exact_mask = flags == 0
        if exact_mask.sum() > 0:
            loss += F.mse_loss(predictions[exact_mask], targets[exact_mask])
            predicted_dg = predictions[exact_mask]
            target_dg = targets[exact_mask]
            self.metrics[stage].update(predicted_dg.detach(), target_dg.detach())

        # Bounded ">" (KD > X): weak binding, dG should be ≥ target (less negative)
        # Penalize when prediction < target (prediction too negative)
        gt_mask = flags == 1
        if gt_mask.sum() > 0:
            gt_errors = torch.relu(targets[gt_mask] - predictions[gt_mask]) ** 2
            loss += self.synthetic_weight * gt_errors.mean()

            self.log(
                f"{stage}/loss_gt",
                gt_errors.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # Bounded "<" (KD < X): strong binding, dG should be ≤ target (more negative)
        # Penalize when prediction > target (prediction not negative enough)
        lt_mask = flags == 2
        if lt_mask.sum() > 0:
            lt_errors = torch.relu(predictions[lt_mask] - targets[lt_mask]) ** 2
            loss += self.synthetic_weight * lt_errors.mean()

        # loss = F.mse_loss(predictions, dg)
        # convert to dG from log10(KD)
        # predicted_dg = predictions
        # target_dg = dg

        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return {"loss": loss, "predictions": predictions}

    def training_step(self, batch: BatchType) -> dict[str, Tensor]:
        return self.batch_step(batch, "train")

    def validation_step(self, batch: BatchType) -> dict[str, Tensor]:
        return self.batch_step(batch, "val")

    def test_step(self, batch: BatchType) -> dict[str, Tensor]:
        return self.batch_step(batch, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.metrics["train"].compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.metrics["train"].reset()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log_dict(
                self.metrics["val"].compute(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        self.metrics["val"].reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics["test"].compute(), prog_bar=True, on_step=False, on_epoch=True)
        self.metrics["test"].reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": self._scheduler,
            "monitor": "train/loss",
        }
