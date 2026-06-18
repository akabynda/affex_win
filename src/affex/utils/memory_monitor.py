"""Lightweight memory monitoring callback for Lightning Trainer.

Logs RSS, VMS, swap usage, disk free space, and open file descriptors at key
points (fit start/end, epoch boundaries, validation). Writes a CSV log per run
and logs warnings to a separate file when growth exceeds thresholds.

Enable via config:
    callbacks:
      memory_monitor:
        _target_: affex.utils.memory_monitor.MemoryMonitor
        warn_growth_mb: 500
"""

import csv
import logging
import os
import shutil
import time
from pathlib import Path

import lightning as L
import psutil
from lightning.pytorch import Trainer
from lightning.pytorch.utilities import rank_zero_only

_CSV_FIELDS = [
    "timestamp",
    "event",
    "epoch",
    "global_step",
    "rss_mb",
    "vms_mb",
    "swap_system_used_mb",
    "swap_system_total_mb",
    "disk_free_gb",
    "num_fds",
    "num_threads",
    "children",
]


def _get_mem_info() -> dict[str, float]:
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    swap = psutil.swap_memory()
    disk_free_gb = shutil.disk_usage("/").free / 1024 / 1024 / 1024
    return {
        "rss_mb": mem.rss / 1024 / 1024,
        "vms_mb": mem.vms / 1024 / 1024,
        "swap_system_used_mb": swap.used / 1024 / 1024,
        "swap_system_total_mb": swap.total / 1024 / 1024,
        "disk_free_gb": disk_free_gb,
        "num_fds": proc.num_fds() if hasattr(proc, "num_fds") else -1,
        "num_threads": proc.num_threads(),
        "children": len(proc.children(recursive=True)),
    }


class MemoryMonitor(L.Callback):
    def __init__(self, warn_growth_mb: float = 500, output_dir: str | None = None) -> None:
        self.warn_growth_mb = warn_growth_mb
        self._output_dir = Path(output_dir) if output_dir else None
        self._fit_start_rss: float = 0.0
        self._fit_start_swap: float = 0.0
        self._epoch_start_rss: float = 0.0
        self._epoch_start_swap: float = 0.0
        self._csv_path: Path | None = None
        self._log_path: Path | None = None
        self._logger: logging.Logger | None = None

    def _setup_file_logger(self, output_dir: Path) -> None:
        self._log_path = output_dir / "memory_monitor.log"
        self._logger = logging.getLogger(f"memory_monitor.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        handler = logging.FileHandler(self._log_path, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def _log_msg(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)

    def _record(self, event: str, trainer: Trainer) -> None:
        info = _get_mem_info()
        entry = {
            "timestamp": time.time(),
            "event": event,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            **info,
        }

        if self._csv_path is not None:
            file_exists = self._csv_path.exists() and self._csv_path.stat().st_size > 0
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        output_dir = self._output_dir or Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = output_dir / "memory_log.csv"
        self._setup_file_logger(output_dir)

        info = _get_mem_info()
        self._fit_start_rss = info["rss_mb"]
        self._fit_start_swap = info["swap_system_used_mb"]
        self._record("fit_start", trainer)
        self._log_msg(
            f"fit_start: RSS={info['rss_mb']:.0f}MB, VMS={info['vms_mb']:.0f}MB, "
            f"swap_used={info['swap_system_used_mb']:.0f}MB, "
            f"disk_free={info['disk_free_gb']:.1f}GB"
        )

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        info = _get_mem_info()
        self._epoch_start_rss = info["rss_mb"]
        self._epoch_start_swap = info["swap_system_used_mb"]
        self._record("epoch_start", trainer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        self._record("epoch_end", trainer)
        info = _get_mem_info()
        epoch_rss_growth = info["rss_mb"] - self._epoch_start_rss
        total_rss_growth = info["rss_mb"] - self._fit_start_rss
        swap_growth = info["swap_system_used_mb"] - self._fit_start_swap
        epoch_swap_growth = info["swap_system_used_mb"] - self._epoch_start_swap

        self._log_msg(
            f"epoch {trainer.current_epoch} end: "
            f"RSS={info['rss_mb']:.0f}MB ({epoch_rss_growth:+.0f}MB epoch, {total_rss_growth:+.0f}MB total), "
            f"swap_used={info['swap_system_used_mb']:.0f}MB ({epoch_swap_growth:+.0f}MB epoch, {swap_growth:+.0f}MB total), "
            f"disk_free={info['disk_free_gb']:.1f}GB, FDs={info['num_fds']}"
        )

        if total_rss_growth > self.warn_growth_mb:
            self._log_msg(
                f"WARNING: RSS grew {total_rss_growth:+.0f}MB since fit_start"
            )
        if swap_growth > self.warn_growth_mb:
            self._log_msg(
                f"WARNING: System swap grew {swap_growth:+.0f}MB since fit_start"
            )

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        self._record("val_epoch_end", trainer)

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: L.LightningModule) -> None:
        self._record("fit_end", trainer)
        info = _get_mem_info()
        total_rss_growth = info["rss_mb"] - self._fit_start_rss
        swap_growth = info["swap_system_used_mb"] - self._fit_start_swap
        self._log_msg(
            f"fit_end: RSS {self._fit_start_rss:.0f}MB -> {info['rss_mb']:.0f}MB ({total_rss_growth:+.0f}MB), "
            f"swap {self._fit_start_swap:.0f}MB -> {info['swap_system_used_mb']:.0f}MB ({swap_growth:+.0f}MB), "
            f"disk_free={info['disk_free_gb']:.1f}GB, FDs={info['num_fds']}"
        )
        if self._csv_path:
            self._log_msg(f"CSV log: {self._csv_path}")
        if self._logger:
            for handler in self._logger.handlers:
                handler.close()
