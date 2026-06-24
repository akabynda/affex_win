import gc
import logging
import os
import shutil

import hydra
import lightning as L
import psutil
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

from affex.data.datamodule import DataModule
from affex.model.lightning import Alpine

log = logging.getLogger(__name__)


def _log_process_memory(label: str) -> None:
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    swap = psutil.swap_memory()
    disk_free = shutil.disk_usage("/").free / 1024**3
    log.info(
        "[mem] %s: RSS=%.0fMB, VMS=%.0fMB, swap_used=%.0fMB, disk_free=%.1fGB, gc_objects=%d",
        label,
        mem.rss / 1024**2,
        mem.vms / 1024**2,
        swap.used / 1024**2,
        disk_free,
        len(gc.get_objects()),
    )


def _safe_experiment_call(logger, method_name: str, *args, **kwargs) -> None:
    experiment = getattr(logger, "experiment", None)
    method = getattr(experiment, method_name, None)
    if callable(method):
        method(*args, **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    callbacks = [hydra.utils.instantiate(cb_conf) for _, cb_conf in cfg.callbacks.items()] if cfg.callbacks else []
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    hparams = {
        "model": OmegaConf.to_container(cfg.lightning, resolve=True),
        "data": OmegaConf.to_container(cfg.datamodule, resolve=True),
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
        "seed": cfg.seed,
    }

    quiet_overrides = {}
    if cfg.quiet:
        quiet_overrides = {
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }

    if logger:
        if cfg.group:
            _safe_experiment_call(logger, "set", "group", cfg.group, strict=False)
            _safe_experiment_call(logger, "add_tag", cfg.group)
        if cfg.hypothesis:
            _safe_experiment_call(logger, "set", "hypothesis", cfg.hypothesis, strict=False)
            _safe_experiment_call(logger, "add_tag", cfg.hypothesis)
        if cfg.label:
            _safe_experiment_call(logger, "add_tag", cfg.label)
        if cfg.direction:
            _safe_experiment_call(logger, "set", "direction", cfg.direction, strict=False)
        _safe_experiment_call(logger, "set", "seed", cfg.seed, strict=False)
        _safe_experiment_call(logger, "set", "val_fold", cfg.datamodule.val_fold, strict=False)

        logger.log_hyperparams(hparams)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, **quiet_overrides)

    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)

    lit: Alpine = hydra.utils.instantiate(cfg.lightning)
    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)

    trainer.fit(
        model=lit,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint,
    )
    val_results = trainer.validate(model=lit, datamodule=datamodule, ckpt_path="best")
    test_results = trainer.test(model=lit, datamodule=datamodule, ckpt_path="best")

    if cfg.quiet:
        log.info("=== Final Metrics ===")
        for results, label in [(val_results, "Validation"), (test_results, "Test")]:
            if results:
                log.info("%s:", label)
                for key, value in results[0].items():
                    log.info("  %s: %.4f", key, value)

    # Explicit cleanup to prevent memory accumulation across Hydra multirun sweeps.
    # Without this, prior-run tensors and dataset objects linger, and spawn-based
    # mp.Pool workers copy the bloated parent process memory on each new run.
    _log_process_memory("before_cleanup")
    del trainer, lit, datamodule, callbacks
    if hasattr(logger, "close"):
        logger.close()
    del logger
    gc.collect()
    import torch
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    _log_process_memory("after_cleanup")


if __name__ == "__main__":
    train()
