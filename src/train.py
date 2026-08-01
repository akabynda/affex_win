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


def validate_training_protocol(cfg: DictConfig) -> None:
    """Reject accidental changes to the canonical comparable-run settings."""
    protocol = cfg.get("training_protocol")
    if not protocol or not bool(protocol.get("enforce", False)):
        return

    optimizer = cfg.lightning.optimizer_fn
    scheduler = cfg.lightning.lr_scheduler_fn
    early_stopping = cfg.callbacks.early_stopping
    actual = {
        "seed": cfg.seed,
        "datadir": cfg.datamodule.datadir,
        "train_csv": cfg.datamodule.train_csv,
        "folds_csv": cfg.datamodule.folds_csv,
        "test_csv": cfg.datamodule.test_csv,
        "batch_size": cfg.datamodule.batch_size,
        "num_workers": cfg.datamodule.num_workers,
        "graph_radius": cfg.datamodule.graph_builder.radius,
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "max_epochs": cfg.trainer.max_epochs,
        "check_val_every_n_epoch": cfg.trainer.check_val_every_n_epoch,
        "deterministic": cfg.trainer.deterministic,
        "accumulate_grad_batches": cfg.trainer.get("accumulate_grad_batches", 1),
        "optimizer_target": optimizer._target_,
        "learning_rate": optimizer.lr,
        "weight_decay": optimizer.weight_decay,
        "amsgrad": optimizer.get("amsgrad", False),
        "scheduler_target": scheduler._target_,
        "scheduler_patience": scheduler.patience,
        "scheduler_factor": scheduler.factor,
        "scheduler_min_lr": scheduler.min_lr,
        "early_stopping_monitor": early_stopping.monitor,
        "early_stopping_patience": early_stopping.patience,
        "logger_target": cfg.logger._target_ if cfg.logger else None,
    }
    expected = OmegaConf.to_container(protocol.expected, resolve=True)
    mismatches = {
        key: (expected[key], actual.get(key))
        for key in expected
        if actual.get(key) != expected[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {wanted!r}, got {found!r}"
            for key, (wanted, found) in mismatches.items()
        )
        raise ValueError(
            f"Training protocol {protocol.name!r} was overridden ({details}). "
            "Change configs/training_protocol/pcann_shared_gpu.yaml for all comparable "
            "experiments, or explicitly mark an architecturally incompatible run "
            "with training_protocol.enforce=false."
        )


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
    validate_training_protocol(cfg)
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
    skip_postfit_evaluation = bool(cfg.get("skip_postfit_evaluation", False))
    if skip_postfit_evaluation:
        val_results = []
        test_results = []
        log.info("Skipping post-fit validate/test; best checkpoint was still selected from epoch validation")
    else:
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
