# python modules
import argparse
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
import os 
import re
from typing import Optional

# Optional temp directory for W&B logs when logger_dir is unset.
_WANDB_TMPDIR: Optional[TemporaryDirectory] = None

# deep learning modules
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor

# my modules
from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.utils import replace_args_from_config, restore_optimizer
from ml_recon.pl_modules.pl_learn_ssl_undersampling import (
    LearnedSSLLightning, 
    VarnetConfig, 
    LearnPartitionConfig, 
    DualDomainConifg
    )

# pytorch lightning tools and trainers
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.callbacks import ModelCheckpoint 


class BestModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that prints when a new best checkpoint is saved."""

    def __init__(self, *args, **kwargs):
        self.test_on_best: bool = bool(kwargs.pop('test_on_best', True))
        self.test_on_best_update_wandb_summary: bool = bool(kwargs.pop('test_on_best_update_wandb_summary', True))
        # If true, also log best-test metrics into W&B *history* so they appear in charts.
        # We log at a fixed step to behave like an "updating single point" in plots.
        self.test_on_best_log_wandb_history: bool = bool(kwargs.pop('test_on_best_log_wandb_history', True))
        self.test_on_best_history_step: int = int(kwargs.pop('test_on_best_history_step', 0))
        super().__init__(*args, **kwargs)
        self._last_best_model_path: str = ""

    @staticmethod
    def _safe_float(x):
        try:
            if x is None:
                return None
            if hasattr(x, 'detach'):
                x = x.detach().cpu()
            return float(x)
        except Exception:
            return None

    def _update_wandb_summary_from_model(self, model, run) -> None:
        """Write mean test metrics into the given wandb run summary under `best_test/*` keys."""
        if not getattr(self, 'test_on_best_update_wandb_summary', True):
            return

        if run is None:
            return

        contrast_labels = list(getattr(model, 'metric_contrast_order', getattr(model, 'contrast_order', [])))

        def _mean_of_list(values):
            if not values:
                return None
            vals = [self._safe_float(v) for v in values]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        # Overall (no prefix) metrics accumulated in plReconModel.
        ssim_means = []
        ssim_masked_means = []
        nmse_means = []
        nmse_masked_means = []
        psnr_means = []
        psnr_masked_means = []

        for i, label in enumerate(contrast_labels):
            ssim_m = _mean_of_list(getattr(model, 'contrast_ssim', [])[i] if hasattr(model, 'contrast_ssim') else [])
            ssim_mm = _mean_of_list(getattr(model, 'contrast_ssim_masked', [])[i] if hasattr(model, 'contrast_ssim_masked') else [])
            nmse_m = _mean_of_list(getattr(model, 'contrast_nmse', [])[i] if hasattr(model, 'contrast_nmse') else [])
            nmse_mm = _mean_of_list(getattr(model, 'contrast_nmse_masked', [])[i] if hasattr(model, 'contrast_nmse_masked') else [])
            psnr_m = _mean_of_list(getattr(model, 'contrast_psnr', [])[i] if hasattr(model, 'contrast_psnr') else [])
            psnr_mm = _mean_of_list(getattr(model, 'contrast_psnr_masked', [])[i] if hasattr(model, 'contrast_psnr_masked') else [])

            if ssim_m is not None:
                run.summary[f"best_test/ssim_{label}"] = ssim_m
                ssim_means.append(ssim_m)
            if ssim_mm is not None:
                run.summary[f"best_test/masked_ssim_{label}"] = ssim_mm
                ssim_masked_means.append(ssim_mm)
            if nmse_m is not None:
                run.summary[f"best_test/nmse_{label}"] = nmse_m
                nmse_means.append(nmse_m)
            if nmse_mm is not None:
                run.summary[f"best_test/masked_nmse_{label}"] = nmse_mm
                nmse_masked_means.append(nmse_mm)
            if psnr_m is not None:
                run.summary[f"best_test/psnr_{label}"] = psnr_m
                psnr_means.append(psnr_m)
            if psnr_mm is not None:
                run.summary[f"best_test/masked_psnr_{label}"] = psnr_mm
                psnr_masked_means.append(psnr_mm)

        if ssim_means:
            run.summary["best_test/mean_ssim"] = float(sum(ssim_means) / len(ssim_means))
        if ssim_masked_means:
            run.summary["best_test/mean_masked_ssim"] = float(sum(ssim_masked_means) / len(ssim_masked_means))
        if nmse_means:
            run.summary["best_test/mean_nmse"] = float(sum(nmse_means) / len(nmse_means))
        if nmse_masked_means:
            run.summary["best_test/mean_masked_nmse"] = float(sum(nmse_masked_means) / len(nmse_masked_means))
        if psnr_means:
            run.summary["best_test/mean_psnr"] = float(sum(psnr_means) / len(psnr_means))
        if psnr_masked_means:
            run.summary["best_test/mean_masked_psnr"] = float(sum(psnr_masked_means) / len(psnr_masked_means))

        # Also capture per-prefix test means if present (e.g., multiple test loaders).
        bucket_by_prefix = getattr(model, 'test_metrics_by_prefix', None)
        if isinstance(bucket_by_prefix, dict) and bucket_by_prefix:
            try:
                # Helpful for multi-R runs: make it obvious which prefixes (e.g. R4/, R6/) were present.
                run.summary["best_test/prefixes"] = sorted(str(p) for p in bucket_by_prefix.keys())
            except Exception:
                pass

            for prefix, bucket in bucket_by_prefix.items():
                if not isinstance(bucket, dict):
                    continue

                # Make per-prefix summaries easy to see in W&B (avoid nested keys confusion).
                # Example prefix values: "R4/", "R6/", "target/R4/".
                prefix_label = str(prefix).strip("/")
                if not prefix_label:
                    prefix_label = "unknown"
                prefix_label = prefix_label.replace("/", "_")

                per_prefix_means = {
                    "ssim": [],
                    "masked_ssim": [],
                    "nmse": [],
                    "masked_nmse": [],
                    "psnr": [],
                    "masked_psnr": [],
                }

                # Record how many samples contributed per-prefix (useful sanity check for multi-R runs).
                try:
                    nmse_lists = bucket.get("nmse", [])
                    counts = [len(v) for v in nmse_lists] if isinstance(nmse_lists, list) else []
                    run.summary[f"best_test/count_{prefix_label}"] = int(sum(counts)) if counts else 0
                except Exception:
                    pass

                for i, label in enumerate(contrast_labels):
                    for metric_key, out_key in (
                        ('ssim', 'ssim'),
                        ('nmse', 'nmse'),
                        ('psnr', 'psnr'),
                        ('ssim_masked', 'masked_ssim'),
                        ('nmse_masked', 'masked_nmse'),
                        ('psnr_masked', 'masked_psnr'),
                    ):
                        values = bucket.get(metric_key, [])
                        values_i = values[i] if isinstance(values, list) and i < len(values) else []
                        m = _mean_of_list(values_i)
                        if m is not None:
                            run.summary[f"best_test/{prefix}{out_key}_{label}"] = m
                            # Also keep a flat, explicitly labeled copy.
                            run.summary[f"best_test/{out_key}_{label}_{prefix_label}"] = m
                            per_prefix_means[out_key].append(m)

                # Add per-prefix aggregate means across contrasts (useful when you have multiple Rs).
                try:
                    for out_key, vals in per_prefix_means.items():
                        if vals:
                            run.summary[f"best_test/mean_{out_key}_{prefix_label}"] = float(sum(vals) / len(vals))
                except Exception:
                    pass

    def _log_wandb_history_from_model(self, model, run) -> None:
        """Log best-test metrics into W&B history so they show up in charts.

        Note: W&B history is append-only; to emulate "overwrite", we log at a fixed
        `step` (default 0). W&B will then plot a single point that updates.
        """
        if not getattr(self, 'test_on_best_log_wandb_history', True):
            return
        if run is None:
            return

        contrast_labels = list(getattr(model, 'metric_contrast_order', getattr(model, 'contrast_order', [])))

        def _mean_of_list(values):
            if not values:
                return None
            vals = [self._safe_float(v) for v in values]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return float(sum(vals) / len(vals))

        data = {}

        # Per-contrast means (overall buckets)
        ssim_means = []
        ssim_masked_means = []
        nmse_means = []
        nmse_masked_means = []
        psnr_means = []
        psnr_masked_means = []

        for i, label in enumerate(contrast_labels):
            ssim_m = _mean_of_list(getattr(model, 'contrast_ssim', [])[i] if hasattr(model, 'contrast_ssim') else [])
            ssim_mm = _mean_of_list(getattr(model, 'contrast_ssim_masked', [])[i] if hasattr(model, 'contrast_ssim_masked') else [])
            nmse_m = _mean_of_list(getattr(model, 'contrast_nmse', [])[i] if hasattr(model, 'contrast_nmse') else [])
            nmse_mm = _mean_of_list(getattr(model, 'contrast_nmse_masked', [])[i] if hasattr(model, 'contrast_nmse_masked') else [])
            psnr_m = _mean_of_list(getattr(model, 'contrast_psnr', [])[i] if hasattr(model, 'contrast_psnr') else [])
            psnr_mm = _mean_of_list(getattr(model, 'contrast_psnr_masked', [])[i] if hasattr(model, 'contrast_psnr_masked') else [])

            if ssim_m is not None:
                data[f"best_test/ssim_{label}"] = ssim_m
                ssim_means.append(ssim_m)
            if ssim_mm is not None:
                data[f"best_test/masked_ssim_{label}"] = ssim_mm
                ssim_masked_means.append(ssim_mm)
            if nmse_m is not None:
                data[f"best_test/nmse_{label}"] = nmse_m
                nmse_means.append(nmse_m)
            if nmse_mm is not None:
                data[f"best_test/masked_nmse_{label}"] = nmse_mm
                nmse_masked_means.append(nmse_mm)
            if psnr_m is not None:
                data[f"best_test/psnr_{label}"] = psnr_m
                psnr_means.append(psnr_m)
            if psnr_mm is not None:
                data[f"best_test/masked_psnr_{label}"] = psnr_mm
                psnr_masked_means.append(psnr_mm)

        if ssim_means:
            data["best_test/mean_ssim"] = float(sum(ssim_means) / len(ssim_means))
        if ssim_masked_means:
            data["best_test/mean_masked_ssim"] = float(sum(ssim_masked_means) / len(ssim_masked_means))
        if nmse_means:
            data["best_test/mean_nmse"] = float(sum(nmse_means) / len(nmse_means))
        if nmse_masked_means:
            data["best_test/mean_masked_nmse"] = float(sum(nmse_masked_means) / len(nmse_masked_means))
        if psnr_means:
            data["best_test/mean_psnr"] = float(sum(psnr_means) / len(psnr_means))
        if psnr_masked_means:
            data["best_test/mean_masked_psnr"] = float(sum(psnr_masked_means) / len(psnr_masked_means))

        # Per-prefix buckets (if multiple test loaders)
        bucket_by_prefix = getattr(model, 'test_metrics_by_prefix', None)
        if isinstance(bucket_by_prefix, dict) and bucket_by_prefix:
            for prefix, bucket in bucket_by_prefix.items():
                if not isinstance(bucket, dict):
                    continue

                prefix_label = str(prefix).strip("/")
                if not prefix_label:
                    prefix_label = "unknown"
                prefix_label = prefix_label.replace("/", "_")

                for i, label in enumerate(contrast_labels):
                    for metric_key, out_key in (
                        ('ssim', 'ssim'),
                        ('nmse', 'nmse'),
                        ('psnr', 'psnr'),
                        ('ssim_masked', 'masked_ssim'),
                        ('nmse_masked', 'masked_nmse'),
                        ('psnr_masked', 'masked_psnr'),
                    ):
                        values = bucket.get(metric_key, [])
                        values_i = values[i] if isinstance(values, list) and i < len(values) else []
                        m = _mean_of_list(values_i)
                        if m is not None:
                            data[f"best_test/{prefix}{out_key}_{label}"] = m
                            data[f"best_test/{out_key}_{label}_{prefix_label}"] = m

        if data:
            try:
                # W&B requires monotonically increasing steps; fixed steps (e.g. 0) can be rejected
                # later in training. If a desired step is provided but would go backwards, clamp to
                # the current run step. If no step is desired, let W&B manage it.
                desired_step = getattr(self, 'test_on_best_history_step', None)
                step = None
                if desired_step is not None:
                    try:
                        desired_step_i = int(desired_step)
                        if desired_step_i >= 0:
                            current_step = getattr(run, 'step', None)
                            if current_step is not None:
                                step = max(desired_step_i, int(current_step))
                            else:
                                step = desired_step_i
                    except Exception:
                        step = None

                if step is None:
                    run.log(data, commit=True)
                else:
                    run.log(data, step=step, commit=True)
            except Exception:
                # Don't crash training due to W&B logging.
                pass

    def _test_best_checkpoint(self, trainer, pl_module) -> None:
        if not self.test_on_best:
            return
        if not getattr(trainer, 'is_global_zero', True):
            return
        best_path = getattr(self, 'best_model_path', '')
        if not best_path:
            return
        if not os.path.isfile(best_path):
            return

        datamodule = getattr(trainer, 'datamodule', None)
        if datamodule is None:
            return

        try:
            # Load a fresh model instance so we don't disturb training weights.
            best_model = pl_module.__class__.load_from_checkpoint(best_path, map_location='cpu')

            # IMPORTANT: Do NOT log per-epoch test results to W&B history.
            # We only want a single set of "latest best" test metrics, so we run
            # this test with a DummyLogger and then write means into wandb *summary*.
            test_trainer = pl.Trainer(
                logger=DummyLogger(),
                enable_checkpointing=False,
                accelerator='auto',
                devices=1,
                num_nodes=1,
                enable_progress_bar=False,
                log_every_n_steps=trainer.log_every_n_steps if hasattr(trainer, 'log_every_n_steps') else 50,
            )
            test_trainer.test(model=best_model, datamodule=datamodule, verbose=False)

            # Overwrite/update summary fields so the latest-best test metrics persist.
            run = None
            if isinstance(trainer.logger, WandbLogger):
                run = getattr(trainer.logger, 'experiment', None)
            if run is not None:
                self._update_wandb_summary_from_model(best_model, run)
                self._log_wandb_history_from_model(best_model, run)
                try:
                    run.summary["best_test/best_ckpt_path"] = str(best_path)
                except Exception:
                    pass
            print(f"[BestModelCheckpoint] Tested new best checkpoint: {best_path}")
        except Exception as e:
            # Never crash training due to best-checkpoint testing.
            print(f"[BestModelCheckpoint] Warning: failed to test best checkpoint '{best_path}': {e}")

    def on_validation_end(self, trainer, pl_module):
        # Don't treat Lightning's pre-training sanity check as a real "best".
        if getattr(trainer, 'sanity_checking', False):
            super().on_validation_end(trainer, pl_module)
            return

        # Make sure the monitored metric exists before ModelCheckpoint evaluates/saves.
        # In multi-dataloader validation (e.g., R_values), the un-prefixed aggregate
        # `val/maskedmean_ssim_full` can be logged slightly later than per-loader metrics.
        try:
            monitor_key = self.monitor or ""
            if monitor_key and trainer.callback_metrics.get(monitor_key) is None:
                agg_fn = getattr(pl_module, '_log_aggregate_val_metrics', None)
                if callable(agg_fn):
                    agg_fn()
        except Exception:
            pass

        super().on_validation_end(trainer, pl_module)

        # Print only once in distributed setups.
        if not getattr(trainer, "is_global_zero", True):
            return

        if self.best_model_path and self.best_model_path != self._last_best_model_path:
            monitor_key = self.monitor or ""
            current = trainer.callback_metrics.get(monitor_key)
            try:
                current_val = float(current.detach().cpu()) if current is not None else None
            except Exception:
                current_val = None

            # Note: we monitor masked mean SSIM with mode='max', so "best" means it increased.
            if current_val is None:
                print("validation mean masked ssim improved - best model saved")
            else:
                print(f"validation mean masked ssim improved - best model saved (val={current_val:.6f})")

            self._last_best_model_path = self.best_model_path
            # Immediately test the new best checkpoint so results are recorded even if training crashes later.
            self._test_best_checkpoint(trainer, pl_module)

def main(args):
    pl.seed_everything(8, workers=True)
    file_name = get_unique_file_name(args)

    # build some callbacks for pytorch lightning
    callbacks = build_callbacks(args, file_name)
    
    # setup pytorch lightning dataloder and datamodules
    model, data_module = setup_model_and_dataloaders(args, callbacks)
    # setup wandb logger
    wandb_logger = setup_wandb_logger(args, model)

    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'logger': wandb_logger,
        'callbacks': callbacks,
        'fast_dev_run': args.fast_dev_run,
    }
    if getattr(args, 'disable_checkpoints', False):
        trainer_kwargs['enable_checkpointing'] = False
        if args.logger_dir is None or str(args.logger_dir).lower() in {"none", "null", ""}:
            global _WANDB_TMPDIR
            if _WANDB_TMPDIR is None:
                _WANDB_TMPDIR = TemporaryDirectory()
            trainer_kwargs['default_root_dir'] = _WANDB_TMPDIR.name
        else:
            trainer_kwargs['default_root_dir'] = args.logger_dir
    # Optional gradient clipping for training stability (especially helpful for prompt backbones).
    if getattr(args, 'grad_clip_val', 0.0) and args.grad_clip_val > 0:
        trainer_kwargs['gradient_clip_val'] = args.grad_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = args.grad_clip_algo

    # Quick sanity mode: run a tiny train/val to trigger best-ckpt intermediate test.
    # Useful to verify multi-loader test logging (e.g., R_values=[4,6,8]) without waiting.
    if getattr(args, 'sanity_check_run', False):
        trainer_kwargs['max_epochs'] = 5
        trainer_kwargs['limit_train_batches'] = 2
        trainer_kwargs['limit_val_batches'] = 2
        trainer_kwargs['limit_test_batches'] = 2
        # Disable Lightning's pre-fit sanity validation to keep this mode minimal.
        trainer_kwargs['num_sanity_val_steps'] = 0
        trainer_kwargs['log_every_n_steps'] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    # use tensor cores
    torch.set_float32_matmul_precision('medium')

    trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint)

    # Ensure we test the best checkpoint (not just the last in-memory weights).
    # PyTorch 2.6+ defaults torch.load(weights_only=True) which can fail for Lightning checkpoints
    # that contain non-tensor state (e.g., numpy objects). To avoid this (and to stay compatible
    # across Lightning versions), we manually load the best checkpoint into the model.
    best_ckpt_path = None
    if not getattr(args, 'disable_checkpoints', False):
        ckpt_callbacks = [cb for cb in callbacks if isinstance(cb, BestModelCheckpoint)]
        if ckpt_callbacks:
            best_ckpt_path = Path(ckpt_callbacks[0].best_model_path)

    if best_ckpt_path and best_ckpt_path.is_file():
        checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if getattr(trainer, "is_global_zero", True):
            print(f"Loaded best checkpoint for testing: {best_ckpt_path}")
    else:
        if getattr(trainer, "is_global_zero", True):
            if getattr(args, 'disable_checkpoints', False):
                print("Checkpoint saving disabled; testing current in-memory model.")
            else:
                print("WARNING: Best checkpoint not found; testing current in-memory model.")

    trainer.test(model=model, datamodule=data_module)

    if not getattr(args, 'disable_checkpoints', False):
        process_checkpoint(args, callbacks, wandb_logger)

    # Ensure WandB flushes/saves offline files before process exit (helpful on HPC offline runs)
    try:
        wandb.finish()
    except Exception:
        pass

def process_checkpoint(args, callbacks, wandb_logger):
    checkpoint_path = Path(callbacks[0].best_model_path)
    # log to wandb
    log_weights_to_wandb(wandb_logger, checkpoint_path)

def setup_model_and_dataloaders(args, callbacks):
    if args.checkpoint: 
        model, data_module = load_checkpoint(args, args.data_dir)
        callbacks.append(restore_optimizer(args.checkpoint))
    else:
        model, data_module = setup_model_parameters(args)
    return model,data_module

def build_callbacks(args, file_name):
    callbacks = []
    if not getattr(args, 'disable_checkpoints', False):
        callbacks.append(build_checkpoint_callbacks(args, file_name, args.checkpoint_dir, args.checkpoint))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    return callbacks

def restore_optimizer_state(model):
    optim = model.optimizers()
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    optim.load_state_dict(checkpoint['state_dict'])

def setup_wandb_logger(args, model):
    def _is_global_zero():
        # Check common env vars used by SLURM/torch distributed to detect global-zero
        for key in ("LOCAL_RANK", "SLURM_LOCALID", "RANK", "SLURM_PROCID"):
            v = os.environ.get(key)
            if v is not None:
                try:
                    return int(v) == 0
                except Exception:
                    continue
        # Default to True when no explicit rank information is present
        return True

    global_zero = _is_global_zero()
    logger_dir = args.logger_dir
    # Normalize and validate WANDB_MODE to the allowed literals for wandb.init
    _wandb_mode_env = os.environ.get('WANDB_MODE', '')
    _wandb_mode_env_l = str(_wandb_mode_env).lower()
    if _wandb_mode_env_l in ('online', 'offline', 'disabled'):
        wandb_mode = _wandb_mode_env_l
    else:
        # fall back to None so wandb.init uses its default behavior
        wandb_mode = None

    if global_zero:
        # Print useful debug info so users can confirm logger initialization in job logs
        print("[setup_wandb_logger] initializing logger (global_zero=True)")
        print(f"[setup_wandb_logger] WANDB_MODE={wandb_mode}")
        print(f"[setup_wandb_logger] run_name={args.run_name}, project={args.project}")

        if logger_dir is None or str(logger_dir).lower() in {"none", "null", ""}:
            # Use a temporary directory to avoid filling persistent disk.
            global _WANDB_TMPDIR
            if _WANDB_TMPDIR is None:
                _WANDB_TMPDIR = TemporaryDirectory()
            logger_dir = _WANDB_TMPDIR.name

        print(f"[setup_wandb_logger] logger_dir={logger_dir}")

        wandb_experiment = wandb.init(
            config=model.hparams,
            project=args.project,
            name=args.run_name,
            dir=logger_dir,
            mode=wandb_mode,
            settings=wandb.Settings(save_code=False),
        )
        logger = WandbLogger(experiment=wandb_experiment)
    else:
        print("[setup_wandb_logger] not global zero -> using DummyLogger")
        logger = DummyLogger()

    return logger
    

def setup_model_parameters(args):
    # setup model configurations

    val_permute_contrasts = getattr(args, 'val_permute_contrasts', False)
    val_num_permutations_per_slice = getattr(args, 'val_num_permutations_per_slice', None)
    if val_num_permutations_per_slice is None:
        val_num_permutations_per_slice = getattr(args, 'num_permutations_per_slice', 1)
    val_permutation_seed = getattr(args, 'val_permutation_seed', None)
    if val_permutation_seed is None:
        val_permutation_seed = getattr(args, 'permutation_seed', 0)
    val_include_identity_permutation = getattr(args, 'val_include_identity_permutation', None)
    if val_include_identity_permutation is None:
        val_include_identity_permutation = getattr(args, 'include_identity_permutation', True)

    data_module = UndersampledDataModule(
        args.dataset, 
        args.data_dir, 
        args.test_dir,
        batch_size=args.batch_size, 
        resolution=(args.ny, args.nx),
        num_workers=args.num_workers,
        contrasts=args.contrasts,
        sampling_method=args.sampling_method,
        R=args.R,
        R_values=getattr(args, 'R_values', None),
        R_curriculum_stages=getattr(args, 'R_curriculum_stages', None),
        self_supervised=(not args.supervised), 
        ssdu_partitioning=args.ssdu_partitioning,
        limit_volumes=args.limit_volumes,
        same_mask_every_epoch=args.same_mask_all_epochs, 
        norm_method=args.norm_method,
        seed=8,
        jointly_reconstructing=args.jointly_reconstructing,
        guided_single_contrast=args.guided_single_contrast,
        permute_contrasts=getattr(args, 'permute_contrasts', False),
        num_permutations_per_slice=getattr(args, 'num_permutations_per_slice', 1),
        permutation_seed=getattr(args, 'permutation_seed', 0),
        include_identity_permutation=getattr(args, 'include_identity_permutation', True),
        contrast_excluding_training=getattr(args, 'contrast_excluding_training', None),
        val_permute_contrasts=val_permute_contrasts,
        val_num_permutations_per_slice=val_num_permutations_per_slice,
        val_permutation_seed=val_permutation_seed,
        val_include_identity_permutation=val_include_identity_permutation,
        validate_on_target=getattr(args, 'validate_on_target', False),
        target_dataset=getattr(args, 'target_dataset', None),
        target_dataset_path=getattr(args, 'target_dataset_path', None),
        target_contrasts=getattr(args, 'target_contrasts', None),
    ) 
    data_module.setup('train')

    varnet_config = VarnetConfig(
        contrast_order=data_module.contrast_order,
        metric_contrast_order=getattr(data_module, 'metrics_contrast_order', data_module.contrast_order),
        cascades=args.cascades, 
        channels=args.chans,
        depth=args.depth,
        model=args.model,
        upsample_method=getattr(args, 'upsample_method', 'conv'),
        conv_after_upsample=getattr(args, 'conv_after_upsample', False),
        promptmr_feature_dim_like_unet=getattr(args, 'promptmr_feature_dim_like_unet', False),
        promptmr_contrast_aware_stem=getattr(args, 'promptmr_contrast_aware_stem', True),
        promptmr_use_cabs=getattr(args, 'promptmr_use_cabs', True),
        promptmr_use_instancenorm=getattr(args, 'promptmr_use_instancenorm', True),
        promptmr_use_freq_cab=getattr(args, 'promptmr_use_freq_cab', False),
        promptmr_use_fremodule=getattr(args, 'promptmr_use_fremodule', False),
        promptmr_use_prompt_injection=getattr(args, 'promptmr_use_prompt_injection', True),
        promptmr_contrast_attn_heads=getattr(args, 'promptmr_contrast_attn_heads', 1),
        promptmr_contrast_attn_gate_init=getattr(args, 'promptmr_contrast_attn_gate_init', 0.0),
        promptmr_stem_use_freq_mix=getattr(args, 'promptmr_stem_use_freq_mix', False),
        promptmr_stem_mix_always_on=getattr(args, 'promptmr_stem_mix_always_on', False),
        promptmr_stem_mix_freq_mode=getattr(args, 'promptmr_stem_mix_freq_mode', 'low'),
        promptmr_stem_separate_per_contrast_conv=getattr(args, 'promptmr_stem_separate_per_contrast_conv', False),
        promptmr_enable_buffer=getattr(args, 'promptmr_enable_buffer', False),
        promptmr_enable_history=getattr(args, 'promptmr_enable_history', False),
    )

    partitioning_config = LearnPartitionConfig(
        image_size=(len(data_module.contrast_order), args.ny, args.nx),
        inital_R_value=args.R_hat,
        k_center_region = 10,
        sigmoid_slope_probability = args.sigmoid_slope1,
        sigmoid_slope_sampling = args.sigmoid_slope2,
        is_warm_start = args.warm_start,
        sampling_method = args.sampling_method,
        is_learn_R = args.learn_R,
        line_constrained=args.line_constrained

    )

    tripple_pathway_config = DualDomainConifg(
        is_pass_inverse=args.pass_inverse_data,
        is_pass_original=args.pass_all_data,
        inverse_no_grad=args.inverse_data_no_grad,
        original_no_grad=args.all_data_no_grad,
        pass_all_lines=args.pass_all_lines,
        pass_through_size=args.pass_through_size,
        seperate_models=args.seperate_model
    )

    # Backwards compat: older runs used --supervised_image to enable this.
    # If the explicit flag is set, it takes precedence.
    use_supervised_image_loss = getattr(args, "use_supervised_image_loss", None)
    if use_supervised_image_loss is None:
        use_supervised_image_loss = bool(getattr(args, "supervised_image", False))

    model = LearnedSSLLightning(
        partitioning_config,         # learn_partitioning_config
        varnet_config,               # varnet_config
        tripple_pathway_config,      # dual_domain_config
        args.lr,
        args.lr_scheduler,
        args.image_scaling_lam_inv,
        args.image_scaling_lam_full,
        args.image_scaling_full_inv,
        args.image_loss_grad_scaling,
        args.lambda_scaling,
        args.image_loss,
        args.k_loss,
        args.learn_sampling,
        args.warmup_training,
        bool(use_supervised_image_loss),
        getattr(args, 'is_mask_testing', True),
        args.warmup_adam,
        args.weight_decay,
        args.guided_single_contrast,
        getattr(args, 'enable_synthetic_contrast_eval', False),
    )

    # Ensure `CustomLogger` root uses the provided project/run_name from CLI or config
    try:
        proj = getattr(args, 'project', None)
        run = getattr(args, 'run_name', None)
        logger_dir = getattr(args, 'logger_dir', './logs') or './logs'
        if proj and run and hasattr(model, '_custom_logger'):
            model._custom_logger.root_dir = os.path.join(str(logger_dir), f"{proj}_{run}")
            os.makedirs(model._custom_logger.root_dir, exist_ok=True)

        # Propagate optional epoch-log zipping settings.
        if hasattr(model, '_custom_logger'):
            model._custom_logger.zip_epoch_logs = bool(getattr(args, 'zip_epoch_logs', False))
            model._custom_logger.zip_remove_original = bool(getattr(args, 'zip_remove_original', True))
    except Exception:
        pass

    return model, data_module

def log_weights_to_wandb(wandb_logger, checkpoint_path):
    # only run on the first rank if dataparallel job
    print(os.environ.get('SLURM_LOCALID'))
    if os.environ.get('SLURM_LOCALID'):
        rank = os.environ.get('SLURM_LOCALID')
    else:
        rank = 0

    if rank == 0:
        checkpoint_name = f"model-{wandb_logger.experiment.id}"

        # always remove optimizer state when logging to wandb
        with TemporaryDirectory() as tempdir: 
            temp_checkpoint = Path(tempdir) / 'model.ckpt'
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            torch.save(checkpoint, temp_checkpoint.as_posix())
            remove_optimizer_state(temp_checkpoint)

            artifact = wandb.Artifact(name=checkpoint_name, type="model")
            artifact.add_file(local_path=temp_checkpoint.as_posix(), name='model.ckpt')
            wandb_logger.experiment.log_artifact(artifact, aliases=['latest'])


def remove_optimizer_state(checkpoint_path, ):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'optimizer_states' in checkpoint:
        del checkpoint['optimizer_states']
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(args, data_dir):
    print("Loading Checkpoint!")
    data_module_kwargs = {}
    if os.environ.get('SLURM_LOCALID') is not None:
        device = f"cuda:{int(os.environ['SLURM_LOCALID'])}"
    else:
        device = 'cpu' if torch.cuda.is_available() else 'cuda:0'
    if data_dir:
        data_module_kwargs['data_dir'] = data_dir
        data_module_kwargs['device'] = device
    model = LearnedSSLLightning.load_from_checkpoint(args.checkpoint, lr=args.lr, map_location=device)
    data_module = UndersampledDataModule.load_from_checkpoint(args.checkpoint, **data_module_kwargs)
    data_module.setup('train')
    return model, data_module

def build_checkpoint_callbacks(args, file_name, checkpoint_dir, checkpoint_path=None):
    if checkpoint_path:
        # If resuming, keep checkpoints next to the provided ckpt.
        checkpoint_dir = Path(checkpoint_path).parent

    # Save the best model by validation masked mean SSIM.
    # Metric is logged by LearnedSSLLightning.log_image_space_metrics() as:
    #   "val/maskedmean_ssim_full"
    return BestModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=file_name + '-best-{epoch:02d}',
        save_top_k=1,
        monitor='val/maskedmean_ssim_full',
        mode='max',
        save_last=False,
        save_weights_only=False,
        test_on_best=bool(getattr(args, 'test_on_best', False)),
        test_on_best_update_wandb_summary=bool(getattr(args, 'test_on_best_update_wandb_summary', False)),
        test_on_best_log_wandb_history=bool(getattr(args, 'test_on_best_log_wandb_history', False)),
        test_on_best_history_step=int(getattr(args, 'test_on_best_history_step', 0)),
    )

def get_unique_file_name(args):
    unique_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    file_name = unique_id
    if args.run_name and not args.checkpoint: 
        contrasts = ','.join(args.contrasts)
        if getattr(args, 'R_values', None):
            r_str = '-'.join(str(r) for r in args.R_values)
            r_tag = f'Rvalues[{r_str}]'
        else:
            r_tag = str(args.R)
        file_name = f'{args.run_name}_{r_tag}_{args.supervised}_{contrasts}_{unique_id}'
    if args.checkpoint: 
        file_name = f'{args.run_name}_{unique_id}'
    return file_name


if __name__ == '__main__': 
    parser = ArgumentParser(description="Deep learning multi-contrast self-supervised reconstruction")

    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--num_workers', type=int, default=3)
    training_group.add_argument('--max_epochs', type=int, default=50)
    training_group.add_argument('--batch_size', type=int, default=1)
    training_group.add_argument('--lr', type=float, default=1e-3)
    training_group.add_argument(
        '--disable_checkpoints',
        action='store_true',
        help='Disable saving any checkpoints (best/last).',
    )
    training_group.add_argument('--lr_scheduler', action='store_true') 
    training_group.add_argument('--warmup_adam', action='store_true') 
    training_group.add_argument('--weight_decay', type=float, default=0) 
    training_group.add_argument('--checkpoint', type=str)
    training_group.add_argument("--config", "-c", type=str, help="Path to the YAML configuration file.")
    training_group.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Path to checkpoint save dir")
    training_group.add_argument("--fast_dev_run", action='store_true')
    training_group.add_argument('--grad_clip_val', type=float, default=0.0, help='Global gradient clipping value (0 disables).')
    training_group.add_argument('--grad_clip_algo', type=str, default='norm', choices=['norm', 'value'], help='Gradient clipping algorithm.')
    training_group.add_argument(
        '--sanity_check_run',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If true, run a tiny debugging pass (5 epochs; 2 train batches + 2 val batches per epoch; limited test) to quickly '
            'exercise best-checkpoint intermediate testing (useful for multi-R test loaders).'
        ),
    )
    
    # dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument('--R', type=float, default=6.0)
    dataset_group.add_argument('--R_values', type=float, nargs='*', default=None)
    dataset_group.add_argument('--R_curriculum_stages', default=None)
    dataset_group.add_argument('--dataset', type=str, default='m4raw')
    dataset_group.add_argument('--contrasts', type=str, nargs='+', default=['t1', 't2', 'flair'])
    dataset_group.add_argument(
        '--validate_on_target',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, run extra val/test loaders on a target dataset for generalization.',
    )
    dataset_group.add_argument('--target_dataset_path', type=str, default=None)
    dataset_group.add_argument('--target_dataset', type=str, default=None)
    dataset_group.add_argument('--target_contrasts', type=str, nargs='*', default=None)
    dataset_group.add_argument(
        '--contrast_excluding_training',
        type=str,
        nargs='*',
        default=None,
        help='Optional list of contrasts to exclude from training loss only (still used in val/test).',
    )
    dataset_group.add_argument(
        '--enable_synthetic_contrast_eval',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, evaluate synthetic contrast averages in val/test and log synthetic metrics.',
    )
    dataset_group.add_argument('--data_dir', type=str)
    dataset_group.add_argument('--test_dir', type=str, default=None)
    dataset_group.add_argument('--nx', type=int, default=256)
    dataset_group.add_argument('--ny', type=int, default=256)
    dataset_group.add_argument('--limit_volumes', type=float, default=1.0)
    dataset_group.add_argument('--sampling_method', type=str, choices=['2d', '1d', 'pi'], default='2d')
    dataset_group.add_argument('--ssdu_partitioning', action='store_true')
    dataset_group.add_argument('--same_mask_all_epochs', action='store_true')
    dataset_group.add_argument('--norm_method', type=str, choices=['image_mean', 'k', 'max'], default='image_mean')
    dataset_group.add_argument(
        '--jointly_reconstructing',
        type=lambda v: str(v).lower() in ['true', '1', 'yes', 'y'],
        default=True,
        help='If true, reconstruct all listed contrasts jointly. If false, treat each contrast as a separate single-contrast sample and shuffle across files/slices/contrasts.',
    )

    dataset_group.add_argument(
        '--guided_single_contrast',
        action='store_true',
        help=(
            'Only meaningful when --jointly_reconstructing=false. If set, each sample is a single target contrast '
            '(expanded dataset), but the input stacks all contrasts with the target contrast in channel 0 and the '
            'remaining contrasts in a fixed canonical order. Loss/metrics supervise only channel 0.'
        ),
    )

    dataset_group.add_argument(
        '--permute_contrasts',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Multi-contrast ablation (jointly_reconstructing=true): if set, the contrast axis is permuted per sample '
            'and the dataset is optionally expanded by --num_permutations_per_slice.'
        ),
    )
    dataset_group.add_argument(
        '--num_permutations_per_slice',
        type=int,
        default=1,
        help='Only used when --permute_contrasts=true. Expands dataset length by this factor (e.g. 4).',
    )
    dataset_group.add_argument(
        '--permutation_seed',
        type=int,
        default=0,
        help='Seed for deterministic per-slice permutation generation when --permute_contrasts=true.',
    )
    dataset_group.add_argument(
        '--include_identity_permutation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If set, perm_id=0 uses identity permutation (recommended).',
    )

    dataset_group.add_argument(
        '--val_permute_contrasts',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, apply the same contrast permutation augmentation to validation data.',
    )
    dataset_group.add_argument(
        '--val_num_permutations_per_slice',
        type=int,
        default=None,
        help='Validation-only number of permutations per slice. Defaults to --num_permutations_per_slice.',
    )
    dataset_group.add_argument(
        '--val_permutation_seed',
        type=int,
        default=None,
        help='Validation-only permutation seed. Defaults to --permutation_seed.',
    )
    dataset_group.add_argument(
        '--val_include_identity_permutation',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Validation-only identity permutation toggle. Defaults to --include_identity_permutation.',
    )

    # model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--R_hat', type=float, default=2.0)
    model_group.add_argument('--warm_start', action='store_true')
    model_group.add_argument('--chans', type=int, default=32)
    model_group.add_argument('--model', type=str, default='unet', choices=['unet', 'prompt_unet', 'promptmr_unet'])
    model_group.add_argument('--depth', type=int, default=4)
    model_group.add_argument('--cascades', type=int, default=6)
    model_group.add_argument('--upsample_method', type=str, default='conv', choices=['conv', 'bilinear', 'max'])
    model_group.add_argument(
        '--conv_after_upsample',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, add a 3x3 conv after upsampling (matches baseline UNet option).',
    )

    # PromptMRUNet ablation knobs (used only when --model=promptmr_unet)
    model_group.add_argument(
        '--promptmr_feature_dim_like_unet',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, use a UNet-like doubling channel schedule for PromptMRUNet feature_dim.',
    )
    model_group.add_argument(
        '--promptmr_contrast_aware_stem',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If set, use the contrast-aware stem in PromptMRUNet; if not set, use a UNet-like stem.',
    )
    model_group.add_argument(
        '--promptmr_use_cabs',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If set, use CAB blocks in PromptMRUNet; if not set, use UNet-like conv blocks and disable CABs in skip/bottleneck.',
    )
    model_group.add_argument(
        '--promptmr_use_instancenorm',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If set, use InstanceNorm inside PromptMRUNet blocks; if not set, disable InstanceNorm layers.',
    )
    model_group.add_argument(
        '--promptmr_use_freq_cab',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, enable frequency residuals inside all CABs (PromptMRUNet only).',
    )
    model_group.add_argument(
        '--promptmr_use_fremodule',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, enable the SpectraMR-style frequency branch before each PromptMRUNet decoder stage.',
    )

    model_group.add_argument(
        '--promptmr_use_prompt_injection',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If set, enable prompt injection in PromptMRUNet; if not set, use UNet-like decoder blocks with no prompts.',
    )
    model_group.add_argument(
        '--promptmr_stem_use_freq_mix',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, enable spatial+frequency contrast mixing in the PromptMRUNet stem.',
    )
    model_group.add_argument(
        '--promptmr_stem_mix_always_on',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'PromptMRUNet only: if true, bypass the learned stem_mix gate and always apply stem mixing at full strength. '
            'This forces the model to use the stem mixing path (ablation/debug option).'
        ),
    )
    model_group.add_argument(
        '--promptmr_stem_mix_freq_mode',
        type=str,
        default='low',
        choices=['low', 'high', 'all'],
        help='PromptMRUNet stem mix frequency token mode: low, high, or all.',
    )
    model_group.add_argument(
        '--promptmr_stem_separate_per_contrast_conv',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='PromptMRUNet only: if set, do NOT share per-contrast stem conv weights across contrasts.',
    )
    model_group.add_argument(
        '--promptmr_contrast_attn_heads',
        type=int,
        default=1,
        help='PromptMRUNet only: number of heads for the contrast-attention stem.',
    )
    model_group.add_argument(
        '--promptmr_contrast_attn_gate_init',
        type=float,
        default=0.0,
        help='PromptMRUNet only: initial gate value for contrast-attention mixing.',
    )
    model_group.add_argument(
        '--promptmr_enable_buffer',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='PromptMRUNet only: enable adaptive input buffer (PromptMR+ style).',
    )
    model_group.add_argument(
        '--promptmr_enable_history',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='PromptMRUNet only: enable feature history across cascades (PromptMR+ style).',
    )
    model_group.add_argument('--sigmoid_slope2', type=float, default=200)
    model_group.add_argument('--sigmoid_slope1', type=float, default=5)
    model_group.add_argument('--pass_through_size', type=int, default=10)
    model_group.add_argument('--pass_all_lines', action='store_true')
    model_group.add_argument('--seperate_model', action='store_true')
    model_group.add_argument('--learn_R', action='store_true')

    # loss function parameters
    model_group.add_argument('--image_scaling_lam_inv', type=float, default=0.0)
    model_group.add_argument('--image_scaling_lam_full', type=float, default=0.0)
    model_group.add_argument('--image_scaling_full_inv', type=float, default=0.0)
    model_group.add_argument('--k_loss', type=str, default='l1', choices=['l1', 'l2', 'l1l2'])
    model_group.add_argument('--image_loss', type=str, default='ssim', choices=['ssim', 'l1_grad', 'l1'])
    model_group.add_argument('--image_loss_grad_scaling', type=float, default=1.)
    model_group.add_argument('--lambda_scaling', type=float, default=0.65)
    model_group.add_argument('--line_constrained', action='store_true')

    model_group.add_argument('--use_schedulers', action='store_true')
    model_group.add_argument('--norm_loss_by_mask', action='store_true')
    model_group.add_argument('--warmup_training', action='store_true')

    # configure pathways in triple pathway
    model_group.add_argument('--pass_inverse_data', action='store_true')
    model_group.add_argument('--pass_all_data', action='store_true')
    model_group.add_argument('--inverse_data_no_grad', action='store_true')
    model_group.add_argument('--all_data_no_grad', action='store_true')

    # training type (supervised, self-supervised)
    model_group.add_argument('--supervised', action='store_true')
    model_group.add_argument('--supervised_image', action='store_true')
    model_group.add_argument(
        '--use_supervised_image_loss',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Enable/disable the extra supervised image-space loss computed against fully-sampled (e.g., SSIM). '
            'If unset, this falls back to --supervised_image for backward compatibility.'
        ),
    )
    model_group.add_argument('--learn_sampling', action='store_true')
    
    #logging parameters
    logger_group = parser.add_argument_group('Logging Parameters')
    logger_group.add_argument('--project', type=str, default='MRI Reconstruction')
    logger_group.add_argument('--run_name', type=str)
    logger_group.add_argument('--logger_dir', type=str, default=None)
    logger_group.add_argument(
        '--test_on_best',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Advanced override. If true, whenever a new best checkpoint is saved (by val/maskedmean_ssim_full), '
            'run a test pass immediately on that checkpoint. Prefer using --intermediate_test to enable/disable this behavior.'
        ),
    )

    # Single "master" flag for YAML/CLI: enables/disables intermediate best-checkpoint testing.
    # When false, we revert to the original behavior (only a single test at the end).
    logger_group.add_argument(
        '--intermediate_test',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If true, run test whenever a new best checkpoint is saved and update WandB. If false, only run the final test at the end.',
    )
    logger_group.add_argument(
        '--test_on_best_update_wandb_summary',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'If true (default) and using WandB, write the latest best-checkpoint test means into wandb summary '
            'under best_test/* keys (overwrites previous best_test values).'
        ),
    )
    logger_group.add_argument(
        '--test_on_best_log_wandb_history',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If true (default) and using WandB, also log best_test/* metrics into WandB history so they appear '
            'in charts. Note: W&B steps must be monotonically increasing; if you set a fixed step that would go '
            'backwards (e.g. 0 late in training), we clamp it to the current run step.'
        ),
    )
    logger_group.add_argument(
        '--test_on_best_history_step',
        type=int,
        default=0,
        help='Desired WandB step for best_test/* history logging (default 0). If it would be non-monotonic, it is clamped to the current run step.',
    )
    logger_group.add_argument(
        '--zip_epoch_logs',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, zip per-epoch image log folders (e.g., logs/.../000/) at the end of each validation epoch.',
    )
    logger_group.add_argument(
        '--zip_remove_original',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true (default), delete the per-epoch folder after creating the zip archive.',
    )
    
    args = parser.parse_args()

    args = replace_args_from_config(args.config, args, parser)

    # If enabled, sanity_check_run should force the intermediate-test pathway on and ensure checkpointing.
    # This is intended for fast correctness checks, not for producing reportable results.
    if getattr(args, 'sanity_check_run', False):
        args.disable_checkpoints = False
        args.intermediate_test = True

    # Enforce "intermediate_test" as the single source of truth.
    # If disabled, turn off all intermediate-testing behavior so runs behave like the original code.
    if not getattr(args, 'intermediate_test', False):
        args.test_on_best = False
        args.test_on_best_update_wandb_summary = False
        args.test_on_best_log_wandb_history = False
    else:
        # If enabled, ensure core behavior is on.
        args.test_on_best = True
        args.test_on_best_update_wandb_summary = True
        # User requested: do not plot intermediate-test charts; summary-only.
        args.test_on_best_log_wandb_history = False

    # Backwards-compat: allow configs to use `logging_dir` key (legacy).
    # If `logger_dir` wasn't provided via CLI or config replacement, try
    # to read `logging_dir` directly from the YAML file so offline wandb
    # runs write to the intended folder instead of a temp dir.
    if not getattr(args, 'logger_dir', None):
        # First, prefer any `logging_dir` attribute if replace_args_from_config set it.
        if getattr(args, 'logging_dir', None):
            args.logger_dir = getattr(args, 'logging_dir')
        else:
            # Fall back to reading the YAML config file directly (safe, optional).
            try:
                import yaml
                if getattr(args, 'config', None):
                    with open(args.config, 'r') as f:
                        cfg = yaml.safe_load(f)
                    # Top-level key `logging_dir` used in some configs
                    ld = cfg.get('logging_dir') if isinstance(cfg, dict) else None
                    if ld:
                        args.logger_dir = ld
            except Exception:
                pass

    main(args)
