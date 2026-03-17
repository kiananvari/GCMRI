from typing import Literal, Union, Optional
import dataclasses


import numpy as np
import os
import torch
from ml_recon.utils.custom_logger import CustomLogger
from ml_recon.utils.get_logging_root import get_logging_root
from pytorch_lightning.loggers import WandbLogger


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


from ml_recon.losses import L1L2Loss, SSIM_Loss, L1ImageGradLoss
from ml_recon.pl_modules.pl_ReconModel import plReconModel
from ml_recon.utils.evaluation_functions import nmse, psnr
from ml_recon.utils import k_to_img
from ml_recon.models.LearnPartitioning import LearnPartitioning, LearnPartitionConfig
from ml_recon.models.TriplePathway import TriplePathway, DualDomainConifg, VarnetConfig
from ml_recon.utils.evaluate_over_contrasts import evaluate_over_contrasts


class LearnedSSLLightning(plReconModel):
    def __init__(
        self,
        learn_partitioning_config: Union[LearnPartitionConfig, dict],
        varnet_config: Union[VarnetConfig, dict],
        dual_domain_config: Union[DualDomainConifg, dict],
        lr: float = 1e-3,
        lr_scheduler: bool = False,
        image_loss_scaling_lam_inv: float = 1e-4,
        image_loss_scaling_lam_full: float = 1e-4,
        image_loss_scaling_full_inv: float = 1e-4,
        image_loss_grad_scaling: float = 0.5,
        lambda_scaling: float = 1,
        image_loss_function: str = "ssim",
        k_space_loss_function: Literal["l1l2", "l1", "l2"] = "l1l2",
        enable_learn_partitioning: bool = True,
        enable_warmup_training: bool = False,
        use_supervised_image_loss: bool = False,
        is_mask_testing: bool = True,
        warmup_adam: bool = True,
        weight_decay: float = 0.0,
        guided_single_contrast: bool = False,
        enable_synthetic_contrast_eval: bool = False,
    ):
        """
        This function trains all MRI reconstruction models

        We train all models with this class. Supervised training is decided automatically based on the data given.

        We assume the dataloader has the keys:
        'undersampled': undersampled k-space from inital undersampling (Omega mask)
        'fs_k_space': fully sampled k-space
        'lambda_mask': One partition of k-space. If supervised, lambda_mask is the initial undersampling mask (Omega Mask)
        'loss_mask': Other partition of k-space. If supervised, loss_mask is all ones. 
        """

        self.automatic_optimization=False

        # since we convert to dicts for uploading to wandb, we need to convert back to dataclasses
        if isinstance(learn_partitioning_config, dict):
            learn_partitioning_config = LearnPartitionConfig(
                **learn_partitioning_config
            )
        if isinstance(varnet_config, dict):
            varnet_config = VarnetConfig(**varnet_config)
        if isinstance(dual_domain_config, dict):
            dual_domain_config = DualDomainConifg(**dual_domain_config)

        super().__init__(
            contrast_order=varnet_config.contrast_order,
            is_mask_testing=is_mask_testing,
            metric_contrast_order=getattr(varnet_config, 'metric_contrast_order', None),
        )

        if enable_learn_partitioning:
            self.partition_model = LearnPartitioning(learn_partitioning_config)
        else:
            self.partition_model = None

        self.recon_model = TriplePathway(dual_domain_config, varnet_config)

        # convert to dicts because save hyperparameter method does not like dataclasses
        dual_domain_config = dataclasses.asdict(dual_domain_config)  # type: ignore
        varnet_config = dataclasses.asdict(varnet_config)  # type: ignore
        learn_partitioning_config = dataclasses.asdict(learn_partitioning_config)  # type: ignore

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.image_scaling_lam_inv = image_loss_scaling_lam_inv
        self.image_scaling_lam_full = image_loss_scaling_lam_full
        self.image_scaling_full_inv = image_loss_scaling_full_inv
        self.lambda_loss_scaling = lambda_scaling
        self.enable_warmup_training = enable_warmup_training
        self.enable_learn_partitioning = enable_learn_partitioning
        self.use_superviesd_image_loss = use_supervised_image_loss
        self.test_metrics = is_mask_testing
        self.warmup_adam = warmup_adam
        self.weight_decay = weight_decay
        self.guided_single_contrast = guided_single_contrast
        self.enable_synthetic_contrast_eval = enable_synthetic_contrast_eval

        
        # loss function init
        self._setup_image_space_loss(image_loss_function, image_loss_grad_scaling)
        self._setup_k_space_loss(k_space_loss_function)
        self.save_hyperparameters()

        # --- Custom Logger Setup ---
        # Try to get config from varnet_config if possible, else fallback
        config = {
            'training_params': {
                'project': varnet_config.get('project', 'project') if isinstance(varnet_config, dict) else 'project',
                'run_name': varnet_config.get('run_name', 'run') if isinstance(varnet_config, dict) else 'run'
            },
            'logging_dir': './logs'
        }
        self._custom_logger = CustomLogger(config)

    def forward(self, k_space, mask, fs_k_space):
        estimate_k = self.recon_model.pass_through_model(
            self.recon_model.recon_model, k_space * mask, mask, fs_k_space
        )
        return estimate_k

    def training_step(self, batch, _):
        """
        Training loop function 

        This function loads data, partitions k-space if self-supervised then passes 
        data through triple pathways reconstruction network. The estimated outputs are
        then used to cacluate the loss function.

        Args:
            batch: dict, batch of data from above

        Returns:
            torch.Tensor, Returns final loss of this training batch

        Example:
            This is called internally by PyTorch Lightning
        """
        # get data
        fully_sampled = batch["fs_k_space"]
        undersampled_k = batch["undersampled"]

        # split data (loss mask is all ones if supervised)
        input_mask, loss_mask = self.partition_k_space(batch)
        include_mask = batch.get('include_contrast_mask', None)  # Ensure excluded contrasts do not contribute to any loss term (including image-space losses).
        include_mask_exp = None
        if include_mask is not None:
            include_mask_exp = include_mask
            while include_mask_exp.ndim < undersampled_k.ndim:
                include_mask_exp = include_mask_exp.unsqueeze(-1)
            undersampled_k = undersampled_k * include_mask_exp
            fully_sampled = fully_sampled * include_mask_exp
            input_mask = input_mask * include_mask_exp
            loss_mask = loss_mask * include_mask_exp

        # recon undersampled data
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            return_all=False,
        )

        # Also zero the corresponding output channels so excluded contrasts cannot
        # contribute through image-space losses.
        if include_mask_exp is not None:
            for key, value in list(estimates.items()):
                if value is not None:
                    estimates[key] = value * include_mask_exp

        # calculate loss
        contrast_ids = batch.get('contrast_id', None)
        contrast_perm = batch.get('contrast_perm', None)
        loss_dict = self.calculate_loss(
            estimates,
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            "train",
            contrast_ids=contrast_ids,
            contrast_perm=contrast_perm,
        )
        loss: torch.Tensor = sum(loss for loss in loss_dict.values())  # type: ignore

        # log loss components
        for key, value in loss_dict.items():
            self.log_scalar(f"train/{key}", value)

        # log full loss
        self.log_scalar("train/loss", loss, on_step=True, prog_bar=True)

        # log R value (for partitioning)
        if self.enable_learn_partitioning:
            self.log_R_value()

        # log ratio of sets
        initial_mask = undersampled_k != 0
        if contrast_perm is not None:
            input_mask_log = self._unpermute_contrast_axis(input_mask, contrast_perm)
            initial_mask_log = self._unpermute_contrast_axis(initial_mask, contrast_perm)
        else:
            input_mask_log = input_mask
            initial_mask_log = initial_mask
        self.log_k_space_set_ratios(input_mask_log, initial_mask_log)

        return loss

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_curriculum_epoch"):
            datamodule.set_curriculum_epoch(int(self.current_epoch))

    def log_scalar(self, label, metric, **kwargs):
        """
        Logs a scalar value to the logger.

        Args:
            label (str): The label for the scalar value.
            metric (float or torch.Tensor): The scalar value to log.
            **kwargs: Additional keyword arguments for the logger.
        """
        if "on_step" not in kwargs:
            kwargs["on_step"] = False

        self.log(label, metric, on_epoch=True, **kwargs)

    # Plotting of different metrics during training
    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx): 
        """
        Hook to call when training batch is finished

        This function plots images and training  metrics for visualization on WandB

        Args:
            outputs: torch.Tensor, loss from training_step
            batch: dict, same from training_step
            batch_idx: int, Integer of batch during training loop

        Returns:
            None

        Example:
            Called internally by PyTorch Lightning        
        """
        if not isinstance(self.logger, WandbLogger):
            return

        #log first batch of every 1st epoch
        if batch_idx != 0 or self.current_epoch % 1 != 0:
           return

        wandb_logger = self.logger

        fully_sampled = batch["fs_k_space"]
        undersampled_k = batch["undersampled"]

        input_mask, loss_mask = self.partition_k_space(batch)
        include_mask = batch.get('include_contrast_mask', None)
        if include_mask is not None:
            include_mask_exp = include_mask
            while include_mask_exp.ndim < undersampled_k.ndim:
                include_mask_exp = include_mask_exp.unsqueeze(-1)
            undersampled_k = undersampled_k * include_mask_exp
            fully_sampled = fully_sampled * include_mask_exp
            input_mask = input_mask * include_mask_exp
            loss_mask = loss_mask * include_mask_exp

        estimates = self.recon_model.forward(
            undersampled_k, fully_sampled, input_mask, loss_mask, return_all=True
        )

        # plot images (first of the batch)
        image_scaling = k_to_img(fully_sampled).amax((-1, -2), keepdim=True)

        # plot estimated images from each path 
        for pathway, estimate in estimates.items():
            estimate_images = self.k_to_img_scaled_clipped(estimate, image_scaling)

            wandb_logger.log_image(
                f"train/estimate_{pathway}",
                self.split_along_contrasts(estimate_images[0]),
            )

        # plot masks (first of the batch)
        initial_mask = (undersampled_k != 0)[0, :, 0, :, :]
        lambda_set_plot = input_mask[0, :, 0, :, :]
        loss_mask_plot = loss_mask[0, :, 0, :, :]
        loss_mask_wo_acs, lambda_k_wo_acs = TriplePathway.create_inverted_masks(
            input_mask,
            loss_mask,
            self.recon_model.dual_domain_config.pass_through_size,
            self.recon_model.dual_domain_config.pass_all_lines,
        )
        wandb_logger.log_image(
            "train_masks/lambda_set", self.split_along_contrasts(lambda_set_plot)
        )
        wandb_logger.log_image(
            "train_masks/loss_set", self.split_along_contrasts(loss_mask_plot)
        )
        wandb_logger.log_image(
            "train_masks/lambda_set_inverse", self.split_along_contrasts(lambda_k_wo_acs[0, :, 0, :, :])
        )
        wandb_logger.log_image(
            "train_masks/loss_set_inverse", self.split_along_contrasts(loss_mask_wo_acs[0, :, 0, :, :])
        )
        wandb_logger.log_image(
            "train_masks/initial_mask", self.split_along_contrasts(initial_mask)
        )

        # plot probability if learn partitioning
        if self.enable_learn_partitioning and self.partition_model:
            probability = self.partition_model.get_probability_distribution()
            if probability.shape[1] == 1: 
                probability = probability.tile((1, probability.shape[2], 1))
            wandb_logger.log_image(
                "probability", self.split_along_contrasts(probability), self.global_step
            )
    
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prefix = self._val_prefix(dataloader_idx)
        undersampled_k = batch["undersampled"]
        fully_sampled = batch["fs_k_space"]

        input_mask, loss_mask = self.partition_k_space(batch)
        estimates = self.recon_model.forward(
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            return_all=False,
        )

        # log loss
        contrast_ids = batch.get('contrast_id', None)
        contrast_perm = batch.get('contrast_perm', None)
        loss_dict = self.calculate_loss(
            estimates,
            undersampled_k,
            fully_sampled,
            input_mask,
            loss_mask,
            "val",
            contrast_ids=contrast_ids,
            contrast_perm=contrast_perm,
        )
        loss = sum(loss for loss in loss_dict.values())

        self.log_scalar(f"val_losses/{prefix}loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log_scalar(f"val_losses/{prefix}{key}", value)

        self.calculate_k_nmse(batch, prefix=prefix)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        prefix = self._val_prefix(dataloader_idx)

        # Log every epoch, up to 20 batches per epoch
        if batch_idx >= 10:
            plot_images = False
        else:
            plot_images = True


        estimate_k, fully_sampled_k, mask = self.infer_k_space(batch)

        # get images
        estimate_image = k_to_img(estimate_k, coil_dim=2)
        fully_sampled_image = k_to_img(fully_sampled_k, coil_dim=2)

        background_mask = self.get_image_background_mask(fully_sampled_image)

        scaling_factor = fully_sampled_image.amax((-1, -2), keepdim=True)

        estimate_image /= scaling_factor
        fully_sampled_image /= scaling_factor

        estimate_image_masked = estimate_image * background_mask
        fully_sampled_image_masked = fully_sampled_image * background_mask

        if self.enable_synthetic_contrast_eval:
            syn_full_k, syn_mask, syn_names = self._build_synthetic_contrasts(fully_sampled_k, mask)
            if syn_full_k is not None and syn_mask is not None:
                syn_under_k = syn_full_k * syn_mask
                syn_est_k = self.recon_model.pass_through_model(
                    self.recon_model.recon_model,
                    syn_under_k,
                    syn_mask,
                    syn_full_k,
                )
                syn_est_k = self.recon_model.final_dc_step(syn_under_k, syn_est_k, syn_mask)

                syn_est_img = k_to_img(syn_est_k, coil_dim=2)
                syn_full_img = k_to_img(syn_full_k, coil_dim=2)

                syn_scaling = syn_full_img.amax((-1, -2), keepdim=True)
                syn_est_img /= syn_scaling
                syn_full_img /= syn_scaling

                syn_background = self.get_image_background_mask(syn_full_img)
                syn_est_masked = syn_est_img * syn_background
                syn_full_masked = syn_full_img * syn_background

                self._log_image_space_metrics_named(
                    syn_est_img,
                    syn_full_img,
                    label="syn_",
                    prefix=prefix,
                    contrast_names=syn_names,
                )
                self._log_image_space_metrics_named(
                    syn_est_masked,
                    syn_full_masked,
                    label="syn_masked_",
                    prefix=prefix,
                    contrast_names=syn_names,
                )

        diff_masked = (estimate_image_masked - fully_sampled_image_masked).abs() * 10
        diff_unmasked = (estimate_image - fully_sampled_image).abs() * 10

        # --- Robust image dict construction ---
        def ensure_chw(x):
            # x: torch.Tensor or np.ndarray
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            else:
                x = np.asarray(x)
            # Handle batch dimension robustly
            if x.ndim == 5:  # (B, C, coils, H, W)
                return np.stack([ensure_chw(xi) for xi in x])
            if x.ndim == 4:
                # (C, coils, H, W) or (C, H, W, ?) or (B, H, W)
                x = np.squeeze(x)
                if x.ndim == 2:
                    x = x[None, ...]
                return x
            x = np.squeeze(x)
            if x.ndim == 2:
                x = x[None, ...]
            return x

        if plot_images:
            from ml_recon.utils.evaluation_functions import psnr
            from torchmetrics.functional.image import structural_similarity_index_measure as ssim
            sample_id = str(batch.get('sample_id', str(batch_idx)))
            slice_id = int(batch.get('slice_id', 0))
            contrast_names = self.contrast_order if hasattr(self, 'contrast_order') else [str(i) for i in range(estimate_image.shape[1])]

            # Compute zerofilled image from the (unpermuted/sliced) undersampled k-space so
            # the contrast channels align with `fully_sampled_image` and `scaling_factor`.
            undersampled_k = batch["undersampled"].clone()
            # If dataset permutes contrast axis, unpermute undersampled k-space as well
            if (not self.guided_single_contrast) and isinstance(batch, dict) and ("contrast_perm" in batch):
                undersampled_k = self._unpermute_contrast_axis(undersampled_k, batch["contrast_perm"])
            if self.guided_single_contrast:
                undersampled_k = undersampled_k[:, 0:1, ...]

            zerofilled_img = k_to_img(undersampled_k, coil_dim=2)
            # If tensor: do RSS and normalization in torch, then convert to numpy
            if isinstance(zerofilled_img, torch.Tensor):
                # ensure batch dim exists
                if zerofilled_img.ndim == 3:
                    zerofilled_img = zerofilled_img.unsqueeze(0)
                if zerofilled_img.ndim == 4 and zerofilled_img.shape[2] != scaling_factor.shape[2]:
                    # If coil dim still present in unexpected position, try RSS over coils (safe no-op if already removed)
                    # handled inside k_to_img normally; this is a safeguard
                    pass
                # RSS was applied by k_to_img; divide using scaling_factor (shapes should now align)
                # Broadcast-safe division
                try:
                    zerofilled_img = zerofilled_img / scaling_factor
                except Exception:
                    # fallback: align channel dim by slicing or expanding
                    cf = scaling_factor.shape[1]
                    cz = zerofilled_img.shape[1]
                    if cz > cf:
                        zerofilled_img = zerofilled_img[:, :cf, ...]
                    elif cz < cf:
                        # repeat zerofilled channels to match
                        repeats = (1, int(cf / cz) + 1, 1, 1)
                        zerofilled_img = zerofilled_img.repeat(repeats)[:, :cf, ...]
                    zerofilled_img = zerofilled_img / scaling_factor
                zerofilled_img = zerofilled_img.detach().cpu().numpy()
            else:
                # numpy array path
                if zerofilled_img.ndim == 4:
                    zerofilled_img = np.sqrt(np.sum(np.abs(zerofilled_img) ** 2, axis=1))
                # convert scaling_factor to numpy for safe division
                sf_np = scaling_factor.detach().cpu().numpy()
                # align shapes: zerofilled_img may be (C,H,W) or (B,C,H,W)
                if zerofilled_img.ndim == 3:
                    zerofilled_img = zerofilled_img[None, ...]
                # slice or repeat channels to match
                if zerofilled_img.shape[1] != sf_np.shape[1]:
                    cf = sf_np.shape[1]
                    cz = zerofilled_img.shape[1]
                    if cz > cf:
                        zerofilled_img = zerofilled_img[:, :cf, ...]
                    else:
                        rep = int(np.ceil(cf / cz))
                        zerofilled_img = np.tile(zerofilled_img, (1, rep, 1, 1))[:, :cf, ...]
                zerofilled_img = zerofilled_img / sf_np
            zerofilled_img = ensure_chw(zerofilled_img)

            # Helper to convert tensors/arrays into numpy with shape (B, C, H, W)
            def to_bchw(x):
                # x can be torch.Tensor or np.ndarray
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu()
                    if x.ndim == 5:
                        # (B, C, coils, H, W) -> RSS over coils
                        x = torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=2))
                    elif x.ndim == 4:
                        # assume (B, C, H, W)
                        pass
                    elif x.ndim == 3:
                        # (C, H, W) -> (1, C, H, W)
                        x = x.unsqueeze(0)
                    else:
                        raise ValueError(f"Unsupported tensor shape {x.shape} for conversion to B,C,H,W")
                    return x.numpy()
                else:
                    x = np.asarray(x)
                    if x.ndim == 5:
                        # (B, C, coils, H, W)
                        x = np.sqrt(np.sum(np.abs(x) ** 2, axis=2))
                    elif x.ndim == 4:
                        # assume (B, C, H, W)
                        pass
                    elif x.ndim == 3:
                        x = x[None, ...]
                    else:
                        raise ValueError(f"Unsupported array shape {x.shape} for conversion to B,C,H,W")
                    return x

            # Convert all images to (B, C, H, W) numpy arrays for per-sample logging
            zer_arr = to_bchw(zerofilled_img)
            est_arr = to_bchw(estimate_image)
            gt_arr = to_bchw(fully_sampled_image)
            diff_arr = to_bchw(diff_unmasked)

            batch_size = est_arr.shape[0]
            num_contrasts = est_arr.shape[1]

            # Broadcast arrays with batch dim == 1 to match `est_arr` batch size.
            def _broadcast_to_batch(arr, target_batch):
                arr = np.asarray(arr)
                if arr.ndim != 4:
                    return arr
                if arr.shape[0] == target_batch:
                    return arr
                if arr.shape[0] == 1:
                    return np.tile(arr, (target_batch, 1, 1, 1))
                # If arr has unexpected batch dim, slice or pad conservatively
                if arr.shape[0] > target_batch:
                    return arr[:target_batch]
                return arr

            zer_arr = _broadcast_to_batch(zer_arr, batch_size)
            gt_arr = _broadcast_to_batch(gt_arr, batch_size)
            diff_arr = _broadcast_to_batch(diff_arr, batch_size)
            # Ensure contrast names length matches available contrasts
            contrast_names_batch = list(contrast_names)[:num_contrasts]

            for b in range(batch_size):
                # sample and slice ids: support per-sample lists or scalars
                sample_id_all = batch.get('sample_id', None)
                if sample_id_all is not None and hasattr(sample_id_all, '__len__') and len(sample_id_all) == batch_size:
                    sample_id_b = str(sample_id_all[b])
                else:
                    sample_id_b = f"{sample_id}_{b}"

                slice_id_all = batch.get('slice_id', None)
                if slice_id_all is not None and hasattr(slice_id_all, '__len__') and len(slice_id_all) == batch_size:
                    slice_id_b = int(slice_id_all[b])
                else:
                    slice_id_b = slice_id

                for c, contrast in enumerate(contrast_names_batch):
                        # Build images_dict only with available image channels for this contrast
                        images_dict_c = {}
                        if c < zer_arr.shape[1]:
                            images_dict_c['zerofilled'] = zer_arr[b, c]
                        if c < est_arr.shape[1]:
                            images_dict_c['output'] = est_arr[b, c]
                        if c < gt_arr.shape[1]:
                            images_dict_c['ground_truth'] = gt_arr[b, c]
                        if c < diff_arr.shape[1]:
                            images_dict_c['diff'] = diff_arr[b, c]

                        # Compute metrics only if both estimate and gt exist for this contrast
                        if (c < est_arr.shape[1]) and (c < gt_arr.shape[1]):
                            # Use masked image-space metrics for filenames (masked PSNR/SSIM)
                            psnr_val = float(psnr(fully_sampled_image_masked[b, c], estimate_image_masked[b, c]))
                            ssim_res = ssim(
                                estimate_image_masked[b : b + 1, c : c + 1],
                                fully_sampled_image_masked[b : b + 1, c : c + 1],
                                data_range=1.0,
                                reduction='none',
                            )
                            if isinstance(ssim_res, tuple):
                                ssim_res = ssim_res[0]
                            ssim_val = float(ssim_res.item())
                            metrics = {'psnr': psnr_val, 'ssim': ssim_val}
                        else:
                            metrics = None

                        # Only log if at least one image is present for this contrast
                        if images_dict_c:
                            self._custom_logger.log_sample(
                                sample_id_b,
                                slice_id_b,
                                [contrast],
                                images_dict_c,
                                metrics,
                                epoch=self.current_epoch,
                                group=prefix,
                            )

        # log image space metrics
        contrast_ids = batch.get('contrast_id', None)
        perm_ids = batch.get('perm_id', None)
        self.log_image_space_metrics(
            estimate_image,
            fully_sampled_image,
            label="",
            prefix=prefix,
            contrast_ids=contrast_ids,
            perm_ids=perm_ids,
        )
        self.log_image_space_metrics(
            estimate_image_masked,
            fully_sampled_image_masked,
            label="masked",
            prefix=prefix,
            contrast_ids=contrast_ids,
            perm_ids=perm_ids,
        )

    def on_validation_epoch_end(self) -> None:
        self._log_aggregate_val_metrics()

        # Optionally zip per-epoch image logs to reduce inode/file-count usage.
        # This is useful on HPC filesystems with per-user file count limits.
        try:
            if (
                hasattr(self, '_custom_logger')
                and getattr(self._custom_logger, 'zip_epoch_logs', False)
                and getattr(self.trainer, 'is_global_zero', True)
            ):
                self._custom_logger.zip_epoch_dir(
                    int(self.current_epoch),
                    remove_original=getattr(self._custom_logger, 'zip_remove_original', True),
                )
        except Exception as e:
            # Don't crash training due to logging/archiving issues.
            print(f"[CustomLogger] Warning: failed to zip epoch {int(self.current_epoch):03d}: {e}")

    def _log_aggregate_val_metrics(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        r_values = getattr(datamodule, "val_r_values", None) if datamodule is not None else None
        if not r_values:
            return

        def _mean_across_r(group: str, name: str):
            values = []
            for idx in range(len(r_values)):
                prefix = self._val_prefix(idx)
                key = f"{group}/{prefix}{name}/dataloader_idx_{idx}"
                value = self.trainer.callback_metrics.get(key)
                if value is None:
                    fallback_key = f"{group}/{prefix}{name}"
                    value = self.trainer.callback_metrics.get(fallback_key)
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=self.device)
                values.append(value)
            if not values:
                return None
            return torch.stack(values).mean()

        for name in [
            "mean_ssim_full",
            "mean_nmse_full",
            "maskedmean_ssim_full",
            "maskedmean_nmse_full",
        ]:
            agg = _mean_across_r("val", name)
            if agg is not None:
                self.log(f"val/{name}", agg, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        for name in ["k-space_nmse"]:
            agg = _mean_across_r("val_nmse", name)
            if agg is not None:
                self.log(f"val_nmse/{name}", agg, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        for name in ["loss", "k_loss"]:
            agg = _mean_across_r("val_losses", name)
            if agg is not None:
                self.log(f"val_losses/{name}", agg, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

    def log_image_space_metrics(self, estimate_image, fully_sampled_image, label, prefix="", contrast_ids=None, perm_ids=None):
        # If permutations are present, log the overall metrics and then per-permutation curves.
        if perm_ids is not None:
            self._log_image_space_metrics_impl(
                estimate_image,
                fully_sampled_image,
                label=label,
                prefix=prefix,
                contrast_ids=contrast_ids,
            )

            if perm_ids.ndim > 1:
                perm_ids_flat = perm_ids.view(-1)
            else:
                perm_ids_flat = perm_ids

            contrast_ids_flat = None
            if contrast_ids is not None:
                contrast_ids_flat = contrast_ids.view(-1) if contrast_ids.ndim > 1 else contrast_ids

            for perm_value in torch.unique(perm_ids_flat).tolist():
                perm_value = int(perm_value)
                mask = perm_ids_flat == perm_value
                if not mask.any():
                    continue
                estimate_perm = estimate_image[mask]
                fully_sampled_perm = fully_sampled_image[mask]
                contrast_ids_perm = contrast_ids_flat[mask] if contrast_ids_flat is not None else None
                perm_label = perm_value + 1
                self._log_image_space_metrics_impl(
                    estimate_perm,
                    fully_sampled_perm,
                    label=f"{label}perm{perm_label}_",
                    prefix=prefix,
                    contrast_ids=contrast_ids_perm,
                )
            return

        self._log_image_space_metrics_impl(
            estimate_image,
            fully_sampled_image,
            label=label,
            prefix=prefix,
            contrast_ids=contrast_ids,
        )

    def _log_image_space_metrics_impl(self, estimate_image, fully_sampled_image, label, prefix="", contrast_ids=None):
        # Default multi-contrast behavior (same as original): compute metrics per contrast dimension
        # and log mean over contrasts.
        if (
            contrast_ids is None
            or fully_sampled_image.shape[1] > 1
            or (not self.guided_single_contrast and len(self.metric_contrast_order) == len(self.contrast_order))
        ):
            ssim_full = evaluate_over_contrasts(ssim, fully_sampled_image, estimate_image)
            nmse_full = evaluate_over_contrasts(nmse, fully_sampled_image, estimate_image)
            for i, contrast in enumerate(self.contrast_order):
                self.log(
                    f"val_ssim/{prefix}{label}ssim_full_{contrast}",
                    ssim_full[i],
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"val_nmse/{prefix}{label}nmse_full_{contrast}",
                    nmse_full[i],
                    on_epoch=True,
                    sync_dist=True,
                )
            self.log(
                f"val/{prefix}{label}mean_nmse_full",
                sum(nmse_full) / len(nmse_full),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/{prefix}{label}mean_ssim_full",
                sum(ssim_full) / len(nmse_full),
                on_epoch=True,
                sync_dist=True,
            )
            return

        # Single-contrast-per-sample: aggregate metrics by true contrast id so we still log
        # t1/t2/flair/t1ce (and mean over contrasts) like joint reconstruction runs.
        if contrast_ids.ndim > 1:
            contrast_ids = contrast_ids.view(-1)

        batch_values_ssim = []
        batch_values_nmse = []
        for contrast_id in torch.unique(contrast_ids).tolist():
            contrast_id = int(contrast_id)
            if contrast_id < 0 or contrast_id >= len(self.metric_contrast_order):
                continue
            mask = (contrast_ids == contrast_id)
            if not mask.any():
                continue

            gt = fully_sampled_image[mask]
            est = estimate_image[mask]
            # Shapes are [N, 1, H, W] so evaluate_over_contrasts returns a list of length 1.
            ssim_val = evaluate_over_contrasts(ssim, gt, est)[0]
            nmse_val = evaluate_over_contrasts(nmse, gt, est)[0]

            contrast_label = self.metric_contrast_order[contrast_id]
            self.log(
                f"val_ssim/{prefix}{label}ssim_full_{contrast_label}",
                ssim_val,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(mask.sum().item()),
            )
            self.log(
                f"val_nmse/{prefix}{label}nmse_full_{contrast_label}",
                nmse_val,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(mask.sum().item()),
            )
            batch_values_ssim.append(ssim_val)
            batch_values_nmse.append(nmse_val)

        if batch_values_ssim:
            self.log(
                f"val/{prefix}{label}mean_nmse_full",
                sum(batch_values_nmse) / len(batch_values_nmse),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val/{prefix}{label}mean_ssim_full",
                sum(batch_values_ssim) / len(batch_values_ssim),
                on_epoch=True,
                sync_dist=True,
            )

    def _log_image_space_metrics_named(self, estimate_image, fully_sampled_image, label, prefix, contrast_names):
        ssim_full = evaluate_over_contrasts(ssim, fully_sampled_image, estimate_image)
        nmse_full = evaluate_over_contrasts(nmse, fully_sampled_image, estimate_image)
        for i, contrast in enumerate(contrast_names):
            self.log(
                f"val_ssim/{prefix}{label}ssim_full_{contrast}",
                ssim_full[i],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val_nmse/{prefix}{label}nmse_full_{contrast}",
                nmse_full[i],
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            f"val/{prefix}{label}mean_nmse_full",
            sum(nmse_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"val/{prefix}{label}mean_ssim_full",
            sum(ssim_full) / len(nmse_full),
            on_epoch=True,
            sync_dist=True,
        )

    def _log_test_metrics_named(self, estimate_k, ground_truth_image, prefix, contrast_names):
        estimated_image = k_to_img(estimate_k, coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        estimated_image /= scaling_factor
        ground_truth_image = ground_truth_image / scaling_factor

        background_mask = self.get_image_background_mask(ground_truth_image)
        estimated_image_masked = estimated_image * background_mask
        ground_truth_image_masked = ground_truth_image * background_mask

        for contrast_index, contrast_label in enumerate(contrast_names):
            for i in range(ground_truth_image.shape[0]):
                contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                contrast_ground_truth_masked = ground_truth_image_masked[i, contrast_index, :, :]
                contrast_estimated = estimated_image[i, contrast_index, :, :]
                contrast_estimated_masked = estimated_image_masked[i, contrast_index, :, :]

                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]
                contrast_ground_truth_masked = contrast_ground_truth_masked[None, None, :, :]
                contrast_estimated_masked = contrast_estimated_masked[None, None, :, :]

                nmse_val = nmse(contrast_ground_truth, contrast_estimated)
                nmse_val_masked = nmse(contrast_ground_truth_masked, contrast_estimated_masked)

                ssim_val_masked = ssim(
                    contrast_ground_truth_masked,
                    contrast_estimated_masked,
                    data_range=(contrast_ground_truth_masked.max().item()),
                    kernel_size=7,
                )
                ssim_val = ssim(
                    contrast_ground_truth,
                    contrast_estimated,
                    data_range=(0, contrast_ground_truth.max().item()),
                    kernel_size=7,
                )

                assert isinstance(ssim_val, torch.Tensor)
                assert isinstance(ssim_val_masked, torch.Tensor)

                psnr_val_masked = psnr(contrast_ground_truth_masked, contrast_estimated_masked)
                psnr_val = psnr(contrast_ground_truth, contrast_estimated)

                self.log(f"metrics/{prefix}nmse_{contrast_label}", nmse_val, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}ssim_{contrast_label}", ssim_val, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}psnr_{contrast_label}", psnr_val, sync_dist=True, on_step=True)

                self.log(
                    f"metrics/{prefix}masked_nmse_{contrast_label}",
                    nmse_val_masked,
                    sync_dist=True,
                    on_step=True,
                )
                self.log(
                    f"metrics/{prefix}masked_ssim_{contrast_label}",
                    ssim_val_masked,
                    sync_dist=True,
                    on_step=True,
                )
                self.log(
                    f"metrics/{prefix}masked_psnr_{contrast_label}",
                    psnr_val_masked,
                    sync_dist=True,
                    on_step=True,
                )


    def test_step(self, batch, batch_index, dataloader_idx=0):
        fully_sampled_image = None

        if type(batch) is list:
            fully_sampled_image = batch[1]
            batch = batch[0]

        estimate_k, fully_sampled_k, mask = self.infer_k_space(batch)

        if fully_sampled_image is None:
            fully_sampled_image = k_to_img(fully_sampled_k, coil_dim=2)  # fully sampled ground truth

        contrast_ids = None
        if isinstance(batch, dict):
            contrast_ids = batch.get('contrast_id', None)
        prefix = self._test_prefix(dataloader_idx)
        fs_metrics = self.my_test_step(
            (estimate_k, fully_sampled_image),
            batch_index,
            contrast_ids=contrast_ids,
            prefix=prefix,
        )

        if self.enable_synthetic_contrast_eval:
            syn_full_k, syn_mask, syn_names = self._build_synthetic_contrasts(fully_sampled_k, mask)
            if syn_full_k is not None and syn_mask is not None:
                syn_under_k = syn_full_k * syn_mask
                syn_est_k = self.recon_model.pass_through_model(
                    self.recon_model.recon_model,
                    syn_under_k,
                    syn_mask,
                    syn_full_k,
                )
                syn_est_k = self.recon_model.final_dc_step(syn_under_k, syn_est_k, syn_mask)
                syn_full_img = k_to_img(syn_full_k, coil_dim=2)
                syn_prefix = f"{prefix}syn_"
                self._log_test_metrics_named(
                    syn_est_k,
                    syn_full_img,
                    prefix=syn_prefix,
                    contrast_names=syn_names,
                )

        return {"fs_metrics": fs_metrics}

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        fully_sampled_image = None
        if type(batch) is list:
            fully_sampled_image = batch[1]
            batch = batch[0]
        estimate_k, fully_sampled, mask = self.infer_k_space(batch)

        if fully_sampled_image is None:
            fully_sampled_image = k_to_img(fully_sampled, coil_dim=2)  # noisy, fully sampled ground truth

        self.my_test_batch_end(
            outputs,
            (estimate_k, fully_sampled_image, mask),
            batch_idx,
            dataloader_idx,
        )

        # Also write per-slice PNG logs to disk via CustomLogger (if available).
        # To keep test runs lightweight, default is to only log the first batch per test loader.
        # If `self.log_all_test_images` is set True (e.g., from test_demo.py), log every batch.
        try:
            if (
                hasattr(self, '_custom_logger')
                and getattr(self, '_custom_logger') is not None
                and getattr(self.trainer, 'is_global_zero', True)
                and (int(batch_idx) == 0 or bool(getattr(self, 'log_all_test_images', False)))
            ):
                prefix = self._test_prefix(dataloader_idx)

                # Build normalized (0-1-ish) images and masked variants.
                # Note: background mask is computed on the (un-normalized) fully-sampled image for robustness.
                background_mask = self.get_image_background_mask(fully_sampled_image)
                scaling_factor = fully_sampled_image.amax((-1, -2), keepdim=True).clamp_min(1e-8)

                est_img = k_to_img(estimate_k, coil_dim=2) / scaling_factor
                gt_img = fully_sampled_image / scaling_factor
                est_masked = est_img * background_mask
                gt_masked = gt_img * background_mask

                diff = (gt_img - est_img).abs() * 10
                diff_masked = (gt_masked - est_masked).abs() * 10

                # Zerofilled image from undersampled k-space (if present in batch)
                zf_img = None
                if isinstance(batch, dict) and ('undersampled' in batch):
                    undersampled_k = batch['undersampled'].clone()
                    if (not self.guided_single_contrast) and ('contrast_perm' in batch):
                        undersampled_k = self._unpermute_contrast_axis(undersampled_k, batch['contrast_perm'])
                    if self.guided_single_contrast:
                        undersampled_k = undersampled_k[:, 0:1, ...]
                    zf_img = k_to_img(undersampled_k, coil_dim=2) / scaling_factor
                    zf_masked = zf_img * background_mask
                else:
                    zf_masked = None

                # Identify sample/slice ids if available.
                sample_id = None
                slice_id = None
                if isinstance(batch, dict):
                    sample_id = batch.get('sample_id', None)
                    slice_id = batch.get('slice_id', None)
                if sample_id is None:
                    sample_id = str(batch_idx)
                if slice_id is None:
                    slice_id = 0

                # Use canonical contrast labels.
                contrast_names = self.contrast_order if hasattr(self, 'contrast_order') else [str(i) for i in range(gt_img.shape[1])]

                # Per-sample loop (test batch_size is usually 1, but keep this safe).
                batch_size = int(gt_img.shape[0])
                num_contrasts = int(gt_img.shape[1])

                for b in range(batch_size):
                    # Resolve per-sample ids
                    if isinstance(sample_id, (list, tuple)) and len(sample_id) == batch_size:
                        sample_id_b = str(sample_id[b])
                    else:
                        sample_id_b = str(sample_id)

                    if isinstance(slice_id, (list, tuple)):
                        if len(slice_id) == batch_size:
                            slice_id_b = int(slice_id[b])
                        elif len(slice_id) > 0:
                            slice_id_b = int(slice_id[0])
                        else:
                            slice_id_b = 0
                    else:
                        slice_id_b = int(slice_id)

                    for c, contrast in enumerate(list(contrast_names)[:num_contrasts]):
                        images_dict_c = {
                            'output': est_img[b, c].detach().cpu().numpy(),
                            'ground_truth': gt_img[b, c].detach().cpu().numpy(),
                            'diff': diff[b, c].detach().cpu().numpy(),
                            'output_masked': est_masked[b, c].detach().cpu().numpy(),
                            'ground_truth_masked': gt_masked[b, c].detach().cpu().numpy(),
                            'diff_masked': diff_masked[b, c].detach().cpu().numpy(),
                        }
                        if zf_img is not None:
                            images_dict_c['zerofilled'] = zf_img[b, c].detach().cpu().numpy()
                        if zf_masked is not None:
                            images_dict_c['zerofilled_masked'] = zf_masked[b, c].detach().cpu().numpy()

                        # Filenames use masked metrics (PSNR/SSIM) when available.
                        try:
                            psnr_val = float(psnr(gt_masked[b, c], est_masked[b, c]))
                            ssim_res = ssim(
                                est_masked[b : b + 1, c : c + 1],
                                gt_masked[b : b + 1, c : c + 1],
                                data_range=1.0,
                                reduction='none',
                            )
                            if isinstance(ssim_res, tuple):
                                ssim_res = ssim_res[0]
                            ssim_val = float(ssim_res.item())
                            metrics = {'psnr': psnr_val, 'ssim': ssim_val}
                        except Exception:
                            metrics = None

                        self._custom_logger.log_sample(
                            sample_id_b,
                            slice_id_b,
                            [contrast],
                            images_dict_c,
                            metrics,
                            epoch=int(getattr(self, 'current_epoch', 0)),
                            group=prefix,
                        )
        except Exception as e:
            if bool(getattr(self, 'raise_on_image_log_error', False)):
                raise
            print(f"[CustomLogger] Warning: failed to log test sample images: {e}")

    def _build_synthetic_contrasts(self, fully_sampled_k, mask):
        if fully_sampled_k.shape[1] != 4:
            return None, None, None

        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        syn_k = []
        syn_mask = []

        if hasattr(self, "contrast_order"):
            contrast_labels = self.contrast_order
        else:
            contrast_labels = [str(i) for i in range(fully_sampled_k.shape[1])]

        syn_names = []
        for a, b in pairs:
            syn_k.append(0.5 * (fully_sampled_k[:, a] + fully_sampled_k[:, b]))
            if mask is not None:
                syn_mask.append(((mask[:, a] + mask[:, b]) > 0).to(mask.dtype))
            syn_names.append(f"syn_{contrast_labels[a]}_{contrast_labels[b]}")

        syn_k = torch.stack(syn_k, dim=1)
        syn_mask_out = torch.stack(syn_mask, dim=1) if syn_mask else None
        return syn_k, syn_mask_out, syn_names


    @staticmethod
    def _unpermute_contrast_axis(x: torch.Tensor, contrast_perm: torch.Tensor) -> torch.Tensor:
        """Unpermute contrast axis back to canonical order.

        `contrast_perm` is shaped [B, C] (or [C] for a single sample) and stores the mapping used
        when permuting the dataset:
          x_perm[:, i] = x_canon[:, contrast_perm[:, i]]

        We compute inv = argsort(contrast_perm) so:
          x_canon[:, j] = x_perm[:, inv[:, j]]
        """
        if contrast_perm.ndim == 1:
            contrast_perm = contrast_perm.unsqueeze(0)

        inv = torch.argsort(contrast_perm, dim=1)
        b, c = inv.shape
        index_shape = (b, c) + (1,) * (x.ndim - 2)
        gather_index = inv.view(*index_shape).expand((b, c) + tuple(x.shape[2:]))
        return torch.gather(x, dim=1, index=gather_index)


    def infer_k_space(self, batch):
        k_space = batch
        scaling_factor = batch["scaling_factor"]
        fully_sampled_k = k_space["fs_k_space"].clone()
        undersampled = k_space["undersampled"].clone()
        mask = k_space["mask"].clone()

        # if self-supervised combine masks to pass all data.
        if k_space["is_self_supervised"].all():
            mask += k_space["loss_mask"]  # combine to get original sampling mask

        # pass inital data through model
        estimate_k = self.recon_model.pass_through_model(
            self.recon_model.recon_model, undersampled, mask, fully_sampled_k
        )
        estimate_k = self.recon_model.final_dc_step(undersampled, estimate_k, mask)

        # rescale based on scaling factor
        estimate_k *= scaling_factor
        fully_sampled_k *= scaling_factor

        # If dataset permutes contrast axis, unpermute back to canonical order so metrics/logs are meaningful.
        if (not self.guided_single_contrast) and isinstance(batch, dict) and ("contrast_perm" in batch):
            perm = batch["contrast_perm"]
            estimate_k = self._unpermute_contrast_axis(estimate_k, perm)
            fully_sampled_k = self._unpermute_contrast_axis(fully_sampled_k, perm)
            mask = self._unpermute_contrast_axis(mask, perm)
        if self.guided_single_contrast:
            estimate_k = estimate_k[:, 0:1, ...]
            fully_sampled_k = fully_sampled_k[:, 0:1, ...]
            mask = mask[:, 0:1, ...]
        return estimate_k, fully_sampled_k, mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        schedulers = []

        if self.warmup_adam:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1, total_iters=10
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )
        if self.lr_scheduler:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2000, T_mult=2, eta_min=1e-4
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            )
        return [optimizer], schedulers

    def calculate_k_nmse(self, batch, *, prefix=""):
        estimate_k, fully_sampled_k, _ = self.infer_k_space(batch)
        mse_value = (fully_sampled_k - estimate_k).pow(2).abs().sum((-1, -2, -3))
        l2_norm = fully_sampled_k.pow(2).abs().sum((-1, -2, -3))
        self.log_scalar(f"val_nmse/{prefix}k-space_nmse", (mse_value/l2_norm).mean())

    def _val_prefix(self, dataloader_idx: int) -> str:
        datamodule = getattr(self.trainer, "datamodule", None)
        prefixes = getattr(datamodule, "val_prefixes", None) if datamodule is not None else None
        if prefixes and int(dataloader_idx) < len(prefixes):
            return prefixes[int(dataloader_idx)]
        r_values = getattr(datamodule, "val_r_values", None) if datamodule is not None else None
        if r_values:
            r_val = r_values[int(dataloader_idx) % len(r_values)]
            if float(r_val).is_integer():
                r_label = f"R{int(r_val)}"
            else:
                r_label = f"R{r_val}"
            return f"{r_label}/"
        return ""

    def _test_prefix(self, dataloader_idx: int) -> str:
        datamodule = getattr(self.trainer, "datamodule", None)
        prefixes = getattr(datamodule, "test_prefixes", None) if datamodule is not None else None
        if prefixes and int(dataloader_idx) < len(prefixes):
            return prefixes[int(dataloader_idx)]
        r_values = getattr(datamodule, "test_r_values", None) if datamodule is not None else None
        if r_values:
            r_val = r_values[int(dataloader_idx) % len(r_values)]
            if float(r_val).is_integer():
                r_label = f"R{int(r_val)}"
            else:
                r_label = f"R{r_val}"
            return f"{r_label}/"
        return ""

    def _setup_image_space_loss(self, image_loss_function, image_loss_grad_scaling):
        if image_loss_function == "ssim":
            image_loss = SSIM_Loss(kernel_size=7, data_range=(0.0, 1.0))
        elif image_loss_function == "l1":
            image_loss = torch.nn.L1Loss()
        elif image_loss_function == "l1_grad":
            image_loss = L1ImageGradLoss(grad_scaling=image_loss_grad_scaling)
        else:
            raise ValueError(f"unsuported image loss function: {image_loss_function}")
        self.image_loss_func = image_loss

    def compute_image_loss(self, kspace1, kspace2, undersampled_k):
        scaling_factor = k_to_img(undersampled_k).amax((-1, -2), keepdim=True).clamp_min(1e-6)
        img_1 = k_to_img(kspace1) / scaling_factor
        img_2 = k_to_img(kspace2) / scaling_factor

        b, c, h, w = img_1.shape
        img_1 = img_1.view(b * c, 1, h, w)
        img_2 = img_2.view(b * c, 1, h, w)
        image_loss = self.image_loss_func(img_1, img_2)

        return image_loss

    def calculate_k_loss(
        self, estimate, fully_sampled, loss_mask, loss_scaling, loss_name=""
    ):
        k_losses = {}
        for contrast, index in zip(self.contrast_order, range(estimate.shape[1])):
            k_loss = self.k_space_loss(
                torch.view_as_real(fully_sampled[:, index, ...] * loss_mask[:, index, ...]),
                torch.view_as_real(estimate[:, index, ...] * loss_mask[:, index, ...]),
            )


            k_losses[f"k_loss_{loss_name}_{contrast}"] = k_loss * loss_scaling

        return k_losses

    def _setup_k_space_loss(self, k_space_loss_function):
        reduce = "mean"

        if k_space_loss_function == "l1l2":
            self.k_space_loss = L1L2Loss(norm_all_k=False)
        elif k_space_loss_function == "l1":
            self.k_space_loss = torch.nn.L1Loss(reduction=reduce)
        elif k_space_loss_function == "l2":
            self.k_space_loss = torch.nn.MSELoss(reduction=reduce)
        else:
            raise ValueError("No k-space loss!")

    def log_R_value(self):
        """
        Logs the R value for each contrast in the contrast order.

        The R value is obtained from the partition model and logged for each contrast.
        """
        if self.partition_model is None:
            return

        R_value = self.partition_model.get_R()
        for i, contrast in enumerate(self.contrast_order):
            self.log(f"sampling_metrics/R_{contrast}", R_value[i])

    def log_k_space_set_ratios(self, input_mask, initial_mask):
        for i, contrast in enumerate(self.contrast_order):
            self.log(
                f"sampling_metrics/lambda-over-inverse_{contrast}",
                input_mask[:, i, 0, :, :].sum() / initial_mask[:, i, 0, :, :].sum(),
                on_epoch=True,
                on_step=False,
            )

    def calculate_inverse_k_loss(
        self, lambda_mask, inverse_mask, inverse_estimate, undersampled_k
    ):
        _, lambda_k_wo_acs = TriplePathway.create_inverted_masks(
            lambda_mask,
            inverse_mask,
            self.recon_model.dual_domain_config.pass_through_size,
            self.recon_model.dual_domain_config.pass_all_lines,
        )


        k_loss_inverse = self.calculate_k_loss(
            inverse_estimate,
            undersampled_k,
            lambda_k_wo_acs,
            1 - self.lambda_loss_scaling,
            "inverse",
        )
        return k_loss_inverse

    def partition_k_space(self, batch):
        # compute either learned or heuristic partioning masks
        if self.enable_learn_partitioning and self.partition_model:
            assert (batch["mask"] * batch["loss_mask"] == 0).all()
            initial_mask = batch["mask"] + batch["loss_mask"]
            input_mask, loss_mask = self.partition_model(initial_mask)
        else:
            input_mask, loss_mask = batch["mask"], batch["loss_mask"]

        return input_mask, loss_mask

    def split_along_contrasts(self, image):
        return np.split(image.cpu().detach().numpy(), image.shape[0], 0)

    def k_to_img_scaled_clipped(self, k_space, scaling_factor):
        return (k_to_img(k_space) / scaling_factor).clip(0, 1)

    def calculate_loss(
        self,
        estimates,
        undersampled_k,
        fully_sampled,
        input_mask,
        dc_mask,
        label,
        contrast_ids=None,
        contrast_perm: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss for different pathways in the reconstruction process.
        Args:
            estimates (dict): Dictionary containing estimated k-space data from different paths.
                - 'lambda_path': Estimated k-space from the lambda path.
                - 'full_path': Estimated k-space from the full path.
                - 'inverse_path': Estimated k-space from the inverse path.
            undersampled_k (torch.Tensor): The undersampled k-space data.
            fully_sampled (torch.Tensor): The fully sampled k-space data.
            input_mask (torch.Tensor): The input mask for the k-space data.
            dc_mask (torch.Tensor): The data consistency mask.
        Returns:
            dict: A dictionary containing the calculated losses for different pathways if available.
            Only the losses for the available pathways are returned.
                - 'k_loss_lambda': Loss for the lambda path in k-space.
                - 'image_loss_full_lambda': Image loss for the full lambda path.
                - 'k_loss_inverse': Loss for the inverse path in k-space.
                - 'image_loss_inverse_lambda': Image loss for the inverse lambda path.
                - 'image_loss_inverse_full': Image loss for the inverse full path.
                - 'image_loss: Image loss for the lambda path vs fully sampled.'
        """
        if self.guided_single_contrast:
            return self._calculate_loss_guided(
                estimates,
                undersampled_k,
                fully_sampled,
                input_mask,
                dc_mask,
                label,
                contrast_ids=contrast_ids,
            )

        # Canonicalize tensors for stable per-contrast logging when the dataset permutes contrast axis.
        # This does not change the loss value (it only reorders channels), but it makes logged keys
        # like k_loss_*_{t1,t2,...} meaningful.
        if contrast_perm is not None:
            undersampled_k = self._unpermute_contrast_axis(undersampled_k, contrast_perm)
            fully_sampled = self._unpermute_contrast_axis(fully_sampled, contrast_perm)
            input_mask = self._unpermute_contrast_axis(input_mask, contrast_perm)
            dc_mask = self._unpermute_contrast_axis(dc_mask, contrast_perm)

            for key in ("lambda_path", "full_path", "inverse_path"):
                if key in estimates and (estimates[key] is not None):
                    estimates[key] = self._unpermute_contrast_axis(estimates[key], contrast_perm)

        # estimated k-space from different paths
        lambda_esitmate = estimates["lambda_path"]
        full_estimate = estimates["full_path"]
        inverse_estimate = estimates["inverse_path"]

        lam_full_scaling, lam_inv_scaling, inv_full_scaling = (
            self.get_image_space_scaling_factors()
        )

        loss_dict = {}

        if self.use_superviesd_image_loss:
            target_img = k_to_img(fully_sampled, coil_dim=2)
            lambda_img = k_to_img(lambda_esitmate, coil_dim=2)
            ssim_val = ssim(
                target_img, 
                lambda_img, 
                data_range=(target_img.min().item(), target_img.max().item()),
            )
            assert isinstance(ssim_val, torch.Tensor)
            loss_dict["image_loss"] = 1 - ssim_val

        # calculate the loss of the lambda estimation

        k_losses = self.calculate_k_loss(
            lambda_esitmate, 
            fully_sampled, 
            dc_mask,
            self.lambda_loss_scaling, 
            "lambda"
        )

        for contrast, value in k_losses.items():
            self.log(f"{label}/{contrast}", value)

        loss_dict["k_loss"] = sum([values for values in k_losses.values()]) / len(k_losses)

        # calculate full lambda image loss pathway
        if full_estimate is not None:
            #k_losses_full = self.calculate_k_loss(
            #    full_estimate, 
            #    undersampled_k, 
            #    undersampled_k != 0, 
            #    1,
            #    "full"
            #)
            #loss_dict["k_loss_full"] = sum([values for values in k_losses_full.values()]) / len(k_losses_full)


            loss_dict["image_loss_full_lambda"] = self.compute_image_loss(
                full_estimate,
                lambda_esitmate,
                undersampled_k,
            )
            self.log(
                f"{label}/unscaled_full_lambda", loss_dict["image_loss_full_lambda"]
            )
            loss_dict["image_loss_full_lambda"] *= lam_full_scaling

        # calculate inverse lambda image and k-space loss pathway
        if inverse_estimate is not None:
            # k space loss
            inverse_k_losses = self.calculate_inverse_k_loss(
                input_mask, dc_mask, inverse_estimate, undersampled_k
            )
            for key, value in inverse_k_losses.items():
                self.log(f"{label}/{key}", value)
            loss_dict["k_loss_inverse"] = sum([values for values in inverse_k_losses.values()]) / len(inverse_k_losses) # image space loss

            loss_dict["image_loss_inverse_lambda"] = self.compute_image_loss(
                lambda_esitmate,
                inverse_estimate,
                undersampled_k,
            )
            self.log(
                "train/unscaled_inverse_lambda", loss_dict["image_loss_inverse_lambda"]
            )
            loss_dict["image_loss_inverse_lambda"] *= lam_inv_scaling

        # calculate inverse full_inverse image pathway
        if (inverse_estimate is not None) and (full_estimate is not None):
            loss_dict["image_loss_inverse_full"] = self.compute_image_loss(
                full_estimate,
                inverse_estimate,
                undersampled_k,
            )
            self.log(
                "train/unscaled_inverse_full", loss_dict["image_loss_inverse_full"]
            )
            loss_dict["image_loss_inverse_full"] *= inv_full_scaling

        return loss_dict

    def _calculate_loss_guided(
        self,
        estimates,
        undersampled_k,
        fully_sampled,
        input_mask,
        dc_mask,
        label: str,
        contrast_ids=None,
    ):
        """Guided single-contrast supervision.

        Assumes the dataset packs the target contrast into channel 0, with other contrasts
        provided as additional input channels. We supervise only channel 0.
        """
        # Slice to target (channel 0) for all losses.
        undersampled_k_t = undersampled_k[:, 0:1, ...]
        fully_sampled_t = fully_sampled[:, 0:1, ...]
        input_mask_t = input_mask[:, 0:1, ...]
        dc_mask_t = dc_mask[:, 0:1, ...]

        lambda_estimate = estimates["lambda_path"]
        full_estimate = estimates["full_path"]
        inverse_estimate = estimates["inverse_path"]
        lambda_estimate_t = lambda_estimate[:, 0:1, ...]
        full_estimate_t = full_estimate[:, 0:1, ...] if full_estimate is not None else None
        inverse_estimate_t = inverse_estimate[:, 0:1, ...] if inverse_estimate is not None else None

        lam_full_scaling, lam_inv_scaling, inv_full_scaling = (
            self.get_image_space_scaling_factors()
        )

        loss_dict = {}

        if self.use_superviesd_image_loss:
            target_img = k_to_img(fully_sampled_t, coil_dim=2)
            lambda_img = k_to_img(lambda_estimate_t, coil_dim=2)
            ssim_val = ssim(
                target_img,
                lambda_img,
                data_range=(target_img.min().item(), target_img.max().item()),
            )
            assert isinstance(ssim_val, torch.Tensor)
            loss_dict["image_loss"] = 1 - ssim_val

        # K-space loss on target only (channel 0)
        k_loss_val = self.k_space_loss(
            torch.view_as_real(fully_sampled_t[:, 0, ...] * dc_mask_t[:, 0, ...]),
            torch.view_as_real(lambda_estimate_t[:, 0, ...] * dc_mask_t[:, 0, ...]),
        )
        assert isinstance(k_loss_val, torch.Tensor)
        loss_dict["k_loss"] = k_loss_val * self.lambda_loss_scaling

        # Optional per-contrast logging (aggregated by true target id)
        if contrast_ids is not None:
            if contrast_ids.ndim > 1:
                contrast_ids = contrast_ids.view(-1)
            for contrast_id in torch.unique(contrast_ids).tolist():
                contrast_id = int(contrast_id)
                if contrast_id < 0 or contrast_id >= len(self.metric_contrast_order):
                    continue
                mask_ids = (contrast_ids == contrast_id)
                if not mask_ids.any():
                    continue

                gt = fully_sampled_t[mask_ids]
                est = lambda_estimate_t[mask_ids]
                m = dc_mask_t[mask_ids]
                k_loss_c = self.k_space_loss(
                    torch.view_as_real(gt[:, 0, ...] * m[:, 0, ...]),
                    torch.view_as_real(est[:, 0, ...] * m[:, 0, ...]),
                )
                contrast_label = self.metric_contrast_order[contrast_id]
                self.log(
                    f"{label}/k_loss_lambda_{contrast_label}",
                    k_loss_c * self.lambda_loss_scaling,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=int(mask_ids.sum().item()),
                )

        # Full pathway image loss (target only)
        if full_estimate_t is not None:
            loss_dict["image_loss_full_lambda"] = self.compute_image_loss(
                full_estimate_t,
                lambda_estimate_t,
                undersampled_k_t,
            )
            self.log(
                f"{label}/unscaled_full_lambda",
                loss_dict["image_loss_full_lambda"],
                on_epoch=True,
            )
            loss_dict["image_loss_full_lambda"] *= lam_full_scaling

        # Inverse pathway losses (target only)
        if inverse_estimate_t is not None:
            _, lambda_k_wo_acs = TriplePathway.create_inverted_masks(
                input_mask_t,
                dc_mask_t,
                self.recon_model.dual_domain_config.pass_through_size,
                self.recon_model.dual_domain_config.pass_all_lines,
            )

            inv_k_loss_val = self.k_space_loss(
                torch.view_as_real(undersampled_k_t[:, 0, ...] * lambda_k_wo_acs[:, 0, ...]),
                torch.view_as_real(inverse_estimate_t[:, 0, ...] * lambda_k_wo_acs[:, 0, ...]),
            )
            loss_dict["k_loss_inverse"] = inv_k_loss_val * (1 - self.lambda_loss_scaling)

            if contrast_ids is not None:
                if contrast_ids.ndim > 1:
                    contrast_ids = contrast_ids.view(-1)
                for contrast_id in torch.unique(contrast_ids).tolist():
                    contrast_id = int(contrast_id)
                    if contrast_id < 0 or contrast_id >= len(self.metric_contrast_order):
                        continue
                    mask_ids = (contrast_ids == contrast_id)
                    if not mask_ids.any():
                        continue
                    gt = undersampled_k_t[mask_ids]
                    est = inverse_estimate_t[mask_ids]
                    m = lambda_k_wo_acs[mask_ids]
                    inv_k_c = self.k_space_loss(
                        torch.view_as_real(gt[:, 0, ...] * m[:, 0, ...]),
                        torch.view_as_real(est[:, 0, ...] * m[:, 0, ...]),
                    )
                    contrast_label = self.metric_contrast_order[contrast_id]
                    self.log(
                        f"{label}/k_loss_inverse_{contrast_label}",
                        inv_k_c * (1 - self.lambda_loss_scaling),
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=int(mask_ids.sum().item()),
                    )

            loss_dict["image_loss_inverse_lambda"] = self.compute_image_loss(
                lambda_estimate_t,
                inverse_estimate_t,
                undersampled_k_t,
            )
            self.log(
                f"{label}/unscaled_inverse_lambda",
                loss_dict["image_loss_inverse_lambda"],
                on_epoch=True,
            )
            loss_dict["image_loss_inverse_lambda"] *= lam_inv_scaling

        if (inverse_estimate_t is not None) and (full_estimate_t is not None):
            loss_dict["image_loss_inverse_full"] = self.compute_image_loss(
                full_estimate_t,
                inverse_estimate_t,
                undersampled_k_t,
            )
            self.log(
                f"{label}/unscaled_inverse_full",
                loss_dict["image_loss_inverse_full"],
                on_epoch=True,
            )
            loss_dict["image_loss_inverse_full"] *= inv_full_scaling

        return loss_dict

    def get_image_space_scaling_factors(self):
        if self.enable_warmup_training and self.current_epoch < 10:
            scaling_factor = 0
        else:
            scaling_factor = 1

        return (
            scaling_factor * self.image_scaling_lam_full,
            scaling_factor * self.image_scaling_lam_inv,
            scaling_factor * self.image_scaling_full_inv,
        )
