import torch
import numpy as np
from typing import Union 

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

import torch.nn.functional as F 
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.transforms.functional import gaussian_blur


from ml_recon.utils import root_sum_of_squares, ifft_2d_img
from ml_recon.utils.evaluation_functions import nmse, psnr



class plReconModel(pl.LightningModule):
    """This is a superclass for all reconstruction models. It tests the output 
    vs the ground truth using SSIM, PSNR, and NMSE. Handles multiple contrasts.
    Most recon networks here inhereit from this class. 
    """

    def __init__(
        self,
        contrast_order,
        is_mask_testing=True,
        mask_threshold: Union[dict, None] = None,
        metric_contrast_order=None,
    ):
        super().__init__()
        self.contrast_order = contrast_order
        self.metric_contrast_order = metric_contrast_order or contrast_order
        self.is_mask_testing = is_mask_testing
        self.contrast_psnr_masked = [[] for _ in self.metric_contrast_order]
        self.contrast_nmse_masked = [[] for _ in self.metric_contrast_order]
        self.contrast_ssim_masked = [[] for _ in self.metric_contrast_order]
        self.contrast_psnr = [[] for _ in self.metric_contrast_order]
        self.contrast_nmse = [[] for _ in self.metric_contrast_order]
        self.contrast_ssim = [[] for _ in self.metric_contrast_order]
        self.test_metrics_by_prefix: dict[str, dict[str, list[list[torch.Tensor]]]] = {}


    def _get_test_bucket(self, prefix: str):
        if not prefix:
            return None
        bucket = self.test_metrics_by_prefix.get(prefix)
        if bucket is None:
            bucket = {
                "nmse": [[] for _ in self.metric_contrast_order],
                "ssim": [[] for _ in self.metric_contrast_order],
                "psnr": [[] for _ in self.metric_contrast_order],
                "nmse_masked": [[] for _ in self.metric_contrast_order],
                "ssim_masked": [[] for _ in self.metric_contrast_order],
                "psnr_masked": [[] for _ in self.metric_contrast_order],
            }
            self.test_metrics_by_prefix[prefix] = bucket
        return bucket

    def _append_test_metrics(self, bucket, contrast_index, nmse_val, ssim_val, psnr_val, nmse_val_masked, ssim_val_masked, psnr_val_masked):
        if bucket is None:
            return
        bucket["nmse"][contrast_index].append(nmse_val.cpu())
        bucket["ssim"][contrast_index].append(ssim_val.cpu())
        bucket["psnr"][contrast_index].append(psnr_val.cpu())
        bucket["nmse_masked"][contrast_index].append(nmse_val_masked.cpu())
        bucket["ssim_masked"][contrast_index].append(ssim_val_masked.cpu())
        bucket["psnr_masked"][contrast_index].append(psnr_val_masked.cpu())


    def my_test_step(self, batch, batch_index, contrast_ids: Union[torch.Tensor, None] = None, prefix: str = ""):
        estimate_k, ground_truth_image = batch
        background_mask = self.get_image_background_mask(ground_truth_image)
        bucket = self._get_test_bucket(prefix)

        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)

        estimated_image /= scaling_factor
        ground_truth_image = ground_truth_image / scaling_factor

        estimated_image_masked = estimated_image * background_mask
        ground_truth_image_masked = ground_truth_image * background_mask

        # If we're in single-contrast-per-sample mode, contrast_ids indicates which *true* contrast
        # each sample belongs to, so we can aggregate metrics like multi-contrast runs.
        single_contrast_per_sample = (
            contrast_ids is not None
            and ground_truth_image.shape[1] == 1
            and len(self.metric_contrast_order) > 1
        )

        if single_contrast_per_sample:
            assert contrast_ids is not None
            if contrast_ids.ndim > 1:
                contrast_ids = contrast_ids.view(-1)
            for i in range(ground_truth_image.shape[0]):
                contrast_bucket = int(contrast_ids[i].item())
                contrast_bucket = max(0, min(contrast_bucket, len(self.metric_contrast_order) - 1))

                contrast_ground_truth = ground_truth_image[i, 0, :, :]
                contrast_ground_truth_masked = ground_truth_image_masked[i, 0, :, :]
                contrast_estimated = estimated_image[i, 0, :, :]
                contrast_estimated_masked = estimated_image_masked[i, 0, :, :]

                # reshape to proper shape for metrics
                contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                contrast_estimated = contrast_estimated[None, None, :, :]
                contrast_ground_truth_masked = contrast_ground_truth_masked[None, None, :, :]
                contrast_estimated_masked = contrast_estimated_masked[None, None, :, :]
                
                # masked metrics
                nmse_val = nmse(contrast_ground_truth, contrast_estimated)
                nmse_val_masked = nmse(contrast_ground_truth_masked, contrast_estimated_masked)

                # ssim metrics
                ssim_val_masked = ssim(
                    contrast_ground_truth_masked, 
                    contrast_estimated_masked, 
                    data_range=(contrast_ground_truth_masked.max().item()),
                    kernel_size=7
                    )
                ssim_val  = ssim(
                    contrast_ground_truth, 
                    contrast_estimated, 
                    data_range=(0, contrast_ground_truth.max().item()),
                    kernel_size=7
                    )

                assert isinstance(ssim_val, torch.Tensor)
                assert isinstance(ssim_val_masked, torch.Tensor)

                psnr_val_masked = psnr(contrast_ground_truth_masked, contrast_estimated_masked)
                psnr_val = psnr(contrast_ground_truth, contrast_estimated)
                
                self.contrast_ssim[contrast_bucket].append(ssim_val.cpu())
                self.contrast_psnr[contrast_bucket].append(psnr_val.cpu())
                self.contrast_nmse[contrast_bucket].append(nmse_val.cpu())

                self.contrast_ssim_masked[contrast_bucket].append(ssim_val_masked.cpu())
                self.contrast_psnr_masked[contrast_bucket].append(psnr_val_masked.cpu())
                self.contrast_nmse_masked[contrast_bucket].append(nmse_val_masked.cpu())

                self._append_test_metrics(
                    bucket,
                    contrast_bucket,
                    nmse_val,
                    ssim_val,
                    psnr_val,
                    nmse_val_masked,
                    ssim_val_masked,
                    psnr_val_masked,
                )

                contrast_label = self.metric_contrast_order[contrast_bucket]
                self.log(f"metrics/{prefix}nmse_{contrast_label}", nmse_val, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}ssim_{contrast_label}", ssim_val, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}psnr_{contrast_label}", psnr_val, sync_dist=True, on_step=True)

                self.log(f"metrics/{prefix}masked_nmse_{contrast_label}", nmse_val_masked, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}masked_ssim_{contrast_label}", ssim_val_masked, sync_dist=True, on_step=True)
                self.log(f"metrics/{prefix}masked_psnr_{contrast_label}", psnr_val_masked, sync_dist=True, on_step=True)

        else:
            for contrast_index in range(len(self.contrast_order)):
                for i in range(ground_truth_image.shape[0]):
                    # get a slice of a contrast
                    contrast_ground_truth = ground_truth_image[i, contrast_index, :, :]
                    contrast_ground_truth_masked = ground_truth_image_masked[i, contrast_index, :, :]
                    contrast_estimated = estimated_image[i, contrast_index, :, :]
                    contrast_estimated_masked = estimated_image_masked[i, contrast_index, :, :]

                    # reshape to proper shape for metrics
                    contrast_ground_truth = contrast_ground_truth[None, None, :, :]
                    contrast_estimated = contrast_estimated[None, None, :, :]
                    contrast_ground_truth_masked = contrast_ground_truth_masked[None, None, :, :]
                    contrast_estimated_masked = contrast_estimated_masked[None, None, :, :]

                    # masked metrics
                    nmse_val = nmse(contrast_ground_truth, contrast_estimated)
                    nmse_val_masked = nmse(contrast_ground_truth_masked, contrast_estimated_masked)

                    # ssim metrics
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
                    self.contrast_ssim[contrast_index].append(ssim_val.cpu())
                    self.contrast_psnr[contrast_index].append(psnr_val.cpu())
                    self.contrast_nmse[contrast_index].append(nmse_val.cpu())

                    self.contrast_ssim_masked[contrast_index].append(ssim_val_masked.cpu())
                    self.contrast_psnr_masked[contrast_index].append(psnr_val_masked.cpu())
                    self.contrast_nmse_masked[contrast_index].append(nmse_val_masked.cpu())

                    self._append_test_metrics(
                        bucket,
                        contrast_index,
                        nmse_val,
                        ssim_val,
                        psnr_val,
                        nmse_val_masked,
                        ssim_val_masked,
                        psnr_val_masked,
                    )

                    self.log(f"metrics/{prefix}nmse_{self.contrast_order[contrast_index]}", nmse_val, sync_dist=True, on_step=True)
                    self.log(f"metrics/{prefix}ssim_{self.contrast_order[contrast_index]}", ssim_val, sync_dist=True, on_step=True)
                    self.log(f"metrics/{prefix}psnr_{self.contrast_order[contrast_index]}", psnr_val, sync_dist=True, on_step=True)

                    self.log(f"metrics/{prefix}masked_nmse_{self.contrast_order[contrast_index]}", nmse_val_masked, sync_dist=True, on_step=True)
                    self.log(f"metrics/{prefix}masked_ssim_{self.contrast_order[contrast_index]}", ssim_val_masked, sync_dist=True, on_step=True)
                    self.log(f"metrics/{prefix}masked_psnr_{self.contrast_order[contrast_index]}", psnr_val_masked, sync_dist=True, on_step=True)

        
        return {
            'loss': 0,
            'estimate_image': estimated_image,
            'ground_truth_image': ground_truth_image,
            'mask': background_mask
        }

    def on_test_end(self):
        for contrast_index, contrast_label in enumerate(self.metric_contrast_order):
            psnr_array = np.array(self.contrast_psnr[contrast_index])
            ssim_array = np.array(self.contrast_ssim[contrast_index])
            nmse_array = np.array(self.contrast_nmse[contrast_index])
            nmse_masked_array = np.array(self.contrast_nmse_masked[contrast_index])
            ssim_masked_array = np.array(self.contrast_ssim_masked[contrast_index])
            psnr_masked_array = np.array(self.contrast_psnr_masked[contrast_index])
            print(f"metrics_mine/nmse_{contrast_label}", nmse_array.mean())
            print(f"metrics_mine/ssim_{contrast_label}", ssim_array.mean())
            print(f"metrics_mine/psnr_{contrast_label}", psnr_array.mean())

            print(f"metrics_mine/masked_nmse_{contrast_label}", nmse_masked_array.mean())
            print(f"metrics_mine/masked_ssim_{contrast_label}", ssim_masked_array.mean())
            print(f"metrics_mine/masked_psnr_{contrast_label}", psnr_masked_array.mean())

            print(f"metrics_mine/nmse_std_{contrast_label}", nmse_array.std())
            print(f"metrics_mine/ssim_std_{contrast_label}", ssim_array.std())
            print(f"metrics_mine/psnr_std_{contrast_label}", psnr_array.std())

            print(f"metrics_mine/masked_nmse_std_{contrast_label}", nmse_masked_array.std())
            print(f"metrics_mine/masked_ssim_std_{contrast_label}", ssim_masked_array.std())
            print(f"metrics_mine/masked_psnr_std_{contrast_label}", psnr_masked_array.std())

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({f"metrics/nmse_{contrast_label}_std": nmse_array.std()})
                self.logger.experiment.log({f"metrics/ssim_{contrast_label}_std": ssim_array.std()})
                self.logger.experiment.log({f"metrics/psnr_{contrast_label}_std": psnr_array.std()})
                self.logger.experiment.log({f"metrics/nmse_masked_{contrast_label}_std": nmse_masked_array.std()})
                self.logger.experiment.log({f"metrics/ssim_masked_{contrast_label}_std": ssim_masked_array.std()})
                self.logger.experiment.log({f"metrics/psnr_masked_{contrast_label}_std": psnr_masked_array.std()})

        if self.test_metrics_by_prefix:
            for prefix in sorted(self.test_metrics_by_prefix.keys()):
                bucket = self.test_metrics_by_prefix[prefix]
                for contrast_index, contrast_label in enumerate(self.metric_contrast_order):
                    psnr_array = np.array(bucket["psnr"][contrast_index])
                    ssim_array = np.array(bucket["ssim"][contrast_index])
                    nmse_array = np.array(bucket["nmse"][contrast_index])
                    nmse_masked_array = np.array(bucket["nmse_masked"][contrast_index])
                    ssim_masked_array = np.array(bucket["ssim_masked"][contrast_index])
                    psnr_masked_array = np.array(bucket["psnr_masked"][contrast_index])

                    print(f"metrics_mine/{prefix}nmse_{contrast_label}", nmse_array.mean())
                    print(f"metrics_mine/{prefix}ssim_{contrast_label}", ssim_array.mean())
                    print(f"metrics_mine/{prefix}psnr_{contrast_label}", psnr_array.mean())

                    print(f"metrics_mine/{prefix}masked_nmse_{contrast_label}", nmse_masked_array.mean())
                    print(f"metrics_mine/{prefix}masked_ssim_{contrast_label}", ssim_masked_array.mean())
                    print(f"metrics_mine/{prefix}masked_psnr_{contrast_label}", psnr_masked_array.mean())

                    print(f"metrics_mine/{prefix}nmse_std_{contrast_label}", nmse_array.std())
                    print(f"metrics_mine/{prefix}ssim_std_{contrast_label}", ssim_array.std())
                    print(f"metrics_mine/{prefix}psnr_std_{contrast_label}", psnr_array.std())

                    print(f"metrics_mine/{prefix}masked_nmse_std_{contrast_label}", nmse_masked_array.std())
                    print(f"metrics_mine/{prefix}masked_ssim_std_{contrast_label}", ssim_masked_array.std())
                    print(f"metrics_mine/{prefix}masked_psnr_std_{contrast_label}", psnr_masked_array.std())

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log({f"metrics/{prefix}nmse_{contrast_label}": nmse_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}ssim_{contrast_label}": ssim_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}psnr_{contrast_label}": psnr_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}nmse_masked_{contrast_label}": nmse_masked_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}ssim_masked_{contrast_label}": ssim_masked_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}psnr_masked_{contrast_label}": psnr_masked_array.mean()})
                        self.logger.experiment.log({f"metrics/{prefix}nmse_{contrast_label}_std": nmse_array.std()})
                        self.logger.experiment.log({f"metrics/{prefix}ssim_{contrast_label}_std": ssim_array.std()})
                        self.logger.experiment.log({f"metrics/{prefix}psnr_{contrast_label}_std": psnr_array.std()})
                        self.logger.experiment.log({f"metrics/{prefix}nmse_masked_{contrast_label}_std": nmse_masked_array.std()})
                        self.logger.experiment.log({f"metrics/{prefix}ssim_masked_{contrast_label}_std": ssim_masked_array.std()})
                        self.logger.experiment.log({f"metrics/{prefix}psnr_masked_{contrast_label}_std": psnr_masked_array.std()})


    def my_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        estimate_k, ground_truth_image, mask = batch
        estimated_image = root_sum_of_squares(ifft_2d_img(estimate_k), coil_dim=2)

        scaling_factor = ground_truth_image.amax((-1, -2), keepdim=True)
        image_background_mask = self.get_image_background_mask(ground_truth_image)

        estimated_image /= scaling_factor
        ground_truth_image /= scaling_factor

        estimated_image_masked = estimated_image * image_background_mask
        ground_truth_image_masked = ground_truth_image * image_background_mask

        difference_image = (ground_truth_image - estimated_image).abs()
        difference_image_masked = (ground_truth_image_masked - estimated_image_masked).abs()
        
        estimated_image = estimated_image[0].clamp(0, 1)
        ground_truth_image = ground_truth_image[0]
        difference_image = (difference_image[0]*10).clamp(0, 1)

        estimated_image_masked = estimated_image_masked[0].clamp(0, 1)
        ground_truth_image_masked = ground_truth_image_masked[0]
        difference_image_masked = (difference_image_masked[0]*10).clamp(0, 1)
        if isinstance(self.logger, WandbLogger):
            self.plot_test_images(
                ground_truth_image_masked, 
                estimated_image_masked, 
                difference_image_masked,
                label='masked',
                )
            self.plot_test_images(
                ground_truth_image, 
                estimated_image, 
                difference_image,
                label='unmasked',
                )
            self.logger.log_image(f'test/undersampling_mask', self.convert_image_for_plotting(mask[0, :, 0]))

    def plot_test_images(
        self, 
        ground_truth_image, 
        estimated_image, 
        difference_image,
        label='',
    ):
        wandb_logger = self.logger
        assert isinstance(wandb_logger, WandbLogger)

        wandb_logger.log_image(f'test/{label}_recon', self.convert_image_for_plotting(estimated_image))
        wandb_logger.log_image(f'test/{label}_target', self.convert_image_for_plotting(ground_truth_image))
        wandb_logger.log_image(f'test/{label}_diff', self.convert_image_for_plotting(difference_image))


    def get_image_background_mask(self, ground_truth_image):
        # ground truth image shape b, con, h, w
        if not self.is_mask_testing:
            return torch.ones_like(ground_truth_image)


        # gaussian blur image for better masking (blurring improves SNR)
        ground_truth_blurred = gaussian_blur(ground_truth_image, kernel_size=15, sigma=10.0) # type: ignore

        # get noise
        noise = ground_truth_blurred[..., :20, :20]
        # take the max value and scale up a bit
        mask_threshold = noise.amax((-1, -2)) * 1.20

        # same shape as image
        mask_threshold = mask_threshold.unsqueeze(-1).unsqueeze(-1)

        # get mask
        image_background_mask = ground_truth_blurred > mask_threshold 

        mask =  self.dialate_mask(image_background_mask)

        # If there are any masks that are all zero, set to all 1s
        all_zero_masks_indecies = (~mask).all(dim=-1).all(dim=-1)
        # check if there are zero mask indexes
        if all_zero_masks_indecies.any():
            mask[all_zero_masks_indecies, :, :] = True

        return mask



    def dialate_mask(self, mask, kernel_size=3):

        b, contrast, h, w = mask.shape
        mask = mask.view(b*contrast, h, w)
        dialed_mask = self.dilate(mask.to(torch.float32), kernel_size)
        return dialed_mask.to(torch.bool).view(b, contrast, h, w)


    def dilate(self, image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Applies morphological dilation to a 2D image tensor.

        Args:
            image (torch.Tensor): Input tensor of shape (B, H, W).
            kernel_size (int): Size of the square dilation kernel. Should be an odd number.

        Returns:
            torch.Tensor: Dilated tensor of shape (B, H, W).
        """
        if image.dim() != 3:
            raise ValueError("Input tensor must have shape (B, H, W)")

        # Convert (B, H, W) -> (B, 1, H, W) for compatibility with max_pool2d
        image = image.unsqueeze(1)

        # Apply max pooling to simulate dilation
        dilated = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # Remove extra channel dimension
        return dilated.squeeze(1)


    def convert_image_for_plotting(self, image: torch.Tensor):
        contrasts = image.shape[0]
        return np.split(image.unsqueeze(1).cpu().numpy(), contrasts, 0)
