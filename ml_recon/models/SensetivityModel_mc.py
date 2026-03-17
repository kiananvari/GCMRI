import einops
import torch
import math
from contextlib import nullcontext

import torch.nn as nn
from ml_recon.utils import ifft_2d_img
from .UNet import Unet
from .PromptUNet import PromptUNet, PromptUNetConfig
from .PromptMRUNet import PromptMRUNet, PromptMRUNetConfig
from ml_recon.utils import real_to_complex, complex_to_real, root_sum_of_squares

from typing import Tuple, Literal

class SensetivityModel_mc(nn.Module):
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            chans: int, 
            model: Literal['unet', 'prompt_unet', 'promptmr_unet'] = 'unet',
            conv_after_upsample: bool = False,
            upsample_method: Literal['conv', 'max', 'bilinear'] = 'conv'
            ):

        """Module used to estimate sensetivity maps based on masked k-space center

        Args:
            in_chans (int): _description_
            out_chans (int): _description_
            chans (int): Number of convolutional channels for U-Net estimator
            conv_after_upsample (bool): boolean flag for addition of 3x3 convolution after upsampling
            upsample_method (str): upsampling method for U-Net ('bilinear', 'max', 'conv')
        """
        super().__init__()
        if model == 'unet':
            self.model = Unet(
                in_chans,
                out_chans,
                chans=chans,
                conv_after_upsample=conv_after_upsample,
                upsample_method=upsample_method,
            )
        elif model == 'prompt_unet':
            self.model = PromptUNet(
                in_chan=in_chans,
                out_chan=out_chans,
                cfg=PromptUNetConfig(dim=chans),
            )
        elif model == 'promptmr_unet':
            self.model = PromptMRUNet(
                in_chan=in_chans,
                out_chan=out_chans,
                cfg=PromptMRUNetConfig(n_feat0=chans),
            )
        else:
            raise ValueError(f"Unknown sensitivity backbone: {model!r}")

    # recieve coil maps as [B, contrast, channels, H, W]
    def forward(self, images, mask):
        images = self.mask_center(images, mask)
        # Estimate coil sensitivities from the first contrast channel.
        # The dataset permutation logic already ensures excluded contrasts do not occupy index 0.
        images = images[:, [0], :, :, :]

        # Sensitivity estimation is prone to AMP/FP16 instabilities.
        # Force complex64 + float32 math and disable autocast for this submodule.
        if isinstance(images, torch.Tensor) and images.is_complex() and images.dtype != torch.complex64:
            images = images.to(torch.complex64)

        images = ifft_2d_img(images, axes=[-1, -2])
        assert isinstance(images, torch.Tensor)

        number_of_coils = images.shape[2]
        num_contrasts = images.shape[1]

        images = einops.rearrange(images, 'b contrast c h w -> (b contrast c) 1 h w')
        assert isinstance(images, torch.Tensor)

        # convert to real numbers [b * contrast * coils, cmplx, h, w]
        images = complex_to_real(images).to(torch.float32)

        autocast_ctx = (
            torch.autocast(device_type=images.device.type, enabled=False)
            if hasattr(torch, 'autocast')
            else nullcontext()
        )
        with autocast_ctx:
            # norm
            images, mean, std = self.norm(images)
            assert not torch.isnan(images).any()
            # pass through model
            images = self.model(images)
            assert not torch.isnan(images).any()
            # unnorm
            images = self.unnorm(images, mean, std)
        # convert back to complex
        images = real_to_complex(images)
        # rearange back to original format
        images = einops.rearrange(images, '(b contrast c) 1 h w -> b contrast c h w', c=number_of_coils, contrast=num_contrasts)
        # rss to normalize sense maps
        rss_norm = root_sum_of_squares(images, coil_dim=2).unsqueeze(2).clamp_min(1e-6)
        images = images / rss_norm
        return images

    def mask_center(self, coil_k_spaces, mask):
        # I did some strange things here. Before, I tried to find the largest 2d box 
        # that was continuously contained in sampled k-space. However, that added some extra 
        # bugs in self-suprvised training as the center box size could change depending
        # on the sets. I have just hard coded this for now, but could be interesting to 
        # test different coil estimation methods and masking methods. 
        # 
        # There are probably some interesting ideas for deep learning coil estimatation methods. 
        h, w = coil_k_spaces.shape[-2:] 
        center_mask = torch.zeros((h, w), device=coil_k_spaces.device)
        center_mask[h//2-5:h//2+5, w//2-5:w//2+5] = 1

        return coil_k_spaces * center_mask
        

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Instance norm.
        # Compute statistics in float32 for numerical stability under AMP/FP16.
        b, c, h, w = x.shape
        if c % 2 != 0:
            raise ValueError(f"Expected even channel dim (real/imag pairs), got c={c}")

        x_reshaped = x.reshape(b, 2, (c // 2) * h * w)
        x_stats = x_reshaped.to(torch.float32)
        mean = x_stats.mean(dim=2).reshape(b, 2, 1, 1)
        var = x_stats.var(dim=2, unbiased=False).reshape(b, 2, 1, 1)
        std = torch.sqrt(var + 1e-6)

        mean = mean.to(dtype=x.dtype)
        std = std.to(dtype=x.dtype)
        x = x.reshape(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean
