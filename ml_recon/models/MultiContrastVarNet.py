from typing import Tuple, Literal, Optional, Union, cast
from functools import partial
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import torch.nn as nn
import torch

from .UNet import Unet
from .PromptUNet import PromptUNet, PromptUNetConfig
from .PromptMRUNet import PromptMRUNet, PromptMRUNetConfig
from .SensetivityModel_mc import SensetivityModel_mc
from ml_recon.utils import fft_2d_img, ifft_2d_img, complex_to_real, real_to_complex


def _finite_or_raise(name: str, x: Union[torch.Tensor, NDArray]) -> None:
    if isinstance(x, torch.Tensor):
        if torch.isfinite(x).all():
            return

        finite_mask = torch.isfinite(x)
        num_bad = int((~finite_mask).sum().detach().cpu())
        with torch.no_grad():
            finite_vals = x[finite_mask]
            if finite_vals.numel() == 0:
                stats = "no finite values"
            else:
                stats = (
                    f"min={finite_vals.min().item():.3e}, max={finite_vals.max().item():.3e}, "
                    f"mean={finite_vals.mean().item():.3e}, std={finite_vals.std().item():.3e}"
                )
        raise FloatingPointError(
            f"Non-finite values in {name}: {num_bad} / {x.numel()} are NaN/Inf; shape={tuple(x.shape)}; {stats}"
        )

    x_arr = np.asarray(x)
    if np.isfinite(x_arr).all():
        return

    finite_mask = np.isfinite(x_arr)
    num_bad = int((~finite_mask).sum())
    finite_vals = x_arr[finite_mask]
    if finite_vals.size == 0:
        stats = "no finite values"
    else:
        stats = (
            f"min={finite_vals.min():.3e}, max={finite_vals.max():.3e}, "
            f"mean={finite_vals.mean():.3e}, std={finite_vals.std():.3e}"
        )
    raise FloatingPointError(
        f"Non-finite values in {name}: {num_bad} / {x_arr.size} are NaN/Inf; shape={x_arr.shape}; {stats}"
    )

@dataclass
class VarnetConfig:
    contrast_order: list
    metric_contrast_order: Optional[list] = None
    model: Literal['unet', 'prompt_unet', 'promptmr_unet'] = 'unet'
    cascades: int = 5
    sense_chans: int = 8
    channels: int = 18
    dropout: float = 0
    depth: int = 4
    upsample_method: Literal['conv', 'bilinear', 'max'] = 'conv'
    conv_after_upsample: bool = False

    # PromptMRUNet-only: if True, use a baseline UNet-like doubling channel schedule
    # for PromptMRUNet's encoder feature widths when cfg.feature_dim is not provided.
    promptmr_feature_dim_like_unet: bool = False

    # PromptMRUNet-only: ablation for the first feature extraction.
    # - True: use PromptMRUNet's contrast-aware stem (per-contrast + optional cross-contrast mixing).
    # - False: use baseline UNet-like stem (no explicit contrast axis).
    promptmr_contrast_aware_stem: bool = True

    # PromptMRUNet-only: CAB ablation.
    # - True: use PromptMRUNet CAB blocks.
    # - False: use UNet-like conv blocks in down/up, and disable CABs in skip/bottleneck.
    promptmr_use_cabs: bool = True

    # PromptMRUNet-only: normalization ablation.
    # - True: use InstanceNorm2d (baseline behavior).
    # - False: disable InstanceNorm layers inside PromptMRUNet blocks.
    promptmr_use_instancenorm: bool = True

    # PromptMRUNet-only: frequency-augmented CABs.
    # - True: enable frequency residual inside CABs.
    # - False: standard CABs (default).
    promptmr_use_freq_cab: bool = False

    # PromptMRUNet-only: SpectraMR-style frequency branch before each decoder stage.
    promptmr_use_fremodule: bool = False

    # PromptMRUNet-only: prompt injection ablation.
    # - True: enable prompt injection.
    # - False: disable prompts and use UNet-like decoder blocks.
    promptmr_use_prompt_injection: bool = True

    # PromptMRUNet-only: cross-contrast attention strength (stem).
    # You asked to keep other stem-strength knobs as code defaults.
    promptmr_contrast_attn_heads: int = 1
    promptmr_contrast_attn_gate_init: float = 0.0

    # PromptMRUNet-only: spatial+frequency stem mixing (single attention).
    promptmr_stem_use_freq_mix: bool = False
    # If True, bypass the stem_mix gate and always apply mixing at full strength.
    promptmr_stem_mix_always_on: bool = False
    promptmr_stem_mix_freq_mode: Literal['low', 'high', 'all'] = 'low'

    # PromptMRUNet-only: if True, do NOT share per-contrast stem conv weights.
    # This allocates one per-contrast conv block per contrast index.
    promptmr_stem_separate_per_contrast_conv: bool = False

    # PromptMRUNet-only: PromptMR+ extras
    promptmr_enable_buffer: bool = False
    promptmr_enable_history: bool = False


class MultiContrastVarNet(nn.Module):
    def __init__(self, config: VarnetConfig):
        super().__init__()
        contrasts = len(config.contrast_order)
        promptmr_enable_buffer = bool(getattr(config, 'promptmr_enable_buffer', False))
        promptmr_enable_history = bool(getattr(config, 'promptmr_enable_history', False))
        promptmr_n_buffer = 4 if promptmr_enable_buffer else 0
        promptmr_n_history = 11 if promptmr_enable_history else 0
        if config.model == 'unet':
            model_backbone = partial(
                Unet,
                in_chan=contrasts * 2,
                out_chan=contrasts * 2,
                chans=config.channels,
                drop_prob=config.dropout,
                depth=config.depth,
                upsample_method=config.upsample_method,
                conv_after_upsample=config.conv_after_upsample,
            )
        elif config.model == 'prompt_unet':
            model_backbone = partial(
                PromptUNet,
                in_chan=contrasts * 2,
                out_chan=contrasts * 2,
                cfg=PromptUNetConfig(dim=config.channels),
            )
        elif config.model == 'promptmr_unet':
            model_backbone = partial(
                PromptMRUNet,
                in_chan=contrasts * 2,
                out_chan=contrasts * 2,
                cfg=PromptMRUNetConfig(
                    n_feat0=config.channels,
                    depth=config.depth,
                    drop_prob=config.dropout,
                    feature_dim_like_unet=bool(getattr(config, 'promptmr_feature_dim_like_unet', False)),
                    contrast_aware_stem=bool(getattr(config, 'promptmr_contrast_aware_stem', True)),
                    use_cabs=bool(getattr(config, 'promptmr_use_cabs', True)),
                    use_instancenorm=bool(getattr(config, 'promptmr_use_instancenorm', True)),
                    use_freq_cab=bool(getattr(config, 'promptmr_use_freq_cab', False)),
                    use_fremodule=bool(getattr(config, 'promptmr_use_fremodule', False)),
                    use_prompt_injection=bool(getattr(config, 'promptmr_use_prompt_injection', True)),
                    contrast_attn_heads=int(getattr(config, 'promptmr_contrast_attn_heads', 1)),
                    contrast_attn_gate_init=float(getattr(config, 'promptmr_contrast_attn_gate_init', 0.0)),
                    stem_use_freq_mix=bool(getattr(config, 'promptmr_stem_use_freq_mix', False)),
                    stem_mix_always_on=bool(getattr(config, 'promptmr_stem_mix_always_on', False)),
                    stem_mix_freq_mode=cast(Literal['low', 'high', 'all'], getattr(config, 'promptmr_stem_mix_freq_mode', 'low')),
                    stem_separate_per_contrast_conv=bool(
                        getattr(config, 'promptmr_stem_separate_per_contrast_conv', False)
                    ),
                    enable_buffer=promptmr_enable_buffer,
                    n_buffer=promptmr_n_buffer,
                    enable_history=promptmr_enable_history,
                    n_history=promptmr_n_history,
                    upsample_method=config.upsample_method,
                    conv_after_upsample=config.conv_after_upsample,
                ),
            )
        else:
            raise ValueError(f"Unknown model backbone: {config.model!r}")




        # module cascades
        self.cascades = nn.ModuleList(
            [
                VarnetBlock(
                    model_backbone(),
                    enable_buffer=promptmr_enable_buffer and config.model == 'promptmr_unet',
                    n_buffer=promptmr_n_buffer,
                    enable_history=promptmr_enable_history and config.model == 'promptmr_unet',
                )
                for _ in range(config.cascades)
            ]
        )

        # model to estimate sensetivities
        # Use simple UNet for sensitivity estimation for stability.
        sens_backbone = 'unet'
        self.sens_model = SensetivityModel_mc(
            2, 
            2, 
            chans=config.sense_chans, 
            model=sens_backbone,
            upsample_method=config.upsample_method, 
            conv_after_upsample=config.conv_after_upsample
            )
        # regularizer weight
        self.lambda_reg = nn.Parameter(torch.ones(config.cascades))

    # k-space sent in [B, C, H, W]
    def forward(self, reference_k, mask, zf_mask):
        # get sensetivity maps
        _finite_or_raise("reference_k", reference_k)
        _finite_or_raise("mask", mask)
        sense_maps = self.sens_model(reference_k, mask)

        _finite_or_raise("sense_maps", sense_maps)
        
        # current k_space 
        current_k = reference_k.clone()
        history_feat = None
        latent_feat = None
        for i, cascade in enumerate(self.cascades):
            current_k = current_k * zf_mask
            _finite_or_raise(f"current_k (pre-cascade {i})", current_k)
            # go through ith model cascade
            refined_k, history_feat, latent_feat = cascade(
                current_k,
                sense_maps,
                reference_k=reference_k,
                history_feat=history_feat,
                latent_feat=latent_feat,
            )
            _finite_or_raise("reference_k", reference_k)
            _finite_or_raise(f"refined_k (cascade {i})", refined_k)

            data_consistency = mask * (current_k - reference_k)
            # gradient descent step
            # Keep the learned DC step size in a sane range. In self-supervised
            # settings, instability here is the most common source of Inf/NaN.
            dc_weight = self.lambda_reg[i].clamp(min=0.0, max=10.0)
            current_k = current_k - (dc_weight * data_consistency) - refined_k
        return current_k


class VarnetBlock(nn.Module):
    def __init__(self, model: nn.Module, *, enable_buffer: bool = False, n_buffer: int = 0, enable_history: bool = False) -> None:
        super().__init__()
        self.model = model
        self.enable_buffer = bool(enable_buffer)
        self.n_buffer = int(n_buffer)
        self.enable_history = bool(enable_history)
        self._latent_proj: Optional[nn.Conv2d] = None

    # sensetivities data [B, contrast, C, H, W]
    def forward(self, k_space, sensetivities, *, reference_k=None, history_feat=None, latent_feat=None):
        _finite_or_raise("k_space", k_space)
        _finite_or_raise("sensetivities", sensetivities)
        # Reduce
        images = ifft_2d_img(k_space, axes=[-1, -2])

        _finite_or_raise("images (ifft)", images)

        # Images now [B, contrast, h, w] (complex)
        images = torch.sum(images * sensetivities.conj(), dim=2)

        _finite_or_raise("images (coil-combined)", images)

        # Images now [B, contrast * 2, h, w] (real)
        images = complex_to_real(images)
        images, mean, std = self.norm(images)
        _finite_or_raise("images (normed)", images)
        buffer = None
        if self.enable_buffer and self.n_buffer > 0:
            ref_k = reference_k if reference_k is not None else k_space
            ref_images = ifft_2d_img(ref_k, axes=[-1, -2])
            ref_images = torch.sum(ref_images * sensetivities.conj(), dim=2)
            ref_images = complex_to_real(ref_images)
            ref_images = (ref_images - mean) / std

            latent_proj = None
            if latent_feat is not None:
                if self._latent_proj is None:
                    self._latent_proj = nn.Conv2d(
                        latent_feat.shape[1],
                        images.shape[1],
                        kernel_size=1,
                        bias=False,
                    ).to(latent_feat.device)
                latent_proj = self._latent_proj(latent_feat)

            # PromptMR+ buffer order: [x_i, latent, x0, x_i - x0]
            buffer_parts = [images]
            if latent_proj is not None:
                buffer_parts.append(latent_proj)
            buffer_parts.append(ref_images)
            buffer_parts.append(images - ref_images)

            # Pad or trim to n_buffer chunks
            if len(buffer_parts) < self.n_buffer:
                buffer_parts.extend([ref_images] * (self.n_buffer - len(buffer_parts)))
            buffer = torch.cat(buffer_parts[: self.n_buffer], dim=1)

        if self.enable_history and hasattr(self.model, "forward_with_history"):
            images, history_feat = self.model.forward_with_history(images, history_feat=history_feat, buffer=buffer)
        elif self.enable_buffer and hasattr(self.model, "forward_with_history"):
            images, _ = self.model.forward_with_history(images, history_feat=None, buffer=buffer)
        else:
            images = self.model(images)

        _finite_or_raise("images (model out)", images)
        images = self.unnorm(images, mean, std)
        _finite_or_raise("images (unnormed)", images)
        images = real_to_complex(images)

        _finite_or_raise("images (complex)", images)

        # Expand
        images = sensetivities * images.unsqueeze(2)
        images = fft_2d_img(images, axes=[-1, -2])

        _finite_or_raise("images (fft)", images)

        new_latent = getattr(self.model, "_last_latent", None)
        return images, history_feat, new_latent

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # instance norm
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        x = (x - mean) / std
        return x, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        x = x * std + mean
        return x

