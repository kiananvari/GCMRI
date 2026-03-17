from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_channels: int, out_channels: int, kernel_size: int, *, bias: bool = False, stride: int = 1) -> nn.Conv2d:
    padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)


def _instancenorm(channels: int) -> nn.InstanceNorm2d:
    # Match the baseline UNet's normalization style (affine=False).
    return nn.InstanceNorm2d(channels, affine=False)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, *, bias: bool = False) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(self.avg_pool(x))
        return x * w


class CAB(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: int = 4,
        *,
        bias: bool = False,
        act: Optional[nn.Module] = None,
        no_use_ca: bool = False,
    ) -> None:
        super().__init__()
        if act is None:
            act = nn.PReLU()
        self.body = nn.Sequential(
            _conv(channels, channels, kernel_size, bias=bias),
            _instancenorm(channels),
            act,
            _conv(channels, channels, kernel_size, bias=bias),
            _instancenorm(channels),
        )
        self.ca = nn.Identity() if no_use_ca else _ChannelAttention(channels, reduction=reduction, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ca(self.body(x))
        return y + x


class PromptBlock(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        prompt_len: int,
        prompt_size: int,
        lin_dim: int,
        *,
        learnable_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size),
            requires_grad=learnable_prompt,
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))  # [B, C]
        weights = F.softmax(self.linear_layer(emb), dim=1)  # [B, prompt_len]

        bank = self.prompt_param.squeeze(0)  # [prompt_len, prompt_dim, ps, ps]
        prompt = torch.einsum("bl,lchw->bchw", weights, bank)
        prompt = F.interpolate(prompt, (h, w), mode="bilinear", align_corners=False)
        return self.conv3x3(prompt)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
        first_act: bool = False,
    ) -> None:
        super().__init__()
        if n_cab < 1:
            raise ValueError("n_cab must be >= 1")

        blocks: List[nn.Module] = []
        if first_act:
            blocks.append(CAB(in_channels, kernel_size, reduction, bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca))
            for _ in range(n_cab - 1):
                blocks.append(CAB(in_channels, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca))
        else:
            for _ in range(n_cab):
                blocks.append(CAB(in_channels, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca))

        self.encoder = nn.Sequential(*blocks)
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class SkipBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
    ) -> None:
        super().__init__()
        if n_cab <= 0:
            self.body = nn.Identity()
        else:
            self.body = nn.Sequential(
                *[
                    CAB(channels, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
                    for _ in range(n_cab)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        prompt_dim: int,
        n_cab: int,
        kernel_size: int,
        reduction: int,
        *,
        bias: bool,
        act: nn.Module,
        no_use_ca: bool,
        n_history: int = 0,
    ) -> None:
        super().__init__()
        self.n_history = n_history

        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim * (n_history + 1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca),
            )
        else:
            self.momentum = None

        self.fuse = nn.Sequential(
            *[
                CAB(in_dim + prompt_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
                for _ in range(max(1, n_cab))
            ]
        )
        self.reduce = nn.Conv2d(in_dim + prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.post = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(
        self,
        x: torch.Tensor,
        prompt: torch.Tensor,
        skip: torch.Tensor,
        history_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.n_history > 0 and self.momentum is not None:
            if history_feat is None:
                x = torch.tile(x, (1, self.n_history + 1, 1, 1))
            else:
                x = torch.cat([x, history_feat], dim=1)
            x = self.momentum(x)

        x = torch.cat([x, prompt], dim=1)
        x = self.fuse(x)
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.post(x)
        return x


@dataclass(frozen=True)
class PromptMRUNetConfig:
    # Top-level feature width
    n_feat0: int = 48

    # Widths per encoder level (3 levels)
    feature_dim: Tuple[int, int, int] | None = None

    # Prompt widths per decoder level (level1, level2, level3)
    prompt_dim: Tuple[int, int, int] | None = None

    # Prompt bank sizes
    len_prompt: Tuple[int, int, int] = (10, 10, 10)
    prompt_size: Tuple[int, int, int] = (64, 32, 16)

    # CAB counts
    n_enc_cab: Tuple[int, int, int] = (1, 1, 2)
    n_dec_cab: Tuple[int, int, int] = (1, 1, 2)
    n_skip_cab: Tuple[int, int, int] = (1, 1, 2)
    n_bottleneck_cab: int = 2

    # CAB/attention settings
    kernel_size: int = 3
    reduction: int = 4
    bias: bool = False
    no_use_ca: bool = False
    learnable_prompt: bool = False

    # PromptMR+ extras (kept but disabled by default in our integration)
    n_history: int = 0


class PromptMRUNet(nn.Module):
    """PromptMR+ style CNN U-Net with prompt injection at each decoder level.

    This is designed as a *backbone* compatible with our VarNet cascades:
    input/output are real tensors shaped (B, C, H, W).

    Spatially it uses 3 downsamples (stride-2 convs), so H/W are padded to multiples of 8.
    """

    def __init__(self, in_chan: int, out_chan: int, cfg: PromptMRUNetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PromptMRUNetConfig()

        n_feat0 = int(cfg.n_feat0)
        if cfg.feature_dim is None:
            # Keep ratios similar to PromptMR+ defaults when n_feat0=48: [72, 96, 120] = [1.5, 2.0, 2.5] * n_feat0
            feature_dim = (
                max(1, int(round(n_feat0 * 1.5))),
                max(1, int(round(n_feat0 * 2.0))),
                max(1, int(round(n_feat0 * 2.5))),
            )
        else:
            feature_dim = tuple(int(x) for x in cfg.feature_dim)

        if cfg.prompt_dim is None:
            # PromptMR+ defaults when n_feat0=48: [24, 48, 72] = [0.5, 1.0, 1.5] * n_feat0
            prompt_dim = (
                max(1, int(round(n_feat0 * 0.5))),
                max(1, int(round(n_feat0 * 1.0))),
                max(1, int(round(n_feat0 * 1.5))),
            )
        else:
            prompt_dim = tuple(int(x) for x in cfg.prompt_dim)

        act = nn.PReLU()

        self.cfg = cfg
        self._feature_dim = feature_dim
        self._prompt_dim = prompt_dim

        # Minimal early normalization to keep feature scales controlled before
        # passing into many residual CAB blocks.
        self.feat_extract = nn.Sequential(
            _conv(in_chan, n_feat0, cfg.kernel_size, bias=cfg.bias),
            _instancenorm(n_feat0),
        )

        # Encoder (3 downs)
        self.enc_level1 = DownBlock(
            n_feat0,
            feature_dim[0],
            cfg.n_enc_cab[0],
            cfg.kernel_size,
            cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
            first_act=True,
        )
        self.enc_level2 = DownBlock(
            feature_dim[0],
            feature_dim[1],
            cfg.n_enc_cab[1],
            cfg.kernel_size,
            cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
        )
        self.enc_level3 = DownBlock(
            feature_dim[1],
            feature_dim[2],
            cfg.n_enc_cab[2],
            cfg.kernel_size,
            cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
        )

        # Skip attention blocks (match PromptMR+ pattern)
        self.skip1 = SkipBlock(n_feat0, cfg.n_skip_cab[0], cfg.kernel_size, cfg.reduction, bias=cfg.bias, act=act, no_use_ca=cfg.no_use_ca)
        self.skip2 = SkipBlock(feature_dim[0], cfg.n_skip_cab[1], cfg.kernel_size, cfg.reduction, bias=cfg.bias, act=act, no_use_ca=cfg.no_use_ca)
        self.skip3 = SkipBlock(feature_dim[1], cfg.n_skip_cab[2], cfg.kernel_size, cfg.reduction, bias=cfg.bias, act=act, no_use_ca=cfg.no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[
                CAB(feature_dim[2], cfg.kernel_size, cfg.reduction, bias=cfg.bias, act=act, no_use_ca=cfg.no_use_ca)
                for _ in range(cfg.n_bottleneck_cab)
            ]
        )

        # Decoder (3 ups) with prompt injection on each level
        self.prompt_level3 = PromptBlock(
            prompt_dim=prompt_dim[2],
            prompt_len=cfg.len_prompt[2],
            prompt_size=cfg.prompt_size[2],
            lin_dim=feature_dim[2],
            learnable_prompt=cfg.learnable_prompt,
        )
        self.dec_level3 = UpBlock(
            in_dim=feature_dim[2],
            out_dim=feature_dim[1],
            prompt_dim=prompt_dim[2],
            n_cab=cfg.n_dec_cab[2],
            kernel_size=cfg.kernel_size,
            reduction=cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
            n_history=cfg.n_history,
        )

        self.prompt_level2 = PromptBlock(
            prompt_dim=prompt_dim[1],
            prompt_len=cfg.len_prompt[1],
            prompt_size=cfg.prompt_size[1],
            lin_dim=feature_dim[1],
            learnable_prompt=cfg.learnable_prompt,
        )
        self.dec_level2 = UpBlock(
            in_dim=feature_dim[1],
            out_dim=feature_dim[0],
            prompt_dim=prompt_dim[1],
            n_cab=cfg.n_dec_cab[1],
            kernel_size=cfg.kernel_size,
            reduction=cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
            n_history=cfg.n_history,
        )

        self.prompt_level1 = PromptBlock(
            prompt_dim=prompt_dim[0],
            prompt_len=cfg.len_prompt[0],
            prompt_size=cfg.prompt_size[0],
            lin_dim=feature_dim[0],
            learnable_prompt=cfg.learnable_prompt,
        )
        self.dec_level1 = UpBlock(
            in_dim=feature_dim[0],
            out_dim=n_feat0,
            prompt_dim=prompt_dim[0],
            n_cab=cfg.n_dec_cab[0],
            kernel_size=cfg.kernel_size,
            reduction=cfg.reduction,
            bias=cfg.bias,
            act=act,
            no_use_ca=cfg.no_use_ca,
            n_history=cfg.n_history,
        )

        self.conv_last = _conv(n_feat0, out_chan, kernel_size=5, bias=cfg.bias)

        # Stability: in our VarNet cascades, the network output is subtracted as a
        # "regularizer/gradient" term each cascade. PromptMR+ blocks (unlike our baseline
        # U-Net) do not use normalization layers, which can cause large initial outputs
        # and NaNs with the same learning rates.
        # Zero-initializing the last layer makes the initial regularizer term ~0,
        # letting training ramp up safely.
        with torch.no_grad():
            nn.init.zeros_(self.conv_last.weight)
            if self.conv_last.bias is not None:
                nn.init.zeros_(self.conv_last.bias)

        self.in_chan = in_chan
        self.out_chan = out_chan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_sizes = self._pad(x)
        x_in = x

        x0 = self.feat_extract(x_in)

        x1, enc1 = self.enc_level1(x0)
        x2, enc2 = self.enc_level2(x1)
        x3, enc3 = self.enc_level3(x2)

        x3 = self.bottleneck(x3)

        p3 = self.prompt_level3(x3)
        x = self.dec_level3(x3, p3, self.skip3(enc3), history_feat=None)

        p2 = self.prompt_level2(x)
        x = self.dec_level2(x, p2, self.skip2(enc2), history_feat=None)

        p1 = self.prompt_level1(x)
        x = self.dec_level1(x, p1, self.skip1(enc1), history_feat=None)

        out = self.conv_last(x)

        out = self._unpad(out, *pad_sizes)
        return out

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        resize_factor = 8  # 3 downsamples

        h_mult = h if h % resize_factor == 0 else h + (resize_factor - h % resize_factor)
        w_mult = w if w % resize_factor == 0 else w + (resize_factor - w % resize_factor)

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def _unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
