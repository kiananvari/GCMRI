from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(var + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """LayerNorm over channel-dim, applied per spatial location.

    PromptIR implements LN in token space (B, HW, C). This wrapper keeps the same behavior
    while maintaining a (B, C, H, W) interface.
    """

    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = _BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = _WithBiasLayerNorm(dim)
        else:
            raise ValueError(f"Invalid layer_norm_type={layer_norm_type!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


class GatedDconvFFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head ch) h w -> b head ch (h w)", head=self.num_heads)
        k = rearrange(k, "b (head ch) h w -> b head ch (h w)", head=self.num_heads)
        v = rearrange(v, "b (head ch) h w -> b head ch (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b head ch (h w) -> b (head ch) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = MDTA(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = GatedDconvFFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptGenBlock(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        prompt_len: int,
        prompt_size: int,
        lin_dim: int,
    ) -> None:
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # [B, prompt_len]

        # Weighted sum over prompt bank
        prompt_bank = self.prompt_param.squeeze(0)  # [prompt_len, prompt_dim, ps, ps]
        prompt = torch.einsum('bl,lchw->bchw', prompt_weights, prompt_bank)

        prompt = F.interpolate(prompt, (h, w), mode='bilinear')
        prompt = self.conv3x3(prompt)
        return prompt


class Downsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


@dataclass(frozen=True)
class PromptUNetConfig:
    dim: int = 48
    num_blocks: Tuple[int, int, int, int] = (4, 6, 6, 8)
    num_refinement_blocks: int = 4
    heads: Tuple[int, int, int, int] = (1, 2, 4, 8)
    ffn_expansion_factor: float = 2.66
    bias: bool = False
    layer_norm_type: str = "WithBias"
    use_prompts: bool = True
    prompt_len: int = 5
    prompt_size1: int = 64
    prompt_size2: int = 32
    prompt_size3: int = 16


class PromptUNet(nn.Module):
    """Transformer U-Net backbone with PromptIR-style prompt injection.

    Uses padding so H/W are divisible by 8 (3 downsamples).
    """

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        cfg: PromptUNetConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or PromptUNetConfig()
        self.cfg = cfg

        dim = cfg.dim

        # PromptIR prompt channel sizes scale with dim (original PromptIR uses dim=48).
        # We also adjust them so that the attention head split is valid for the
        # prompt-injection transformer blocks (dim must be divisible by num_heads).
        head_noise = cfg.heads[2]

        def _adjust_prompt_dim(base_dim: int, prompt_dim: int, divisor: int) -> int:
            total = base_dim + prompt_dim
            total = int(math.ceil(total / divisor) * divisor)
            return total - base_dim

        prompt1_dim = _adjust_prompt_dim(dim * 2, int(round(dim * 4 / 3)), head_noise)
        prompt2_dim = _adjust_prompt_dim(dim * 4, int(round(dim * 8 / 3)), head_noise)
        prompt3_dim = _adjust_prompt_dim(dim * 8, int(round(dim * 20 / 3)), head_noise)
        self._prompt_dims = (prompt1_dim, prompt2_dim, prompt3_dim)

        self.use_prompts = cfg.use_prompts

        self.patch_embed = OverlapPatchEmbed(in_chan, dim, bias=cfg.bias)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=cfg.heads[0],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=cfg.heads[1],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 4,
                    num_heads=cfg.heads[2],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 8,
                    num_heads=cfg.heads[3],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[3])
            ]
        )

        if self.use_prompts:
            self.prompt1 = PromptGenBlock(
                prompt_dim=prompt1_dim,
                prompt_len=cfg.prompt_len,
                prompt_size=cfg.prompt_size1,
                lin_dim=dim * 2,
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=prompt2_dim,
                prompt_len=cfg.prompt_len,
                prompt_size=cfg.prompt_size2,
                lin_dim=dim * 4,
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=prompt3_dim,
                prompt_len=cfg.prompt_len,
                prompt_size=cfg.prompt_size3,
                lin_dim=dim * 8,
            )

            # PromptIR-style noise-level injection blocks
            self.noise_level3 = TransformerBlock(
                dim=(dim * 8) + prompt3_dim,
                num_heads=cfg.heads[2],
                ffn_expansion_factor=cfg.ffn_expansion_factor,
                bias=cfg.bias,
                layer_norm_type=cfg.layer_norm_type,
            )
            self.reduce_noise_level3 = nn.Conv2d((dim * 8) + prompt3_dim, dim * 4, kernel_size=1, bias=cfg.bias)

            self.noise_level2 = TransformerBlock(
                dim=(dim * 4) + prompt2_dim,
                num_heads=cfg.heads[2],
                ffn_expansion_factor=cfg.ffn_expansion_factor,
                bias=cfg.bias,
                layer_norm_type=cfg.layer_norm_type,
            )
            self.reduce_noise_level2 = nn.Conv2d((dim * 4) + prompt2_dim, dim * 4, kernel_size=1, bias=cfg.bias)

            self.noise_level1 = TransformerBlock(
                dim=(dim * 2) + prompt1_dim,
                num_heads=cfg.heads[2],
                ffn_expansion_factor=cfg.ffn_expansion_factor,
                bias=cfg.bias,
                layer_norm_type=cfg.layer_norm_type,
            )
            self.reduce_noise_level1 = nn.Conv2d((dim * 2) + prompt1_dim, dim * 2, kernel_size=1, bias=cfg.bias)
        else:
            self.latent_reduce = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=cfg.bias)

        # Decoder path expects latent reduced to dim*4 (PromptIR topology)
        self.up4_3 = Upsample(dim * 4)
        self.reduce_chan_level3 = nn.Conv2d(dim * 6, dim * 4, kernel_size=1, bias=cfg.bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 4,
                    num_heads=cfg.heads[2],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=cfg.bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=cfg.heads[1],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=cfg.heads[0],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=cfg.heads[0],
                    ffn_expansion_factor=cfg.ffn_expansion_factor,
                    bias=cfg.bias,
                    layer_norm_type=cfg.layer_norm_type,
                )
                for _ in range(cfg.num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(dim * 2, out_chan, kernel_size=3, stride=1, padding=1, bias=cfg.bias)

        self.in_chan = in_chan
        self.out_chan = out_chan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_sizes = self._pad(x)

        enc1_in = self.patch_embed(x)
        enc1 = self.encoder_level1(enc1_in)

        enc2_in = self.down1_2(enc1)
        enc2 = self.encoder_level2(enc2_in)

        enc3_in = self.down2_3(enc2)
        enc3 = self.encoder_level3(enc3_in)

        enc4_in = self.down3_4(enc3)
        latent = self.latent(enc4_in)

        if self.use_prompts:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], dim=1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        else:
            latent = self.latent_reduce(latent)

        dec3_in = self.up4_3(latent)
        dec3_in = torch.cat([dec3_in, enc3], dim=1)
        dec3_in = self.reduce_chan_level3(dec3_in)
        dec3 = self.decoder_level3(dec3_in)

        if self.use_prompts:
            dec2_param = self.prompt2(dec3)
            dec3 = torch.cat([dec3, dec2_param], dim=1)
            dec3 = self.noise_level2(dec3)
            dec3 = self.reduce_noise_level2(dec3)

        dec2_in = self.up3_2(dec3)
        dec2_in = torch.cat([dec2_in, enc2], dim=1)
        dec2_in = self.reduce_chan_level2(dec2_in)
        dec2 = self.decoder_level2(dec2_in)

        if self.use_prompts:
            dec1_param = self.prompt1(dec2)
            dec2 = torch.cat([dec2, dec1_param], dim=1)
            dec2 = self.noise_level1(dec2)
            dec2 = self.reduce_noise_level1(dec2)

        dec1_in = self.up2_1(dec2)
        dec1_in = torch.cat([dec1_in, enc1], dim=1)
        dec1 = self.decoder_level1(dec1_in)

        dec1 = self.refinement(dec1)

        out = self.output(dec1)

        # residual output when shapes match (common in our VarNet usage)
        if self.in_chan == self.out_chan:
            out = out + x

        out = self._unpad(out, *pad_sizes)
        return out

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        resize_factor = 8  # 3 downsamples via PixelUnshuffle

        h_mult = h if h % resize_factor == 0 else h + (resize_factor - h % resize_factor)
        w_mult = w if w % resize_factor == 0 else w + (resize_factor - w % resize_factor)

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def _unpad(
        self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
