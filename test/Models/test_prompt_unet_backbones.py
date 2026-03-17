import pytest
import torch

from ml_recon.models.PromptUNet import PromptUNet
from ml_recon.models.PromptMRUNet import PromptMRUNet


@torch.no_grad()
@pytest.mark.parametrize("model_cls", [PromptUNet, PromptMRUNet])
@pytest.mark.parametrize("h,w", [(63, 65), (64, 64), (128, 120)])
def test_prompt_backbone_forward_shape(model_cls, h, w):
    # Inputs are real-valued tensors shaped (B, C, H, W).
    # In the VarNet blocks, C is 2 * num_contrasts.
    b = 2
    in_chan = 6
    out_chan = 6

    model = model_cls(in_chan=in_chan, out_chan=out_chan)
    x = torch.randn(b, in_chan, h, w)

    y = model(x)

    assert y.shape == (b, out_chan, h, w)
    assert torch.isfinite(y).all()


@torch.no_grad()
@pytest.mark.parametrize("backbone", ["unet", "prompt_unet", "promptmr_unet"])
def test_backbone_string_wires_into_varnet(backbone):
    # Smoke-test that the full VarNet can be constructed with all backbones.
    from ml_recon.models.MultiContrastVarNet import MultiContrastVarNet, VarnetConfig

    model = MultiContrastVarNet(VarnetConfig(contrast_order=["t2"], model=backbone, cascades=1, channels=8, depth=2))

    # reference_k: (B, contrast, coils, H, W) complex
    b, con, coils, h, w = 1, 1, 4, 32, 40
    reference_k = torch.complex(torch.randn(b, con, coils, h, w), torch.randn(b, con, coils, h, w))

    mask = torch.ones(b, con, coils, h, w, dtype=torch.float32)
    zf_mask = mask

    out = model(reference_k, mask, zf_mask)

    assert out.shape == reference_k.shape
    assert not torch.isnan(out).any()
