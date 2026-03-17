#!/usr/bin/env python3
"""Create a publication-ready two-panel figure combining two existing images.

Saves high-resolution PNG and PDF into `paper_figures/combined_perm_figure.{png,pdf}`.
"""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def main():
    repo_root = Path(__file__).resolve().parents[1]
    img1 = repo_root / "joint_perm_module_probe_full_v2" / "_paper_summary_all_cascades" / "fig_internal_onoff_scatter_grid.png"
    img2 = repo_root / "joint_perm_module_probe_full_NEW_MODEL" / "_paper_summary_all_cascades" / "fig_output_equiv_onoff_log.png"

    out_dir = repo_root / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img1.exists() or not img2.exists():
        print("One or both input images not found:")
        print(f" - {img1}")
        print(f" - {img2}")
        return

    im1 = Image.open(img1).convert("RGBA")
    im2 = Image.open(img2).convert("RGBA")

    # Crop im1 to the right 70% to focus on the cosine-distance plot
    w1, h1 = im1.size
    crop_left = int(w1 * 0.30)
    im1_cropped = im1.crop((crop_left, 0, w1, h1))

    # Create figure with narrow left panel (A) and wide right panel (B)
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.08)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # Left: small panel (A)
    ax_left.imshow(im2)
    ax_left.axis("off")
    ax_left.set_title("A", fontsize=12, weight="bold", loc="left")

    # Right: big panel with cropped cosine-distance image (B)
    ax_right.imshow(im1_cropped)
    ax_right.axis("off")
    ax_right.set_title("B", fontsize=12, weight="bold", loc="left")

    # Tight layout and save
    out_png = out_dir / "combined_perm_figure.png"
    out_pdf = out_dir / "combined_perm_figure.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
