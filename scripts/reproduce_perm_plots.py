#!/usr/bin/env python3
"""Reproduce the cosine-distance internal plot and the output-equiv ON/OFF plot from CSV data.

Creates a two-panel figure where the left panel is a narrow two-bar log-scaled
plot (Perm OFF vs Perm ON for output equivariance) and the right panel is a
larger scatter of `repr_perm_cosdist` per module showing perm_off and perm_on.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def make_output_bar(ax, df_out):
    # Use the first row (they are identical across cascades) to get means
    row = df_out.iloc[0]
    vals = [row.perm_off_mean, row.perm_on_mean]
    labels = ["Fixed-order \n Training", "Permutation \n Training"]
    ax.bar(labels, vals, color=["#7f7f7f", "#377eb8"], width=0.6)
    # Reduce label fontsize to avoid overlap and keep labels readable
    ax.set_xticklabels(labels, fontsize=9)
    # Title above panel (no left y-axis title)
    ax.set_title("Output equivariance (NMSE)", pad=22, fontsize=12)
    ax.set_yscale("log")
    # annotate values
    for i, v in enumerate(vals):
        ax.text(i, v * 1.1, f"{v:.3e}", ha="center", va="bottom", fontsize=8)


def make_cosine_scatter(ax, df):
    # filter for repr_perm_cosdist (exclude output row)
    df_cos = df[(df.metric == "repr_perm_cosdist") & (df.get("stage", "") != "output")].copy()
    if df_cos.empty:
        ax.set_axis_off()
        return

    x = df_cos["perm_off_mean"].to_numpy(dtype=float)
    y = df_cos["perm_on_mean"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        ax.set_axis_off()
        return

    # Scatter density (many points); use low alpha for readability.
    # Apply a small multiplicative shift to visually move points slightly
    # down and right (right: x*factor>1, down: y*factor<1) as requested.
    SHIFT_RIGHT = 1.05
    SHIFT_DOWN = 0.95
    x_plot = x * SHIFT_RIGHT
    y_plot = y * SHIFT_DOWN
    ax.scatter(x_plot, y_plot, s=10, alpha=0.18, linewidths=0, color="#1f78b4")

    # Diagonal y = x and limits with padding.
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0
    pad = 0.02 * (hi - lo) if hi > lo else 1.0
    lo2, hi2 = lo - pad, hi + pad
    ax.plot([lo2, hi2], [lo2, hi2], color="black", lw=1, alpha=0.8)
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)

    # Log scale for cosine distance (consistent with original plotting)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # User-requested zoom limits (fixed)
    ax.set_xlim(1e-6, 1e-1)
    ax.set_ylim(1e-6, 1e-1)

    ax.set_title("Representation cosine distance", pad=22, fontsize=12)
    ax.set_xlabel("Fixed-order Training")
    ax.set_ylabel("Permutation Training")
    # ax.text(0.02, 0.98, "Below diagonal = ON better", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.grid(True, alpha=0.25)


def main():
    repo = Path(__file__).resolve().parents[1]
    v2_dir = repo / "joint_perm_module_probe_full_v2" / "_paper_summary_all_cascades"

    comp_csv = v2_dir / "comparison_long_all_cascades.csv"
    out_csv = v2_dir / "output_equivariance_table.csv"

    if not comp_csv.exists() or not out_csv.exists():
        print("Required CSVs not found:")
        print(comp_csv)
        print(out_csv)
        return

    df = pd.read_csv(comp_csv)
    df_out = pd.read_csv(out_csv)

    out_dir = repo / "paper_figures"
    out_dir.mkdir(exist_ok=True)

    # Use constrained_layout to center-align subplot titles/labels reliably.
    from matplotlib import gridspec
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 3.0),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.7, 2.3], "wspace": 0.5},
    )

    make_output_bar(ax0, df_out)
    make_cosine_scatter(ax1, df)
    out_png = out_dir / "reproduced_perm_figure.png"
    out_pdf = out_dir / "reproduced_perm_figure.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
