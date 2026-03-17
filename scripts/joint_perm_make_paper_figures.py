#!/usr/bin/env python
"""Make paper-friendly ON vs OFF figures from joint permutation probe outputs.

Consumes per-cascade artifacts written by `scripts/joint_perm_all_modules_analysis.py`:
- cascade_X/comparison_long.csv
- cascade_X/metrics.csv (optional; used only for cross-checks)

Outputs a compact summary bundle into an output directory, including:
- paper_summary.md
- comparison_long_all_cascades.csv
- per_cascade_metric_summary.csv
- stage_summary_by_cascade.csv
- module_consistency_all_cascades.csv
- figures (PNG + PDF)

Design goal: plots that are easy to read as "PERM ON vs PERM OFF".
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


Metric = Literal["output_equiv_nmse", "repr_perm_cosdist", "knn_mixing_k10", "mmd2_rbf"]


@dataclass(frozen=True)
class CascadeInfo:
    idx: int
    path: Path
    comparison_long_csv: Path


def _find_cascades(run_root: Path) -> list[CascadeInfo]:
    cascades: list[CascadeInfo] = []
    for p in sorted(run_root.glob("cascade_*")):
        if not p.is_dir():
            continue
        suffix = p.name.split("cascade_", 1)[-1]
        try:
            idx = int(suffix)
        except Exception:
            continue
        comparison_long_csv = p / "comparison_long.csv"
        cascades.append(CascadeInfo(idx=idx, path=p, comparison_long_csv=comparison_long_csv))
    return cascades


def _is_complete(c: CascadeInfo) -> bool:
    return c.comparison_long_csv.exists()


def _stage_from_module_name(module: str) -> str:
    m = str(module)

    # Avoid polluting stage stats with the synthetic output row.
    if m == "__output__":
        return "output"

    # Heuristic stage bins (kept simple + stable).
    if m.startswith("down_sample_layers") or m.startswith("encoder"):
        return "encoder"
    if m.startswith("up_sample_layers") or m.startswith("decoder") or m.startswith("up_sample"):
        return "decoder"
    if "bottleneck" in m:
        return "bottleneck"
    return "other"


def _delta_good(row: pd.Series) -> float:
    """Signed improvement where positive means PERM(ON) is better."""
    expected = str(row.get("expected", ""))
    d = float(row.get("delta_on_minus_off", np.nan))
    if not np.isfinite(d):
        return np.nan

    if expected.startswith("lower_is_better"):
        return -d
    if expected.startswith("higher_is_better"):
        return d

    # Fall back: infer from metric name.
    metric = str(row.get("metric", ""))
    if metric in {"output_equiv_nmse", "repr_perm_cosdist", "mmd2_rbf"}:
        return -d
    if metric in {"knn_mixing_k10"}:
        return d
    return d


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_markdown_summary(
    out_dir: Path,
    *,
    run_root: Path,
    complete: list[int],
    incomplete: list[int],
    output_equiv_table: pd.DataFrame,
    per_cascade_metric_summary: pd.DataFrame,
) -> None:
    # Output equivariance summary (aggregate across cascades if multiple).
    if output_equiv_table.empty:
        output_lines = ["## Output equivariance", "- (missing)"]
    else:
        # Use mean across cascades (usually identical across cascades in this probe).
        off_mean = float(output_equiv_table["perm_off_mean"].mean())
        on_mean = float(output_equiv_table["perm_on_mean"].mean())
        ratio = on_mean / off_mean if off_mean > 0 else float("nan")

        # Also compute CI bounds conservatively by averaging per-cascade CI bounds.
        off_lo = float(output_equiv_table["perm_off_ci_low"].mean())
        off_hi = float(output_equiv_table["perm_off_ci_high"].mean())
        on_lo = float(output_equiv_table["perm_on_ci_low"].mean())
        on_hi = float(output_equiv_table["perm_on_ci_high"].mean())

        output_lines = [
            "## Output equivariance",
            f"- perm_off NMSE: {off_mean:.6g} (CI [{off_lo:.6g}, {off_hi:.6g}])",
            f"- perm_on  NMSE: {on_mean:.6g} (CI [{on_lo:.6g}, {on_hi:.6g}])",
            f"- ratio perm_on/perm_off: {ratio:.3f} (≈{(1.0/ratio):.2f}× lower)" if np.isfinite(ratio) and ratio > 0 else "- ratio perm_on/perm_off: (nan)",
        ]

    # Internal metrics summary.
    if per_cascade_metric_summary.empty:
        metric_lines = ["## Internal metrics", "- (missing)"]
    else:
        internal = per_cascade_metric_summary[per_cascade_metric_summary["metric"] != "output_equiv_nmse"].copy()
        frac = internal.groupby("metric", as_index=False)["frac_improved"].mean().sort_values("frac_improved", ascending=False)
        metric_lines = [
            "## Internal metrics: what does “PERM ON better” mean?",
            "- For metrics where lower is better (`repr_perm_cosdist`, `mmd2_rbf`): PERM ON is better when `perm_on_mean < perm_off_mean`.",
            "- For metrics where higher is better (`knn_mixing_k10`): PERM ON is better when `perm_on_mean > perm_off_mean`.",
            "",
            "## Internal metrics: fraction of modules where PERM ON is better (mean across cascades)",
        ]
        for _, r in frac.iterrows():
            metric_lines.append(f"- {r['metric']}: {float(r['frac_improved']):.3f}")

    md = "\n".join(
        [
            "# Joint permutation probe — all cascades summary",
            f"Run root: {run_root}",
            f"Complete cascades used: {complete} (n={len(complete)})",
            f"Incomplete cascades skipped: {incomplete}",
            "",
            *output_lines,
            "",
            *metric_lines,
            "",
        ]
    )
    (out_dir / "paper_summary.md").write_text(md)


def _ensure_plot_deps() -> tuple[object, object, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import seaborn as sns

    return matplotlib, plt, sns


def _save_fig(fig, out_dir: Path, stem: str, *, dpi: int = 220) -> None:
    fig.tight_layout()
    fig.savefig(str(out_dir / f"{stem}.png"), dpi=dpi)
    fig.savefig(str(out_dir / f"{stem}.pdf"))


def _plot_output_equiv_onoff(output_equiv_table: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    _, plt, sns = _ensure_plot_deps()

    if output_equiv_table.empty:
        return

    # Build two-row frame for plotting.
    off = output_equiv_table[["cascade", "perm_off_mean", "perm_off_ci_low", "perm_off_ci_high"]].copy()
    off = off.rename(
        columns={"perm_off_mean": "mean", "perm_off_ci_low": "ci_low", "perm_off_ci_high": "ci_high"}
    )
    off["model"] = "PERM OFF"

    on = output_equiv_table[["cascade", "perm_on_mean", "perm_on_ci_low", "perm_on_ci_high"]].copy()
    on = on.rename(columns={"perm_on_mean": "mean", "perm_on_ci_low": "ci_low", "perm_on_ci_high": "ci_high"})
    on["model"] = "PERM ON"

    df = pd.concat([off, on], axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    palette = {"PERM OFF": "#4C72B0", "PERM ON": "#55A868"}

    # Point + CI errorbar (per-cascade), plus a bold mean marker.
    sns.stripplot(
        data=df,
        x="model",
        y="mean",
        hue="model",
        palette=palette,
        ax=ax,
        dodge=False,
        jitter=0.06,
        size=3.5,
        alpha=0.55,
        linewidth=0,
    )

    for model, g in df.groupby("model"):
        x = 0 if model == "PERM OFF" else 1
        y = float(g["mean"].mean())
        lo = float(g["ci_low"].mean())
        hi = float(g["ci_high"].mean())
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", color="black", capsize=3, zorder=10)

    ax.set_yscale("log")
    ax.set_ylabel("Output equivariance NMSE (lower is better)")
    ax.set_xlabel("")
    ax.get_legend().remove()

    off_mean = float(off["mean"].mean())
    on_mean = float(on["mean"].mean())
    ratio = on_mean / off_mean if off_mean > 0 else float("nan")
    if np.isfinite(ratio) and ratio > 0:
        ax.set_title(f"Output equivariance: PERM ON is {1.0/ratio:.2f}× lower")

    ax.grid(True, which="both", axis="y", alpha=0.25)
    _save_fig(fig, out_dir, "fig_output_equiv_onoff_log", dpi=dpi)
    plt.close(fig)


def _plot_output_equiv_nmse_ci_seaborn(output_equiv_table: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """A non-log, explicit CI plot (kept for backward compatibility with older bundles)."""
    _, plt, sns = _ensure_plot_deps()

    if output_equiv_table.empty:
        return

    off = output_equiv_table[["cascade", "perm_off_mean", "perm_off_ci_low", "perm_off_ci_high"]].copy()
    off = off.rename(columns={"perm_off_mean": "mean", "perm_off_ci_low": "ci_low", "perm_off_ci_high": "ci_high"})
    off["model"] = "PERM OFF"

    on = output_equiv_table[["cascade", "perm_on_mean", "perm_on_ci_low", "perm_on_ci_high"]].copy()
    on = on.rename(columns={"perm_on_mean": "mean", "perm_on_ci_low": "ci_low", "perm_on_ci_high": "ci_high"})
    on["model"] = "PERM ON"

    df = pd.concat([off, on], axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    palette = {"PERM OFF": "#4C72B0", "PERM ON": "#55A868"}

    sns.stripplot(
        data=df,
        x="model",
        y="mean",
        hue="model",
        palette=palette,
        ax=ax,
        dodge=False,
        jitter=0.08,
        size=4,
        alpha=0.6,
        linewidth=0,
    )

    for model, g in df.groupby("model"):
        x = 0 if model == "PERM OFF" else 1
        y = float(g["mean"].mean())
        lo = float(g["ci_low"].mean())
        hi = float(g["ci_high"].mean())
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", color="black", capsize=3, zorder=10)

    ax.set_ylabel("Output equivariance NMSE (lower is better)")
    ax.set_xlabel("")
    ax.get_legend().remove()
    ax.grid(True, axis="y", alpha=0.25)
    _save_fig(fig, out_dir, "fig_output_equiv_nmse_ci_seaborn", dpi=dpi)
    plt.close(fig)


def _plot_frac_improved(per_cascade_metric_summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    _, plt, sns = _ensure_plot_deps()

    if per_cascade_metric_summary.empty:
        return

    df = per_cascade_metric_summary[per_cascade_metric_summary["metric"] != "output_equiv_nmse"].copy()

    # Mean and 95% CI across cascades using simple bootstrap.
    rows = []
    rng = np.random.default_rng(0)
    for metric, g in df.groupby("metric"):
        vals = g["frac_improved"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        boots = []
        for _ in range(2000):
            boots.append(float(rng.choice(vals, size=vals.size, replace=True).mean()))
        boots = np.asarray(boots)
        rows.append(
            {
                "metric": metric,
                "mean": float(vals.mean()),
                "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
            }
        )

    s = pd.DataFrame(rows).sort_values("mean", ascending=False)

    metric_name = {
        "knn_mixing_k10": "kNN mixing (higher better)",
        "mmd2_rbf": "MMD² (lower better)",
        "repr_perm_cosdist": "Representation cos dist (lower better)",
    }
    s["metric_label"] = s["metric"].map(metric_name).fillna(s["metric"])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.barplot(data=s, x="metric_label", y="mean", ax=ax, color="#4C72B0")

    for i, r in enumerate(s.itertuples(index=False)):
        ax.errorbar([i], [r.mean], yerr=[[r.mean - r.ci_low], [r.ci_high - r.mean]], fmt="none", ecolor="black", capsize=3)
        ax.text(i, r.mean + 0.015, f"{r.mean:.2f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0.5, color="gray", lw=1, alpha=0.6, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of modules where PERM ON is better")
    ax.set_xlabel("")
    ax.set_title("How often does PERM ON improve invariance? (across cascades)")
    ax.grid(True, axis="y", alpha=0.25)

    _save_fig(fig, out_dir, "fig_frac_modules_improved_by_metric", dpi=dpi)
    plt.close(fig)


def _plot_stage_frac_improved(stage_summary_by_cascade: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    _, plt, sns = _ensure_plot_deps()

    if stage_summary_by_cascade.empty:
        return

    df = stage_summary_by_cascade[stage_summary_by_cascade["stage"].isin(["encoder", "decoder", "bottleneck", "other"])].copy()
    df = df[df["metric"] != "output_equiv_nmse"].copy()

    metric_name = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    df["metric_label"] = df["metric"].map(metric_name).fillna(df["metric"])

    # Aggregate across cascades.
    agg = (
        df.groupby(["stage", "metric_label"], as_index=False)["frac_improved"]
        .mean()
        .sort_values(["metric_label", "stage"])
    )

    stage_order = ["encoder", "bottleneck", "decoder", "other"]
    agg["stage"] = pd.Categorical(agg["stage"], categories=stage_order, ordered=True)

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    sns.barplot(
        data=agg,
        x="metric_label",
        y="frac_improved",
        hue="stage",
        hue_order=stage_order,
        ax=ax,
    )
    ax.axhline(0.5, color="gray", lw=1, alpha=0.6, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction improved (PERM ON better)")
    ax.set_xlabel("")
    ax.set_title("Where does PERM ON help? (stage summary)")
    ax.legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid(True, axis="y", alpha=0.25)

    _save_fig(fig, out_dir, "fig_stage_frac_improved_grouped", dpi=dpi)
    plt.close(fig)


def _plot_delta_good_distributions(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    _, plt, sns = _ensure_plot_deps()

    if all_rows.empty:
        return

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"] != "output")].copy()
    df = df[np.isfinite(df["delta_good"]).to_numpy()]

    metric_order = ["knn_mixing_k10", "mmd2_rbf", "repr_perm_cosdist"]
    metric_label = {
        "knn_mixing_k10": "kNN mixing (right=better)",
        "mmd2_rbf": "MMD² (right=better)",
        "repr_perm_cosdist": "Cos dist (right=better)",
    }
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sns.violinplot(
        data=df,
        x="metric_label",
        y="delta_good",
        order=[metric_label[m] for m in metric_order if m in metric_label],
        ax=ax,
        inner="quartile",
        cut=0,
    )
    ax.axhline(0.0, color="black", lw=1, alpha=0.7)
    ax.set_ylabel("Signed change (PERM ON better →)")
    ax.set_xlabel("")
    ax.set_title("Distribution of per-module improvements (all cascades)")
    ax.grid(True, axis="y", alpha=0.25)

    _save_fig(fig, out_dir, "fig_violin_delta_good_simple", dpi=dpi)
    plt.close(fig)


def _plot_violin_delta_good_by_metric(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Delta-good distributions by metric, with stage hue (older bundles used this view)."""
    _, plt, sns = _ensure_plot_deps()

    if all_rows.empty:
        return

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"] != "output")].copy()
    df = df[np.isfinite(df["delta_good"]).to_numpy()]

    # Keep a stable, simple stage set.
    df = df[df["stage"].isin(["encoder", "bottleneck", "decoder", "other"])].copy()

    metric_order = ["knn_mixing_k10", "mmd2_rbf", "repr_perm_cosdist"]
    metric_label = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])

    stage_order = ["encoder", "bottleneck", "decoder", "other"]
    df["stage"] = pd.Categorical(df["stage"], categories=stage_order, ordered=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.violinplot(
        data=df,
        x="metric_label",
        y="delta_good",
        order=[metric_label[m] for m in metric_order if m in metric_label],
        hue="stage",
        hue_order=stage_order,
        ax=ax,
        inner="quartile",
        cut=0,
    )
    ax.axhline(0.0, color="black", lw=1, alpha=0.7)
    ax.set_ylabel("Signed change (PERM ON better →)")
    ax.set_xlabel("")
    ax.set_title("Per-module improvements by metric (colored by stage)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    _save_fig(fig, out_dir, "fig_violin_delta_good_by_metric", dpi=dpi)
    plt.close(fig)


def _plot_boxes_delta_good_by_stage_metric(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Boxplots of delta-good by stage/metric (useful diagnostic; present in older bundles)."""
    _, plt, sns = _ensure_plot_deps()

    if all_rows.empty:
        return

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"] != "output")].copy()
    df = df[np.isfinite(df["delta_good"]).to_numpy()]
    df = df[df["stage"].isin(["encoder", "bottleneck", "decoder", "other"])].copy()

    metric_label = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])
    stage_order = ["encoder", "bottleneck", "decoder", "other"]
    df["stage"] = pd.Categorical(df["stage"], categories=stage_order, ordered=True)

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    sns.boxplot(
        data=df,
        x="stage",
        y="delta_good",
        hue="metric_label",
        ax=ax,
        showfliers=False,
    )
    ax.axhline(0.0, color="black", lw=1, alpha=0.7)
    ax.set_ylabel("Signed change (PERM ON better →)")
    ax.set_xlabel("")
    ax.set_title("Per-module improvements by stage and metric")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    _save_fig(fig, out_dir, "fig_boxes_delta_good_by_stage_metric", dpi=dpi)
    plt.close(fig)


def _plot_heat_frac_improved_metric_by_cascade(per_cascade_metric_summary: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Heatmap: cascades (rows) x metrics (cols) showing frac improved."""
    _, plt, sns = _ensure_plot_deps()

    if per_cascade_metric_summary.empty:
        return

    df = per_cascade_metric_summary[per_cascade_metric_summary["metric"] != "output_equiv_nmse"].copy()
    metric_label = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])
    piv = df.pivot_table(index="cascade", columns="metric_label", values="frac_improved", aggfunc="mean")
    piv = piv.sort_index(axis=0)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.heatmap(
        piv,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Fraction improved"},
    )
    ax.set_title("Fraction improved by cascade and metric")
    ax.set_xlabel("")
    ax.set_ylabel("Cascade")

    _save_fig(fig, out_dir, "fig_heat_frac_improved_metric_by_cascade", dpi=dpi)
    plt.close(fig)


def _plot_heat_stage_frac_improved(stage_summary_by_cascade: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Heatmap: stages (rows) x metrics (cols) showing mean frac improved across cascades."""
    _, plt, sns = _ensure_plot_deps()

    if stage_summary_by_cascade.empty:
        return

    df = stage_summary_by_cascade[stage_summary_by_cascade["metric"] != "output_equiv_nmse"].copy()
    df = df[df["stage"].isin(["encoder", "bottleneck", "decoder", "other"])].copy()

    metric_label = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    df["metric_label"] = df["metric"].map(metric_label).fillna(df["metric"])

    agg = df.groupby(["stage", "metric_label"], as_index=False)["frac_improved"].mean()
    stage_order = ["encoder", "bottleneck", "decoder", "other"]
    agg["stage"] = pd.Categorical(agg["stage"], categories=stage_order, ordered=True)
    piv = agg.pivot_table(index="stage", columns="metric_label", values="frac_improved", aggfunc="mean")
    piv = piv.reindex(stage_order)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    sns.heatmap(
        piv,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Fraction improved"},
        annot=True,
        fmt=".2f",
    )
    ax.set_title("Stage × metric: mean fraction improved")
    ax.set_xlabel("")
    ax.set_ylabel("")

    _save_fig(fig, out_dir, "fig_heat_stage_frac_improved", dpi=dpi)
    plt.close(fig)


def _make_4panel(out_dir: Path, *, dpi: int) -> None:
    """Assemble a single easy-to-drop-in composite figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    panel_stems = [
        "fig_output_equiv_onoff_log",
        "fig_internal_onoff_scatter_grid",
        "fig_internal_metric_violin_onoff",
        "fig_stage_mean_value_onoff",
    ]

    imgs: list[np.ndarray] = []
    for s in panel_stems:
        p = out_dir / f"{s}.png"
        if not p.exists():
            return
        imgs.append(mpimg.imread(str(p)))

    # 2x2 grid
    # (w, h) not used; keep around in case we later want to enforce uniform panel sizes.
    _ = max(im.shape[1] for im in imgs), max(im.shape[0] for im in imgs)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel().tolist()
    letters = ["A", "B", "C", "D"]
    # Avoid Python 3.10+'s zip(strict=...).
    if not (len(axes) == len(imgs) == len(letters)):
        return
    for ax, im, letter in zip(axes, imgs, letters):
        ax.imshow(im)
        ax.set_axis_off()
        ax.text(0.01, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontsize=18, weight="bold")

    fig.tight_layout()
    fig.savefig(str(out_dir / "fig_paper_main_4panel.png"), dpi=dpi)
    fig.savefig(str(out_dir / "fig_paper_main_4panel.pdf"))
    plt.close(fig)


def _metric_better_direction(metric: str) -> str:
    # Used only for plot labels.
    if metric in {"knn_mixing_k10"}:
        return "higher"
    return "lower"


def _axis_scale_for_metric(metric: str) -> str:
    # Return one of: 'linear', 'log', 'symlog'
    if metric in {"output_equiv_nmse", "repr_perm_cosdist"}:
        return "log"
    if metric in {"mmd2_rbf"}:
        return "symlog"
    return "linear"


def _plot_internal_onoff_scatter_grid(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Direct ON vs OFF plot with y=x diagonal (no deltas).

    Interpretation:
    - lower-better metrics: points below diagonal => ON better
    - higher-better metrics: points above diagonal => ON better
    """

    _, plt, sns = _ensure_plot_deps()

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"] != "output")].copy()
    if df.empty:
        return

    metrics = ["knn_mixing_k10", "mmd2_rbf", "repr_perm_cosdist"]
    df = df[df["metric"].isin(metrics)].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, metric in zip(axes, metrics):
        g = df[df["metric"] == metric]
        x = g["perm_off_mean"].to_numpy(dtype=float)
        y = g["perm_on_mean"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            ax.set_axis_off()
            continue

        # Scatter density (many points); use low alpha for readability.
        ax.scatter(x, y, s=10, alpha=0.18, linewidths=0)

        # Diagonal y=x
        lo = float(np.nanmin(np.concatenate([x, y])))
        hi = float(np.nanmax(np.concatenate([x, y])))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = 0.0, 1.0

        # Keep a small padding.
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        lo2, hi2 = lo - pad, hi + pad

        ax.plot([lo2, hi2], [lo2, hi2], color="black", lw=1, alpha=0.8)
        ax.set_xlim(lo2, hi2)
        ax.set_ylim(lo2, hi2)

        scale = _axis_scale_for_metric(metric)
        if scale == "log":
            # Avoid non-positive values.
            min_pos = float(np.nanmin(np.concatenate([x[x > 0], y[y > 0]]))) if np.any(x > 0) or np.any(y > 0) else 1e-12
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(min_pos, hi2)
            ax.set_ylim(min_pos, hi2)
        elif scale == "symlog":
            # Show around 0; keep symmetric-ish.
            vmax = float(np.nanmax(np.abs(np.concatenate([x, y]))))
            lt = max(1e-12, vmax * 1e-3)
            ax.set_xscale("symlog", linthresh=lt)
            ax.set_yscale("symlog", linthresh=lt)

        better = _metric_better_direction(metric)
        if metric == "knn_mixing_k10":
            title = "kNN mixing"
            note = "Above diagonal = ON better"
        elif metric == "mmd2_rbf":
            title = "MMD² (RBF)"
            note = "Below diagonal = ON better"
        else:
            title = "Rep. cosine distance"
            note = "Below diagonal = ON better"

        ax.set_title(title)
        ax.set_xlabel("PERM OFF")
        ax.set_ylabel("PERM ON")
        ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Internal metrics: PERM ON vs PERM OFF (each point = one module, one cascade)")
    _save_fig(fig, out_dir, "fig_internal_onoff_scatter_grid", dpi=dpi)
    plt.close(fig)


def _plot_internal_metric_violin_onoff(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Violin distributions of module-level metric values (ON and OFF both shown)."""

    _, plt, sns = _ensure_plot_deps()

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"] != "output")].copy()
    if df.empty:
        return

    metrics = ["knn_mixing_k10", "mmd2_rbf", "repr_perm_cosdist"]
    df = df[df["metric"].isin(metrics)].copy()

    # Long-form: one row per value.
    off = df[["metric", "perm_off_mean"]].rename(columns={"perm_off_mean": "value"})
    off["model"] = "PERM OFF"
    on = df[["metric", "perm_on_mean"]].rename(columns={"perm_on_mean": "value"})
    on["model"] = "PERM ON"
    long = pd.concat([off, on], axis=0, ignore_index=True)
    long = long[np.isfinite(long["value"].to_numpy(dtype=float))]
    if long.empty:
        return

    metric_label = {
        "knn_mixing_k10": "kNN mixing (higher better)",
        "mmd2_rbf": "MMD² (lower better)",
        "repr_perm_cosdist": "Rep. cosine distance (lower better)",
    }
    long["metric_label"] = long["metric"].map(metric_label).fillna(long["metric"])

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    sns.violinplot(
        data=long,
        x="metric_label",
        y="value",
        hue="model",
        split=True,
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Module-level metric value")
    ax.set_title("Distributions across modules+cascades (PERM ON and PERM OFF shown explicitly)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    _save_fig(fig, out_dir, "fig_internal_metric_violin_onoff", dpi=dpi)
    plt.close(fig)


def _plot_stage_mean_value_onoff(all_rows: pd.DataFrame, out_dir: Path, *, dpi: int) -> None:
    """Stage-level mean values for ON and OFF (no deltas)."""

    _, plt, sns = _ensure_plot_deps()

    df = all_rows[(all_rows["metric"] != "output_equiv_nmse") & (all_rows["stage"].isin(["encoder", "bottleneck", "decoder", "other"]))].copy()
    if df.empty:
        return

    metrics = ["knn_mixing_k10", "mmd2_rbf", "repr_perm_cosdist"]
    df = df[df["metric"].isin(metrics)].copy()

    # Mean across modules and cascades per stage/metric.
    agg = (
        df.groupby(["stage", "metric"], as_index=False)
        .agg(
            perm_off_mean=("perm_off_mean", "mean"),
            perm_on_mean=("perm_on_mean", "mean"),
        )
        .copy()
    )

    stage_order = ["encoder", "bottleneck", "decoder", "other"]
    agg["stage"] = pd.Categorical(agg["stage"], categories=stage_order, ordered=True)

    # Convert to long to plot grouped bars.
    long = pd.concat(
        [
            agg[["stage", "metric", "perm_off_mean"]].rename(columns={"perm_off_mean": "value"}).assign(model="PERM OFF"),
            agg[["stage", "metric", "perm_on_mean"]].rename(columns={"perm_on_mean": "value"}).assign(model="PERM ON"),
        ],
        axis=0,
        ignore_index=True,
    )

    metric_label = {
        "knn_mixing_k10": "kNN mixing",
        "mmd2_rbf": "MMD²",
        "repr_perm_cosdist": "Cos dist",
    }
    long["metric_label"] = long["metric"].map(metric_label).fillna(long["metric"])

    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    sns.barplot(
        data=long,
        x="metric_label",
        y="value",
        hue="model",
        ci=None,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean value across modules+cascades")
    ax.set_title("Stage-aggregated mean metric values (PERM ON vs PERM OFF)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="")
    _save_fig(fig, out_dir, "fig_stage_mean_value_onoff", dpi=dpi)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_root",
        type=Path,
        default=Path("joint_perm_module_probe_full_v2"),
        help="Probe run root containing cascade_*/ folders.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: <run_root>/_paper_summary_all_cascades",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG DPI.",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (run_root / "_paper_summary_all_cascades")
    _safe_mkdir(out_dir)

    cascades = _find_cascades(run_root)
    complete = [c for c in cascades if _is_complete(c)]
    incomplete = [c for c in cascades if not _is_complete(c)]

    if not complete:
        raise SystemExit(f"No complete cascades found under {run_root}")

    frames = []
    for c in complete:
        df = pd.read_csv(c.comparison_long_csv)
        df["cascade"] = int(c.idx)
        frames.append(df)

    all_rows = pd.concat(frames, axis=0, ignore_index=True)

    # Normalize columns + compute derived values.
    all_rows["stage"] = all_rows["module"].astype(str).map(_stage_from_module_name)
    all_rows["delta_good"] = all_rows.apply(_delta_good, axis=1)
    all_rows["improved"] = all_rows["delta_good"] > 0

    # Write concatenated CSV.
    all_rows.to_csv(out_dir / "comparison_long_all_cascades.csv", index=False)

    # Output equivariance table (per cascade).
    output_equiv = all_rows[(all_rows["module"] == "__output__") & (all_rows["metric"] == "output_equiv_nmse")].copy()
    output_equiv = output_equiv[[
        "cascade",
        "perm_off_mean",
        "perm_on_mean",
        "perm_off_ci_low",
        "perm_off_ci_high",
        "perm_on_ci_low",
        "perm_on_ci_high",
        "perm_off_n",
        "perm_on_n",
    ]].sort_values("cascade")
    output_equiv.to_csv(out_dir / "output_equivariance_table.csv", index=False)

    # Per-cascade summary (per metric): fraction improved.
    def _summarize_delta_good(g: pd.DataFrame) -> pd.Series:
        vals = g["delta_good"].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        vals = vals[mask]
        if vals.size == 0:
            return pd.Series({"n_modules": 0, "frac_improved": np.nan, "delta_good_mean": np.nan, "delta_good_median": np.nan})
        return pd.Series(
            {
                "n_modules": int(vals.size),
                "frac_improved": float(np.mean((vals > 0).astype(float))),
                "delta_good_mean": float(np.mean(vals)),
                "delta_good_median": float(np.median(vals)),
            }
        )

    per_cascade_metric = (
        all_rows.groupby(["cascade", "metric"], as_index=False)
        .apply(_summarize_delta_good)
        .reset_index(drop=True)
        .sort_values(["cascade", "metric"])
    )
    per_cascade_metric.to_csv(out_dir / "per_cascade_metric_summary.csv", index=False)

    # Stage summary by cascade.
    stage_summary = (
        all_rows[all_rows["stage"] != "output"]
        .groupby(["cascade", "stage", "metric"], as_index=False)
        .apply(_summarize_delta_good)
        .reset_index(drop=True)
        .sort_values(["cascade", "metric", "stage"])
    )
    stage_summary.to_csv(out_dir / "stage_summary_by_cascade.csv", index=False)

    # Module consistency (across cascades): fraction of cascades improved.
    def _summarize_consistency(g: pd.DataFrame) -> pd.Series:
        vals = g["delta_good"].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        vals = vals[mask]
        if vals.size == 0:
            return pd.Series({"n_cascades": int(g["cascade"].nunique()), "frac_cascades_improved": np.nan, "delta_good_mean": np.nan})
        return pd.Series(
            {
                "n_cascades": int(g["cascade"].nunique()),
                "frac_cascades_improved": float(np.mean((vals > 0).astype(float))),
                "delta_good_mean": float(np.mean(vals)),
            }
        )

    module_consistency = (
        all_rows[all_rows["stage"] != "output"]
        .groupby(["module", "metric"], as_index=False)
        .apply(_summarize_consistency)
        .reset_index(drop=True)
        .sort_values(["metric", "frac_cascades_improved"], ascending=[True, False])
    )
    module_consistency.to_csv(out_dir / "module_consistency_all_cascades.csv", index=False)

    # Write markdown summary.
    _write_markdown_summary(
        out_dir,
        run_root=run_root,
        complete=[c.idx for c in complete],
        incomplete=[c.idx for c in incomplete],
        output_equiv_table=output_equiv,
        per_cascade_metric_summary=per_cascade_metric,
    )

    # Plots.
    dpi = int(args.dpi)
    _plot_output_equiv_onoff(output_equiv, out_dir, dpi=dpi)
    _plot_output_equiv_nmse_ci_seaborn(output_equiv, out_dir, dpi=dpi)
    # Keep the older "wins"/delta plots (still useful for some readers)
    _plot_frac_improved(per_cascade_metric, out_dir, dpi=dpi)
    _plot_stage_frac_improved(stage_summary, out_dir, dpi=dpi)
    _plot_delta_good_distributions(all_rows, out_dir, dpi=dpi)
    _plot_violin_delta_good_by_metric(all_rows, out_dir, dpi=dpi)
    _plot_boxes_delta_good_by_stage_metric(all_rows, out_dir, dpi=dpi)
    _plot_heat_frac_improved_metric_by_cascade(per_cascade_metric, out_dir, dpi=dpi)
    _plot_heat_stage_frac_improved(stage_summary, out_dir, dpi=dpi)

    # New: directly show ON and OFF values (no deltas).
    _plot_internal_onoff_scatter_grid(all_rows, out_dir, dpi=dpi)
    _plot_internal_metric_violin_onoff(all_rows, out_dir, dpi=dpi)
    _plot_stage_mean_value_onoff(all_rows, out_dir, dpi=dpi)
    _make_4panel(out_dir, dpi=dpi)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
