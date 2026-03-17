#!/usr/bin/env python
"""Reproduce paper-ready figures from saved feature_comparison outputs.

This script is *offline*: it does NOT rerun models. It consumes the artifacts
already written by `feature_comparison.py`:

- Figure A: figureA_pca_maps/<probe>/figureA_pca_maps_pc*.png
- Figure B: figureB_contrast_id_probe/<probe>/figureB_contrast_id_probe_acc.png
- Figure C: figureC_alignment/<probe>/figureC_similarity_matrix_*.png + simmaps
- Figure D: figureD_perm_consistency/figureD_perm_consistency_*.npz (we render plots)

Example:
  python scripts/reproduce_paper_figures.py \
    --run_root feature_comparison_outputs_D \
    --out_dir paper_figures/cascade1

"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def _robust_symmetric_vlim(values: np.ndarray, q: float = 0.99) -> float:
    if values.size == 0:
        return 1.0
    a = np.quantile(np.abs(values.astype(np.float64).ravel()), q)
    if not np.isfinite(a) or a <= 0:
        return 1.0
    return float(a)


def _render_figure_a_from_npz(
    a_src: Path,
    out_dir: Path,
    *,
    max_components: int = 5,
    symlog: bool = False,
    symlog_linthresh: float = 0.0,
) -> bool:
    """Offline renderer for Figure A.

    Consumes `figureA_pca_maps_<tag>.npz` artifacts and writes:
    - figureA_pca_maps_pc*.png (with per-image colorbars)
    - figureA_pca_value_distributions.png (overlay per-PC histograms)
    """
    if not a_src.exists():
        return False

    map_files = sorted(a_src.glob("figureA_pca_maps_*.npz"))
    # Expect two tags (e.g., perm_on / perm_off). If missing, fall back to copying PNGs.
    if len(map_files) < 2:
        return False

    # Load first two (stable behavior). If you have more, prefer the first two.
    Za = np.load(map_files[0], allow_pickle=True)
    Zb = np.load(map_files[1], allow_pickle=True)

    tag_a = str(Za.get("tag", Path(map_files[0]).stem.replace("figureA_pca_maps_", "")))
    tag_b = str(Zb.get("tag", Path(map_files[1]).stem.replace("figureA_pca_maps_", "")))

    contrasts_a = [str(x) for x in Za.get("contrasts", np.asarray([], dtype="U")).tolist()]
    contrasts_b = [str(x) for x in Zb.get("contrasts", np.asarray([], dtype="U")).tolist()]
    common = [c for c in contrasts_a if c in set(contrasts_b)]
    if not common:
        return False

    def _get_map(Z, c: str) -> np.ndarray:
        key = f"pca_map__{c}"
        if key not in Z:
            raise KeyError(key)
        return np.asarray(Z[key])  # (H,W,K)

    try:
        # Infer number of components from first contrast.
        sample_map = _get_map(Za, common[0])
        if sample_map.ndim != 3:
            return False
        n_components = int(sample_map.shape[-1])
    except Exception:
        return False

    n_show = int(max(1, min(int(max_components), int(n_components))))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
    except Exception:
        return False

    _safe_mkdir(out_dir)

    # Clean any stale renders so the output reflects the requested component count.
    for f in out_dir.glob("figureA_pca_maps_pc*.png"):
        try:
            f.unlink()
        except Exception:
            pass
    for f in [
        out_dir / "figureA_pca_value_distributions.png",
        out_dir / "figureA_pca_value_distributions_symlog.png",
        out_dir / f"figureA_pca2_{tag_a}.png",
        out_dir / f"figureA_pca2_{tag_b}.png",
        out_dir / f"figureA_pca2_{tag_a}_symlog.png",
        out_dir / f"figureA_pca2_{tag_b}_symlog.png",
    ]:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    def _add_colorbar(fig, ax, im) -> None:
        try:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.85)
            vmin, vmax = im.get_clim()
            cb.set_ticks([vmin, 0.0, vmax])
        except Exception:
            pass

    def _panel_vlim(m: np.ndarray) -> float:
        # "Show original": do not force a shared vlim across panels/rows.
        # Use the map's own max-abs for a symmetric seismic scale.
        mm = np.asarray(m, dtype=np.float64)
        if mm.size == 0:
            return 1.0
        a = float(np.nanmax(np.abs(mm)))
        if not np.isfinite(a) or a <= 0:
            return 1.0
        return a

    def _panel_vlim_symlog(m: np.ndarray) -> float:
        # For symlog, max-abs can be dominated by outliers, making everything near 0.
        # Use a robust quantile to reveal structure.
        mm = np.asarray(m, dtype=np.float64)
        if mm.size == 0:
            return 1.0
        a = float(np.quantile(np.abs(mm).ravel(), 0.995))
        if not np.isfinite(a) or a <= 0:
            a = float(np.nanmax(np.abs(mm)))
        if not np.isfinite(a) or a <= 0:
            return 1.0
        return a

    def _auto_linthresh(v: float) -> float:
        # If user passes 0, choose a data-scaled threshold (smaller => more log behavior near 0).
        if float(symlog_linthresh) > 0:
            return float(symlog_linthresh)
        return max(1e-12, float(v) * 1e-3)

    def _imshow(ax, m: np.ndarray, v: float, *, use_symlog: bool):
        if not use_symlog:
            return ax.imshow(m, cmap="seismic", vmin=-v, vmax=v)
        lt = _auto_linthresh(v)
        norm = SymLogNorm(linthresh=lt, linscale=1.0, vmin=-v, vmax=v, base=10)
        return ax.imshow(m, cmap="seismic", norm=norm)

    # Render per-PC grids (only first n_show PCs).
    for pc_idx in range(n_show):
        for use_symlog, suffix, title_prefix in [
            (False, "", "Figure A"),
            (True, "_symlog", "Figure A (symlog)"),
        ]:
            if use_symlog and (not symlog):
                continue
            fig, axes = plt.subplots(2, len(common), figsize=(2.2 * len(common), 4.8), squeeze=False)
            for col, c in enumerate(common):
                ax0 = axes[0][col]
                ax1 = axes[1][col]
                m0 = _get_map(Za, c)[..., pc_idx]
                m1 = _get_map(Zb, c)[..., pc_idx]
                if use_symlog:
                    v0 = _panel_vlim_symlog(m0)
                    v1 = _panel_vlim_symlog(m1)
                else:
                    v0 = _panel_vlim(m0)
                    v1 = _panel_vlim(m1)
                im0 = _imshow(ax0, m0, v0, use_symlog=use_symlog)
                im1 = _imshow(ax1, m1, v1, use_symlog=use_symlog)
                _add_colorbar(fig, ax0, im0)
                _add_colorbar(fig, ax1, im1)
                ax0.set_title(c)
                ax0.set_axis_off()
                ax1.set_axis_off()

            axes[0][0].set_ylabel(tag_a)
            axes[1][0].set_ylabel(tag_b)
            fig.suptitle(f"{title_prefix}: PCA maps (PC{pc_idx+1})")
            fig.tight_layout()
            fig.savefig(str(out_dir / f"figureA_pca_maps_pc{pc_idx+1}{suffix}.png"), dpi=200)
            plt.close(fig)

    # Render distribution comparison (one row per shown PC).
    try:
        fig, axes = plt.subplots(n_show, 1, figsize=(5.2, 1.6 * n_show), squeeze=False)
        for pc_idx in range(n_show):
            vals_a = np.concatenate([_get_map(Za, c)[..., pc_idx].ravel() for c in common], axis=0).astype(np.float64)
            vals_b = np.concatenate([_get_map(Zb, c)[..., pc_idx].ravel() for c in common], axis=0).astype(np.float64)
            vals_all = np.concatenate([vals_a, vals_b], axis=0)
            vlim = _robust_symmetric_vlim(vals_all, q=0.995)
            bins = np.linspace(-vlim, vlim, 80).tolist()
            ax = axes[pc_idx][0]
            ax.hist(vals_a, bins=bins, density=True, alpha=0.55, label=tag_a)
            ax.hist(vals_b, bins=bins, density=True, alpha=0.55, label=tag_b)
            ax.set_xlim(-vlim, vlim)
            ax.set_ylabel("density")
            ax.set_title(f"PC{pc_idx+1} value distribution")
            if pc_idx == n_show - 1:
                ax.set_xlabel("PCA-projected value")
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(str(out_dir / "figureA_pca_value_distributions.png"), dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Extra: symlog view to reveal small structure in the distribution tails.
    if symlog:
        try:
            fig, axes = plt.subplots(n_show, 1, figsize=(5.2, 1.8 * n_show), squeeze=False)
            for pc_idx in range(n_show):
                vals_a = np.concatenate([_get_map(Za, c)[..., pc_idx].ravel() for c in common], axis=0).astype(np.float64)
                vals_b = np.concatenate([_get_map(Zb, c)[..., pc_idx].ravel() for c in common], axis=0).astype(np.float64)
                vals_all = np.concatenate([vals_a, vals_b], axis=0)
                vlim = _robust_symmetric_vlim(vals_all, q=0.999)
                lt = _auto_linthresh(vlim)
                bins = np.linspace(-vlim, vlim, 120).tolist()
                ax = axes[pc_idx][0]
                ax.hist(vals_a, bins=bins, density=True, alpha=0.45, label=tag_a)
                ax.hist(vals_b, bins=bins, density=True, alpha=0.45, label=tag_b)
                ax.set_xscale("symlog", linthresh=lt)
                ax.set_yscale("log")
                ax.set_xlim(-vlim, vlim)
                ax.set_ylabel("density (log)")
                ax.set_title(f"PC{pc_idx+1} distribution (symlog x)")
                if pc_idx == n_show - 1:
                    ax.set_xlabel("PCA-projected value")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(str(out_dir / "figureA_pca_value_distributions_symlog.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass

    # Render a compact "2 PCA" view per model: rows=PC1/PC2, cols=contrasts.
    # Only makes sense if we are showing at least 2 PCs.
    if n_show >= 2:
        try:
            for tag, Z in [(tag_a, Za), (tag_b, Zb)]:
                for use_symlog, suffix, title_prefix in [
                    (False, "", "Figure A"),
                    (True, "_symlog", "Figure A (symlog)"),
                ]:
                    if use_symlog and (not symlog):
                        continue
                    fig, axes = plt.subplots(2, len(common), figsize=(2.2 * len(common), 5.0), squeeze=False)
                    for col, c in enumerate(common):
                        for row, pc_idx in enumerate([0, 1]):
                            ax = axes[row][col]
                            m = _get_map(Z, c)[..., pc_idx]
                            v = _panel_vlim(m)
                            im = _imshow(ax, m, v, use_symlog=use_symlog)
                            _add_colorbar(fig, ax, im)
                            if row == 0:
                                ax.set_title(c)
                            ax.set_axis_off()
                    axes[0][0].set_ylabel("PC1")
                    axes[1][0].set_ylabel("PC2")
                    fig.suptitle(f"{title_prefix}: 2 PCA maps ({tag})")
                    fig.tight_layout()
                    fig.savefig(str(out_dir / f"figureA_pca2_{tag}{suffix}.png"), dpi=200)
                    plt.close(fig)
        except Exception:
            pass

    # Write a convenience 2-PC components NPZ if the original components exist.
    try:
        comp_file = a_src / "figureA_pca_components.npz"
        if comp_file.exists():
            Zc = np.load(comp_file, allow_pickle=True)
            mean = np.asarray(Zc["mean"]).astype(np.float32)
            comps = np.asarray(Zc["components"]).astype(np.float32)
            np.savez_compressed(
                out_dir / "figureA_pca_components_2pc.npz",
                mean=mean,
                components=comps[:n_show],
                contrasts=np.asarray(common, dtype="U"),
                n_components=np.asarray(n_show, dtype=np.int32),
                tag_a=np.asarray(tag_a, dtype="U"),
                tag_b=np.asarray(tag_b, dtype="U"),
            )
    except Exception:
        pass

    return True


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return True


def _list_probes(run_root: Path) -> list[str]:
    a_root = run_root / "figureA_pca_maps"
    if not a_root.exists():
        return []
    probes = []
    for p in a_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "figureA_pca_components.npz").exists() or any(p.glob("figureA_pca_maps_pc*.png")):
            probes.append(p.name)
    # stable, network-ish ordering
    def key(name: str) -> tuple[int, int]:
        if name == "stem":
            return (0, 0)
        if name.startswith("enc"):
            try:
                return (1, int(name.replace("enc", "")))
            except Exception:
                return (1, 999)
        if name == "bottleneck":
            return (2, 0)
        if name.startswith("dec"):
            try:
                return (3, -int(name.replace("dec", "")))
            except Exception:
                return (3, 0)
        if name == "out":
            return (4, 0)
        return (9, 0)

    return sorted(probes, key=key)


def _render_figure_d(run_root: Path, out_dir: Path) -> None:
    d_root = run_root / "figureD_perm_consistency"
    if not d_root.exists():
        return

    on_npz = d_root / "figureD_perm_consistency_perm_on.npz"
    off_npz = d_root / "figureD_perm_consistency_perm_off.npz"
    if not (on_npz.exists() and off_npz.exists()):
        return

    Z_on = np.load(on_npz, allow_pickle=True)
    Z_off = np.load(off_npz, allow_pickle=True)

    overall_on = Z_on["overall_nrmse"].astype(np.float64)
    overall_off = Z_off["overall_nrmse"].astype(np.float64)

    contrasts = [str(x) for x in Z_on.get("contrasts", np.asarray([], dtype="U")).tolist()]
    pc_on = Z_on.get("per_contrast_nrmse", np.zeros((0, 0), dtype=np.float64)).astype(np.float64)
    pc_off = Z_off.get("per_contrast_nrmse", np.zeros((0, 0), dtype=np.float64)).astype(np.float64)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    _safe_mkdir(out_dir)

    # Plot 1: overlay histogram of overall NRMSE
    fig = plt.figure(figsize=(6.5, 4.0))
    # Robust bins in log space, but D values are small so linear bins also work.
    lo = float(np.nanmin([overall_on.min(), overall_off.min()]))
    hi = float(np.nanmax([overall_on.max(), overall_off.max()]))
    lo = max(lo, 1e-8)
    hi = max(hi, lo * 1.01)
    bins = np.linspace(lo, hi, 50).tolist()
    plt.hist(overall_on, bins=bins, alpha=0.55, label="perm_on")
    plt.hist(overall_off, bins=bins, alpha=0.55, label="perm_off")
    plt.xlabel("overall NRMSE (output change under permutation)")
    plt.ylabel("count")
    plt.title("Figure D: Permutation consistency")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.tight_layout()
    fig.savefig(str(out_dir / "figureD_overall_hist.png"), dpi=200)
    fig.savefig(str(out_dir / "figureD_overall_hist.pdf"))
    plt.close(fig)

    # Plot 2: per-contrast mean bars (if available)
    if pc_on.size and pc_off.size and pc_on.shape == pc_off.shape and pc_on.shape[1] == len(contrasts):
        mean_on = np.nanmean(pc_on, axis=0)
        mean_off = np.nanmean(pc_off, axis=0)

        x = np.arange(len(contrasts), dtype=np.float64)
        w = 0.38
        fig = plt.figure(figsize=(6.5, 4.0))
        plt.bar(x - w / 2, mean_on, width=w, label="perm_on")
        plt.bar(x + w / 2, mean_off, width=w, label="perm_off")
        plt.xticks(x, contrasts)
        plt.ylabel("mean per-contrast NRMSE")
        plt.title("Figure D: Per-contrast permutation consistency")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        fig.tight_layout()
        fig.savefig(str(out_dir / "figureD_per_contrast_mean.png"), dpi=200)
        fig.savefig(str(out_dir / "figureD_per_contrast_mean.pdf"))
        plt.close(fig)


def reproduce(
    run_root: Path,
    out_dir: Path,
    probes: list[str] | None,
    *,
    a_components: int = 5,
    a_symlog: bool = False,
    a_symlog_linthresh: float = 0.0,
) -> None:
    run_root = run_root.resolve()
    out_dir = out_dir.resolve()

    _safe_mkdir(out_dir)

    # Copy sweep summaries for provenance
    for name in ["sweep_summary.txt", "sweep_summary.csv", "sweep_summary.json"]:
        _copy_if_exists(run_root / name, out_dir / name)

    # Figure D is run-level
    _render_figure_d(run_root, out_dir / "figureD")

    available = _list_probes(run_root)
    if probes:
        use = [p for p in probes if p in available]
    else:
        use = available

    for probe in use:
        dst_probe = out_dir / "probes" / probe
        _safe_mkdir(dst_probe)

        # Figure A: render from NPZ if available (ensures consistent visuals even if
        # the original run was created with an older plotting style), otherwise copy.
        a_src = run_root / "figureA_pca_maps" / probe
        rendered = _render_figure_a_from_npz(
            a_src,
            dst_probe,
            max_components=int(a_components),
            symlog=bool(a_symlog),
            symlog_linthresh=float(a_symlog_linthresh),
        )
        if not rendered:
            for f in sorted(a_src.glob("figureA_pca_maps_pc*.png")):
                _copy_if_exists(f, dst_probe / f.name)
            _copy_if_exists(a_src / "figureA_pca_value_distributions.png", dst_probe / "figureA_pca_value_distributions.png")
        _copy_if_exists(a_src / "figureA_pca_components.npz", dst_probe / "figureA_pca_components.npz")
        # copy any available map NPZs for downstream/traceability
        for f in sorted(a_src.glob("figureA_pca_maps_*.npz")):
            _copy_if_exists(f, dst_probe / f.name)
        _copy_if_exists(a_src / "figureA_channel_alignment.npz", dst_probe / "figureA_channel_alignment.npz")

        # Figure B: copy the accuracy plot + summary
        b_src = run_root / "figureB_contrast_id_probe" / probe
        _copy_if_exists(b_src / "figureB_contrast_id_probe_acc.png", dst_probe / "figureB_contrast_id_probe_acc.png")
        _copy_if_exists(b_src / "figureB_contrast_id_probe_summary.txt", dst_probe / "figureB_contrast_id_probe_summary.txt")
        _copy_if_exists(b_src / "figureB_contrast_id_probe_summary.npz", dst_probe / "figureB_contrast_id_probe_summary.npz")

        # Figure C: copy similarity matrices + simmaps
        c_src = run_root / "figureC_alignment" / probe
        for f in sorted(c_src.glob("figureC_similarity_matrix_*.png")):
            _copy_if_exists(f, dst_probe / f.name)
        for f in sorted(c_src.glob("figureC_simmap_*.png")):
            _copy_if_exists(f, dst_probe / f.name)
        for f in sorted(c_src.glob("figureC_similarity_matrix_*.npz")):
            _copy_if_exists(f, dst_probe / f.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_root",
        type=str,
        required=True,
        help="Path to a feature_comparison output directory (contains figureA_pca_maps/, figureB_contrast_id_probe/, ...).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to write paper-ready outputs (copied assets + rendered Figure D plots).",
    )
    p.add_argument(
        "--probes",
        type=str,
        default="",
        help="Comma-separated list of probes to include (default: all available in figureA_pca_maps).",
    )
    p.add_argument(
        "--a_components",
        type=int,
        default=5,
        help="How many PCA components to render for Figure A (default: 5).",
    )
    p.add_argument(
        "--a_symlog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also render signed-log (symlog) versions of Figure A maps + distributions (adds *_symlog.png files).",
    )
    p.add_argument(
        "--a_symlog_linthresh",
        type=float,
        default=0.0,
        help="Symlog linear threshold for Figure A. Use 0 for auto (recommended).",
    )

    args = p.parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    probes = [s.strip() for s in str(args.probes).split(",") if s.strip()] or None

    reproduce(
        run_root=run_root,
        out_dir=out_dir,
        probes=probes,
        a_components=int(args.a_components),
        a_symlog=bool(args.a_symlog),
        a_symlog_linthresh=float(args.a_symlog_linthresh),
    )
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
