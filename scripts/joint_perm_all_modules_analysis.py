#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from ml_recon.pl_modules.pl_UndersampledDataModule import UndersampledDataModule
from ml_recon.pl_modules.pl_learn_ssl_undersampling import LearnedSSLLightning


# ----------------------------
# Paper-facing “Top 5” tests
# ----------------------------
# 1) Output permutation equivariance error (k-space NMSE), aligned back by inverse perm.
# 2) Layer/module representation permutation sensitivity: cosine distance between pooled embeddings.
# 3) Neighborhood mixing score (kNN cross-run neighbor fraction) in embedding space.
# 4) Two-sample distribution shift: MMD (RBF kernel, median heuristic bandwidth).
# 5) Visualization: PCA always; optional UMAP / t-SNE if installed.


DEFAULT_OUT_DIR = Path("./joint_perm_module_probe")

# ----------------------------
# Cluster defaults (user-provided)
# ----------------------------

DEFAULT_CKPT_DIR = Path("/home/anvariha/scratch/kian/kian/mri_machine_learning_reconstruction-main18/checkpoints")
DEFAULT_CKPT_PERM_ON = DEFAULT_CKPT_DIR / (
    "ssl_multi_BRATS_D(3)_ALL(ON)_DIM(OFF)_STEM(DIR)_PERM(ON)_Rvalues[4.0-6.0-8.0]_False_t1,t2,flair,t1ce_"
    "2026-02-21-23:36:57-best-epoch=26.ckpt"
)

DEFAULT_CKPT_PERM_OFF = DEFAULT_CKPT_DIR / (
    "ssl_multi_BRATS_D(3)_ALL(ON)_DIM(OFF)_STEM(DIR)_PERM(OFF)_Rvalues[4.0-6.0-8.0]_False_t1,t2,flair,t1ce_"
    "2026-02-23-21:55:51-best-epoch=32.ckpt"
)

# Default test data directory (user-provided)
DEFAULT_DATA_DIR = Path("/home/anvariha/scratch/kian/sim_256_noise")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x
    # Most of this repo uses complex64 for k-space.
    return x.to(torch.complex64)


def _nmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized MSE per-sample.

    a, b are tensors with batch dim first.
    Returns shape (B,).
    """
    if a.shape != b.shape:
        raise ValueError(f"nmse: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    # Flatten all non-batch dims.
    da = a.reshape(a.shape[0], -1)
    db = b.reshape(b.shape[0], -1)
    # For complex, |x|^2 = x.real^2 + x.imag^2.
    num = (da - db).abs().pow(2).sum(dim=1)
    den = da.abs().pow(2).sum(dim=1).clamp_min(eps)
    return num / den


def _cosine_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Cosine distance per-sample: 1 - cos(u,v). u,v: (B,D)."""
    if u.shape != v.shape:
        raise ValueError(f"cosine_distance: shape mismatch {tuple(u.shape)} vs {tuple(v.shape)}")
    u2 = torch.linalg.norm(u, dim=1).clamp_min(eps)
    v2 = torch.linalg.norm(v, dim=1).clamp_min(eps)
    cos = (u * v).sum(dim=1) / (u2 * v2)
    # Numerical safety.
    cos = cos.clamp(-1.0, 1.0)
    return 1.0 - cos


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 400, seed: int = 0) -> Tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) for the mean via bootstrap."""
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    vals = values.astype(np.float64)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if vals.size == 1:
        return mean, std, mean, mean
    boots = []
    n = int(vals.size)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(vals[idx].mean()))
    boots_arr = np.asarray(boots, dtype=np.float64)
    ci_low, ci_high = np.quantile(boots_arr, [0.025, 0.975])
    return mean, std, float(ci_low), float(ci_high)


class _Reservoir:
    """Reservoir sampler for embeddings with a fixed capacity."""

    def __init__(self, capacity: int, dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.n_seen = 0
        self.x = np.zeros((0, self.dim), dtype=np.float32)
        self.run = np.zeros((0,), dtype=np.int8)  # 0=orig, 1=perm

    def add(self, x: np.ndarray, run: int) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Reservoir.add: expected (N,{self.dim}), got {x.shape}")
        run_arr = np.full((x.shape[0],), int(run), dtype=np.int8)
        for i in range(x.shape[0]):
            self.n_seen += 1
            if self.x.shape[0] < self.capacity:
                self.x = np.vstack([self.x, x[i : i + 1].astype(np.float32)])
                self.run = np.concatenate([self.run, run_arr[i : i + 1]])
                continue
            j = int(self._rng.integers(0, self.n_seen))
            if j < self.capacity:
                self.x[j] = x[i].astype(np.float32)
                self.run[j] = run_arr[i]


def _knn_mixing_score(X: np.ndarray, run: np.ndarray, *, k: int = 10) -> float:
    """kNN mixing: fraction of neighbors from the opposite run, averaged over points.

    X: (N,D), run: (N,) in {0,1}
    Uses cosine similarity with brute-force dot products (after normalization).
    """
    if X.ndim != 2 or run.ndim != 1 or X.shape[0] != run.shape[0]:
        raise ValueError("knn_mixing_score: shape mismatch")
    n = int(X.shape[0])
    if n < 3:
        return float("nan")
    k = int(min(max(1, k), n - 1))
    Xf = X.astype(np.float64)
    denom = np.linalg.norm(Xf, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    Xn = Xf / denom
    # Cosine sim matrix.
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    # top-k neighbors by similarity.
    # Use argpartition for speed.
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    # mixing: neighbors with opposite run label
    run_i = run[:, None]
    mix = (run[idx] != run_i).mean(axis=1)
    return float(mix.mean())


def _mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased MMD^2 with RBF kernel; sigma via median heuristic."""
    if X.size == 0 or Y.size == 0:
        return float("nan")
    Xf = X.astype(np.float64)
    Yf = Y.astype(np.float64)
    n = Xf.shape[0]
    m = Yf.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    # Median heuristic on a subset.
    Z = np.concatenate([Xf, Yf], axis=0)
    # Take up to 256 points to estimate median distance.
    take = min(256, Z.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(Z.shape[0], size=take, replace=False)
    Zs = Z[idx]
    # Pairwise squared distances
    G = (Zs * Zs).sum(axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (Zs @ Zs.T)
    # Use upper triangle excluding diag.
    iu = np.triu_indices(D2.shape[0], k=1)
    med = np.median(np.maximum(D2[iu], 0.0))
    if not np.isfinite(med) or med <= 1e-12:
        med = 1.0
    sigma2 = med

    def k_rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        GA = (A * A).sum(axis=1, keepdims=True)
        GB = (B * B).sum(axis=1, keepdims=True)
        D2ab = GA + GB.T - 2.0 * (A @ B.T)
        D2ab = np.maximum(D2ab, 0.0)
        return np.exp(-D2ab / (2.0 * sigma2))

    Kxx = k_rbf(Xf, Xf)
    Kyy = k_rbf(Yf, Yf)
    Kxy = k_rbf(Xf, Yf)

    # Unbiased estimators: exclude diagonals for Kxx, Kyy
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = (Kxx.sum() / (n * (n - 1))) + (Kyy.sum() / (m * (m - 1))) - (2.0 * Kxy.mean())
    return float(mmd2)


def _try_import_viz():
    """Return (umap_module_or_None, tsne_class_or_None)."""
    umap_mod = None
    tsne_cls = None
    try:
        import umap  # type: ignore

        umap_mod = umap
    except Exception:
        umap_mod = None
    try:
        from sklearn.manifold import TSNE  # type: ignore

        tsne_cls = TSNE
    except Exception:
        tsne_cls = None
    return umap_mod, tsne_cls


def _pca_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("PCA expects 2D")
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    C = Xc @ comps.T
    return C.astype(np.float32)


def _save_scatter(
    *,
    out_path: Path,
    coords: np.ndarray,
    run: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5.2, 4.4))
        c = np.asarray(run, dtype=np.int64)
        plt.scatter(coords[:, 0], coords[:, 1], c=c, s=4, alpha=0.55, cmap="coolwarm")
        plt.title(title)
        plt.xlabel("dim1")
        plt.ylabel("dim2")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return


@dataclass
class _ModelRunResult:
    module_order: List[str]
    # per-module per-sample distances (orig vs perm): list of floats
    inv_dist: Dict[str, List[float]]
    # output equivariance nmse per sample
    out_nmse: List[float]
    # reservoirs for mixing/MMD/viz (bounded)
    reservoirs: Dict[str, _Reservoir]


def _is_stem_name(name: str) -> bool:
    lname = name.lower()
    if lname == "feat_extract" or lname.startswith("feat_extract."):
        return True
    if "stem" in lname:
        return True
    return False


def _select_modules(backbone: torch.nn.Module, *, exclude_stem: bool = True) -> List[Tuple[str, torch.nn.Module]]:
    selected: List[Tuple[str, torch.nn.Module]] = []
    for name, mod in backbone.named_modules():
        if name == "":
            continue
        if exclude_stem and _is_stem_name(name):
            continue
        selected.append((name, mod))
    return selected


class _HookBank:
    def __init__(self, modules: Sequence[Tuple[str, torch.nn.Module]], *, pool: str = "mean") -> None:
        self.modules = list(modules)
        self.pool = str(pool)
        self._handles: List[Any] = []
        self.current: Dict[str, torch.Tensor] = {}

    def _pool(self, t: torch.Tensor) -> Optional[torch.Tensor]:
        # We only use spatial feature maps (B,C,H,W) with H,W>1.
        if t.ndim != 4:
            return None
        if int(t.shape[-1]) <= 1 or int(t.shape[-2]) <= 1:
            return None
        if self.pool == "mean":
            return t.mean(dim=(-2, -1))
        if self.pool == "absmean":
            return t.abs().mean(dim=(-2, -1))
        if self.pool == "center":
            h = int(t.shape[-2])
            w = int(t.shape[-1])
            return t[..., h // 2, w // 2]
        return t.mean(dim=(-2, -1))

    def install(self) -> None:
        self.remove()

        def _hook(name: str):
            def fn(_m: torch.nn.Module, _inp: Tuple[Any, ...], out: Any) -> None:
                x = out
                if isinstance(x, (tuple, list)) and x:
                    x = x[0]
                if not isinstance(x, torch.Tensor):
                    return
                pooled = self._pool(x)
                if pooled is None:
                    return
                self.current[name] = pooled.detach()

            return fn

        for name, mod in self.modules:
            self._handles.append(mod.register_forward_hook(_hook(name)))

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def _load_datamodule(
    *,
    ckpt_path: Path,
    data_dir: Path,
    sampling_method: str,
    R: float,
    batch_size: int,
    num_workers: int,
) -> UndersampledDataModule:
    dm = UndersampledDataModule.load_from_checkpoint(
        ckpt_path.as_posix(),
        data_dir=str(data_dir),
        test_dir=str(data_dir),
        sampling_method=str(sampling_method),
        R=float(R),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    dm.setup("test")
    return dm


def _load_model(*, ckpt_path: Path, device: torch.device) -> LearnedSSLLightning:
    model = LearnedSSLLightning.load_from_checkpoint(
        ckpt_path.as_posix(),
        lr=1e-3,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def _get_backbone(module: LearnedSSLLightning, cascade_idx: int = 0) -> torch.nn.Module:
    cascades = module.recon_model.recon_model.cascades  # type: ignore[attr-defined]
    n = len(cascades)
    idx = int(cascade_idx)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise IndexError(f"cascade_idx out of range: {cascade_idx} for {n} cascades")
    return cascades[idx].model


def _num_cascades(module: LearnedSSLLightning) -> int:
    cascades = module.recon_model.recon_model.cascades  # type: ignore[attr-defined]
    return int(len(cascades))


def _parse_cascade_spec(spec: str, *, n_cascades: int) -> List[int]:
    """Parse cascade selector from CLI.

    Supported:
      - "all"
      - "0" (single int)
      - "0,1,2" (comma-separated)
      - "-1" (last cascade)
    """
    s = str(spec).strip().lower()
    if s == "all":
        return list(range(int(n_cascades)))

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("cascade_idx is empty")

    out: List[int] = []
    for p in parts:
        try:
            idx = int(p)
        except Exception as e:
            raise ValueError(
                f"Invalid --cascade_idx '{spec}'. Use an int, a comma-list like '0,1,2', or 'all'."
            ) from e
        if idx < 0:
            idx = int(n_cascades) + idx
        if idx < 0 or idx >= int(n_cascades):
            raise ValueError(f"cascade_idx out of range: {p} for {n_cascades} cascades")
        out.append(idx)

    # De-duplicate but preserve order.
    seen: set[int] = set()
    uniq: List[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    # permute along contrast axis (dim=1)
    return x.index_select(1, perm)


def _inv_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


@torch.no_grad()
def _run_model_joint_perm(
    *,
    model: LearnedSSLLightning,
    backbone: torch.nn.Module,
    dataloader,
    model_tag: str,
    out_dir: Path,
    max_batches: Optional[int],
    n_perm: int,
    perm_seed: int,
    exclude_stem: bool,
    pool: str,
    reservoir_cap: int,
) -> _ModelRunResult:
    device = next(model.parameters()).device
    modules = _select_modules(backbone, exclude_stem=bool(exclude_stem))
    module_order = [n for n, _ in modules]
    hookbank = _HookBank(modules, pool=pool)
    hookbank.install()

    inv_dist: Dict[str, List[float]] = {n: [] for n in module_order}
    out_nmse: List[float] = []
    reservoirs: Dict[str, _Reservoir] = {}

    rng = np.random.default_rng(int(perm_seed))

    pbar = tqdm(dataloader, desc=f"joint-perm probe: {model_tag}")
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        # Some pipelines wrap the batch (e.g., [batch, extra] or (batch, extra)).
        # We only care about the dict-like first item.
        if isinstance(batch, (list, tuple)) and batch:
            batch = batch[0]

        if not isinstance(batch, dict):
            raise TypeError(
                "Expected batch to be a dict with keys {'undersampled','mask','fs_k_space'}, "
                f"got type={type(batch)!r}. If your dataloader returns multiple dataloaders, "
                "ensure we selected the first one in main()."
            )

        undersampled = _as_complex(batch["undersampled"].to(device))
        mask = batch["mask"].to(device)
        fs_k = _as_complex(batch["fs_k_space"].to(device))

        B = int(undersampled.shape[0])
        K = int(undersampled.shape[1])
        if K < 2:
            continue

        # Original forward once.
        hookbank.current = {}
        y0 = model.forward(undersampled, mask, fs_k)
        y0 = _as_complex(y0)
        orig_emb = {k: v for k, v in hookbank.current.items()}

        # Initialize reservoirs lazily on first observation (dim can vary per module).
        for name, emb in orig_emb.items():
            if name not in reservoirs:
                reservoirs[name] = _Reservoir(reservoir_cap, int(emb.shape[1]), seed=0)
            reservoirs[name].add(emb.detach().cpu().numpy(), run=0)

        # Permuted forwards.
        for _p in range(int(n_perm)):
            perm_np = rng.permutation(K)
            perm = torch.tensor(perm_np, device=device, dtype=torch.long)
            inv = _inv_perm(perm)

            u_p = _apply_perm(undersampled, perm)
            m_p = _apply_perm(mask, perm)
            f_p = _apply_perm(fs_k, perm)

            hookbank.current = {}
            y_p = model.forward(u_p, m_p, f_p)
            y_p = _as_complex(y_p)
            perm_emb = {k: v for k, v in hookbank.current.items()}

            # Output equivariance: unpermute along contrast axis if possible.
            # We assume contrast axis is dim=1 as in the input.
            try:
                y_p_unperm = y_p.index_select(1, inv)
                nmse_b = _nmse(y0, y_p_unperm)
                out_nmse.extend([float(x) for x in nmse_b.detach().cpu().numpy().tolist()])
            except Exception:
                # If output is not contrast-indexable, skip.
                pass

            # Per-module invariance distances.
            for name in module_order:
                a = orig_emb.get(name)
                b = perm_emb.get(name)
                if a is None or b is None:
                    continue
                if a.shape != b.shape:
                    continue
                d = _cosine_distance(a, b)
                inv_dist[name].extend([float(x) for x in d.detach().cpu().numpy().tolist()])

                # Reservoir for distribution-level metrics.
                if name in reservoirs:
                    reservoirs[name].add(b.detach().cpu().numpy(), run=1)

    hookbank.remove()

    # Persist a module list for reproducibility.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"modules_{model_tag}.txt").write_text("\n".join(module_order) + "\n")
    return _ModelRunResult(module_order=module_order, inv_dist=inv_dist, out_nmse=out_nmse, reservoirs=reservoirs)


def _metric_rows_for_model(
    *,
    result: _ModelRunResult,
    model_tag: str,
    bootstrap: int,
    boot_seed: int,
    knn_k: int,
    viz: str,
    viz_modules: List[str],
    viz_seeds: int,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Test 1: output equivariance nmse
    nmse_vals = np.asarray(result.out_nmse, dtype=np.float64)
    mean, std, lo, hi = _bootstrap_ci(nmse_vals, n_boot=bootstrap, seed=boot_seed)
    rows.append(
        {
            "module": "__output__",
            "model": model_tag,
            "metric": "output_equiv_nmse",
            "mean": mean,
            "std": std,
            "ci_low": lo,
            "ci_high": hi,
            "n": int(nmse_vals.size),
        }
    )

    # Per-module metrics (Tests 2-4)
    for name in result.module_order:
        vals = np.asarray(result.inv_dist.get(name, []), dtype=np.float64)
        mean, std, lo, hi = _bootstrap_ci(vals, n_boot=bootstrap, seed=boot_seed)
        rows.append(
            {
                "module": name,
                "model": model_tag,
                "metric": "repr_perm_cosdist",
                "mean": mean,
                "std": std,
                "ci_low": lo,
                "ci_high": hi,
                "n": int(vals.size),
            }
        )

        res = result.reservoirs.get(name)
        if res is None or res.x.shape[0] < 10:
            continue

        X = res.x
        run = res.run
        # Split orig vs perm
        X0 = X[run == 0]
        X1 = X[run == 1]

        # Test 3: kNN mixing
        mix = _knn_mixing_score(X, run, k=knn_k)
        rows.append(
            {
                "module": name,
                "model": model_tag,
                "metric": f"knn_mixing_k{knn_k}",
                "mean": float(mix),
                "std": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": int(X.shape[0]),
            }
        )

        # Test 4: MMD shift
        # Use balanced subsets if possible
        n = min(X0.shape[0], X1.shape[0])
        if n >= 10:
            rng = np.random.default_rng(0)
            idx0 = rng.choice(X0.shape[0], size=n, replace=False)
            idx1 = rng.choice(X1.shape[0], size=n, replace=False)
            mmd2 = _mmd_rbf(X0[idx0], X1[idx1])
            rows.append(
                {
                    "module": name,
                    "model": model_tag,
                    "metric": "mmd2_rbf",
                    "mean": float(mmd2),
                    "std": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n": int(2 * n),
                }
            )

        # Test 5: PCA + optional UMAP/tSNE visuals for selected modules
        if viz != "none" and name in viz_modules:
            coords = _pca_2d(X)
            _save_scatter(
                out_path=out_dir / "viz" / model_tag / f"pca_{_safe_name(name)}.png",
                coords=coords,
                run=run,
                title=f"PCA: {model_tag} :: {name}",
            )

    # Optional UMAP/tSNE for selected modules
    viz = str(viz).lower().strip()
    if viz in {"umap", "tsne"} and viz_modules:
        umap_mod, tsne_cls = _try_import_viz()
        if viz == "umap" and umap_mod is None:
            print("[viz] umap-learn not installed; skipping UMAP.")
        if viz == "tsne" and tsne_cls is None:
            print("[viz] scikit-learn not installed; skipping t-SNE.")

        for name in viz_modules:
            res = result.reservoirs.get(name)
            if res is None or res.x.shape[0] < 50:
                continue
            X = res.x.astype(np.float32)
            run = res.run
            # Limit to keep viz reasonable.
            take = min(800, X.shape[0])
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=take, replace=False)
            Xs = X[idx]
            runs = run[idx]

            # Always save PCA.
            coords = _pca_2d(Xs)
            _save_scatter(
                out_path=out_dir / "viz" / model_tag / f"pca_{_safe_name(name)}_sub.png",
                coords=coords,
                run=runs,
                title=f"PCA(sub): {model_tag} :: {name}",
            )

            for seed in range(int(viz_seeds)):
                if viz == "umap" and umap_mod is not None:
                    reducer = umap_mod.UMAP(
                        n_neighbors=15,
                        min_dist=0.1,
                        metric="cosine",
                        random_state=int(seed),
                    )
                    C = reducer.fit_transform(Xs)
                    _save_scatter(
                        out_path=out_dir / "viz" / model_tag / f"umap_s{seed}_{_safe_name(name)}.png",
                        coords=C,
                        run=runs,
                        title=f"UMAP s={seed}: {model_tag} :: {name}",
                    )
                if viz == "tsne" and tsne_cls is not None:
                    reducer = tsne_cls(
                        n_components=2,
                        perplexity=30.0,
                        init="pca",
                        learning_rate="auto",
                        random_state=int(seed),
                    )
                    C = reducer.fit_transform(Xs)
                    _save_scatter(
                        out_path=out_dir / "viz" / model_tag / f"tsne_s{seed}_{_safe_name(name)}.png",
                        coords=C,
                        run=runs,
                        title=f"t-SNE s={seed}: {model_tag} :: {name}",
                    )

    return rows


def _safe_name(name: str) -> str:
    # filesystem-friendly
    return name.replace("/", "_").replace(".", "_").replace(":", "_")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.8g}")
            else:
                s = str(v)
                # basic CSV escaping for commas
                if "," in s or "\n" in s:
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _expected_direction(metric: str) -> str:
    """Paper expectation direction for PERM(ON) vs PERM(OFF).

    Returns a string describing what “better” looks like for PERM(ON).
    """
    m = str(metric)
    if m == "output_equiv_nmse":
        return "lower_is_better_for_perm_on"
    if m == "repr_perm_cosdist":
        return "lower_is_better_for_perm_on"
    if m == "mmd2_rbf":
        return "lower_is_better_for_perm_on"
    if m.startswith("knn_mixing_"):
        return "higher_is_better_for_perm_on"
    return "unknown"


def _compare_and_write_summaries(*, out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Create analysis-friendly summary files from the long-form metrics table."""
    # Index rows by (module, metric, model)
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in rows:
        module = str(r.get("module", ""))
        metric = str(r.get("metric", ""))
        model = str(r.get("model", ""))
        if not module or not metric or not model:
            continue
        index[(module, metric, model)] = r

    # Collect comparable pairs.
    pairs: List[Dict[str, Any]] = []
    keys = {(m, met) for (m, met, _model) in index.keys()}
    for module, metric in sorted(keys):
        off = index.get((module, metric, "perm_off"))
        on = index.get((module, metric, "perm_on"))
        if off is None or on is None:
            continue
        off_mean = float(off.get("mean", float("nan")))
        on_mean = float(on.get("mean", float("nan")))
        delta = on_mean - off_mean
        pairs.append(
            {
                "module": module,
                "metric": metric,
                "perm_off_mean": off_mean,
                "perm_on_mean": on_mean,
                "delta_on_minus_off": float(delta),
                "expected": _expected_direction(metric),
                "perm_off_n": int(off.get("n", 0) or 0),
                "perm_on_n": int(on.get("n", 0) or 0),
                "perm_off_ci_low": float(off.get("ci_low", float("nan"))),
                "perm_off_ci_high": float(off.get("ci_high", float("nan"))),
                "perm_on_ci_low": float(on.get("ci_low", float("nan"))),
                "perm_on_ci_high": float(on.get("ci_high", float("nan"))),
            }
        )

    # Write a single, analysis-friendly comparison table.
    _write_csv(out_dir / "comparison_long.csv", pairs)

    # Per-metric ranking files (top improvements / worst regressions).
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for r in pairs:
        by_metric.setdefault(str(r["metric"]), []).append(r)

    def score_for_sort(r: Dict[str, Any]) -> float:
        d = float(r.get("delta_on_minus_off", float("nan")))
        if not np.isfinite(d):
            return float("inf")
        exp = str(r.get("expected", ""))
        # We want “best for PERM(ON)” to sort first.
        if exp == "higher_is_better_for_perm_on":
            return -d
        if exp == "lower_is_better_for_perm_on":
            return d
        return float("inf")

    ranking_rows: List[Dict[str, Any]] = []
    for metric, lst in by_metric.items():
        lst_valid = [r for r in lst if np.isfinite(float(r.get("delta_on_minus_off", float("nan"))))]
        lst_sorted_best = sorted(lst_valid, key=score_for_sort)
        lst_sorted_worst = list(reversed(lst_sorted_best))

        top = lst_sorted_best[:50]
        bot = lst_sorted_worst[:50]
        _write_csv(out_dir / f"ranking_best_{metric}.csv", top)
        _write_csv(out_dir / f"ranking_worst_{metric}.csv", bot)

        # Also create a compact combined ranking table.
        for rank, r in enumerate(top, start=1):
            rr = dict(r)
            rr["rank"] = rank
            rr["bucket"] = "best"
            ranking_rows.append(rr)
        for rank, r in enumerate(bot, start=1):
            rr = dict(r)
            rr["rank"] = rank
            rr["bucket"] = "worst"
            ranking_rows.append(rr)

        # Metric-level summary stats.
        deltas = np.asarray([float(r["delta_on_minus_off"]) for r in lst_valid], dtype=np.float64)
        if deltas.size:
            summ = {
                "metric": metric,
                "expected": _expected_direction(metric),
                "n_modules": int(deltas.size),
                "delta_mean": float(deltas.mean()),
                "delta_median": float(np.median(deltas)),
                "delta_p25": float(np.quantile(deltas, 0.25)),
                "delta_p75": float(np.quantile(deltas, 0.75)),
            }
            _write_json(out_dir / f"summary_{metric}.json", summ)

    _write_csv(out_dir / "ranking_compact.csv", ranking_rows)

    # A short human-readable overview.
    overview_lines = [
        "Summary bundle written for joint permutation probe.",
        "",
        "Key files:",
        "  - metrics.csv: raw long-form metrics (per model)",
        "  - comparison_long.csv: PERM(ON) vs PERM(OFF) deltas per module/metric",
        "  - ranking_best_<metric>.csv / ranking_worst_<metric>.csv: top/bottom 50 modules",
        "  - ranking_compact.csv: combined ranking table",
        "  - summary_<metric>.json: per-metric delta distribution summary",
        "",
        "Expected directions (PERM ON vs OFF):",
        "  - output_equiv_nmse: lower",
        "  - repr_perm_cosdist: lower",
        "  - mmd2_rbf: lower",
        "  - knn_mixing_*: higher",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(overview_lines) + "\n")


def _plot_ranked(
    *,
    out_path: Path,
    modules: List[str],
    vals_a: List[float],
    vals_b: List[float],
    tag_a: str,
    tag_b: str,
    top_n: int = 60,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr_a = np.asarray(vals_a, dtype=np.float64)
        arr_b = np.asarray(vals_b, dtype=np.float64)
        diff = arr_b - arr_a
        order = np.argsort(-np.abs(diff))
        order = order[: min(int(top_n), order.size)]
        mods = [modules[i] for i in order.tolist()]
        a = arr_a[order]
        b = arr_b[order]

        x = np.arange(len(mods))
        plt.figure(figsize=(max(10, 0.18 * len(mods)), 4.8))
        plt.plot(x, a, label=tag_a, marker="o", markersize=2)
        plt.plot(x, b, label=tag_b, marker="o", markersize=2)
        plt.xticks(x, mods, rotation=90, fontsize=6)
        plt.ylabel("repr_perm_cosdist (mean)")
        plt.title("Top modules by |Δ invariance|")
        plt.legend()
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
    except Exception:
        return


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Joint-forward (all-contrasts) analysis for PromptMR PERM(ON) vs PERM(OFF). "
            "Computes 5 paper-friendly tests across all non-stem feature modules."
        )
    )
    p.add_argument(
        "--ckpt_perm_off",
        type=str,
        default=str(DEFAULT_CKPT_PERM_OFF),
        help="PromptMR checkpoint trained with PERM(OFF).",
    )
    p.add_argument(
        "--ckpt_perm_on",
        type=str,
        default=str(DEFAULT_CKPT_PERM_ON),
        help="PromptMR checkpoint trained with PERM(ON).",
    )
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--sampling_method", type=str, default="1d", choices=["1d", "2d"])
    p.add_argument("--R", type=float, default=8.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--max_batches",
        type=int,
        default=80,
        help=(
            "Max number of test batches to process. Use 0 (or any negative value) to run all batches."
        ),
    )

    p.add_argument(
        "--cascade_idx",
        type=str,
        default="0",
        help=(
            "Which cascade(s) to probe. Use an int like '0', a comma-separated list like '0,1,2', "
            "a negative index like '-1' (last), or 'all' to run every cascade."
        ),
    )
    p.add_argument(
        "--exclude_stem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude stem/feat_extract modules (default true).",
    )
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "absmean", "center"])

    p.add_argument("--n_perm", type=int, default=3, help="Permutations per slice/batch.")
    p.add_argument("--perm_seed", type=int, default=0)

    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--boot_seed", type=int, default=0)

    p.add_argument("--reservoir_cap", type=int, default=600, help="Max embeddings stored per module for mix/MMD/viz.")
    p.add_argument("--knn_k", type=int, default=10)

    p.add_argument(
        "--viz",
        type=str,
        default="pca",
        choices=["none", "pca", "umap", "tsne"],
        help="Visualization backend: PCA always available; UMAP/tSNE require optional deps.",
    )
    p.add_argument(
        "--viz_top_modules",
        type=int,
        default=12,
        help="How many modules to visualize (ranked by |Δ invariance|).",
    )
    p.add_argument("--viz_seeds", type=int, default=3, help="Number of seeds for UMAP/tSNE (if enabled).")

    args = p.parse_args()

    # Convention: max_batches <= 0 means "no limit".
    max_batches: Optional[int]
    if args.max_batches is None:
        max_batches = None
    else:
        mb = int(args.max_batches)
        max_batches = None if mb <= 0 else mb

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()

    ckpt_off = Path(args.ckpt_perm_off)
    ckpt_on = Path(args.ckpt_perm_on)
    if not ckpt_off.exists():
        raise FileNotFoundError(f"ckpt_perm_off not found: {ckpt_off}")
    if not ckpt_on.exists():
        raise FileNotFoundError(f"ckpt_perm_on not found: {ckpt_on}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # Use PERM(OFF) checkpoint to instantiate the datamodule (same dataset config expected).
    dm = _load_datamodule(
        ckpt_path=ckpt_off,
        data_dir=data_dir,
        sampling_method=str(args.sampling_method),
        R=float(args.R),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    dl = dm.test_dataloader()

    # Lightning DataModules sometimes return a list/tuple/dict of DataLoaders.
    # We only need a single DataLoader here.
    if isinstance(dl, dict):
        # Take first value in key order.
        dl = next(iter(dl.values()))
    if isinstance(dl, (list, tuple)):
        if len(dl) == 0:
            raise RuntimeError("test_dataloader() returned an empty list/tuple")
        dl = dl[0]

    model_off = _load_model(ckpt_path=ckpt_off, device=device)
    model_on = _load_model(ckpt_path=ckpt_on, device=device)

    n_cascades = min(_num_cascades(model_off), _num_cascades(model_on))
    cascade_indices = _parse_cascade_spec(str(args.cascade_idx), n_cascades=n_cascades)

    for cascade_idx in cascade_indices:
        cascade_out_dir = out_dir / f"cascade_{int(cascade_idx)}"
        cascade_out_dir.mkdir(parents=True, exist_ok=True)

        backbone_off = _get_backbone(model_off, cascade_idx=int(cascade_idx))
        backbone_on = _get_backbone(model_on, cascade_idx=int(cascade_idx))

        # Run both models.
        res_off = _run_model_joint_perm(
            model=model_off,
            backbone=backbone_off,
            dataloader=dl,
            model_tag="perm_off",
            out_dir=cascade_out_dir,
            max_batches=max_batches,
            n_perm=int(args.n_perm),
            perm_seed=int(args.perm_seed),
            exclude_stem=bool(args.exclude_stem),
            pool=str(args.pool),
            reservoir_cap=int(args.reservoir_cap),
        )
        res_on = _run_model_joint_perm(
            model=model_on,
            backbone=backbone_on,
            dataloader=dl,
            model_tag="perm_on",
            out_dir=cascade_out_dir,
            max_batches=max_batches,
            n_perm=int(args.n_perm),
            perm_seed=int(args.perm_seed),
            exclude_stem=bool(args.exclude_stem),
            pool=str(args.pool),
            reservoir_cap=int(args.reservoir_cap),
        )

        # Determine visualization module set: top by |Δ invariance|.
        common = [m for m in res_off.module_order if m in set(res_on.module_order)]
        mean_off = []
        mean_on = []
        for m in common:
            vo = np.asarray(res_off.inv_dist.get(m, []), dtype=np.float64)
            vn = np.asarray(res_on.inv_dist.get(m, []), dtype=np.float64)
            mean_off.append(float(np.nanmean(vo) if vo.size else np.nan))
            mean_on.append(float(np.nanmean(vn) if vn.size else np.nan))
        arr_off = np.asarray(mean_off, dtype=np.float64)
        arr_on = np.asarray(mean_on, dtype=np.float64)
        diff = np.abs(arr_on - arr_off)
        # handle NaNs
        diff = np.where(np.isfinite(diff), diff, -np.inf)
        order = np.argsort(-diff)
        topk = int(min(max(0, int(args.viz_top_modules)), order.size))
        viz_modules = [common[i] for i in order[:topk].tolist() if diff[i] != -np.inf]

        rows = []
        rows.extend(
            _metric_rows_for_model(
                result=res_off,
                model_tag="perm_off",
                bootstrap=int(args.bootstrap),
                boot_seed=int(args.boot_seed),
                knn_k=int(args.knn_k),
                viz=str(args.viz),
                viz_modules=viz_modules,
                viz_seeds=int(args.viz_seeds),
                out_dir=cascade_out_dir,
            )
        )
        rows.extend(
            _metric_rows_for_model(
                result=res_on,
                model_tag="perm_on",
                bootstrap=int(args.bootstrap),
                boot_seed=int(args.boot_seed),
                knn_k=int(args.knn_k),
                viz=str(args.viz),
                viz_modules=viz_modules,
                viz_seeds=int(args.viz_seeds),
                out_dir=cascade_out_dir,
            )
        )

        _write_csv(cascade_out_dir / "metrics.csv", rows)

        # Save config so you can reproduce exactly.
        _write_json(
            cascade_out_dir / "config.json",
            {
                "ckpt_perm_off": str(args.ckpt_perm_off),
                "ckpt_perm_on": str(args.ckpt_perm_on),
                "data_dir": str(args.data_dir),
                "sampling_method": str(args.sampling_method),
                "R": float(args.R),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "max_batches": max_batches,
                "cascade_idx": int(cascade_idx),
                "exclude_stem": bool(args.exclude_stem),
                "pool": str(args.pool),
                "n_perm": int(args.n_perm),
                "perm_seed": int(args.perm_seed),
                "bootstrap": int(args.bootstrap),
                "boot_seed": int(args.boot_seed),
                "reservoir_cap": int(args.reservoir_cap),
                "knn_k": int(args.knn_k),
                "viz": str(args.viz),
                "viz_top_modules": int(args.viz_top_modules),
                "viz_seeds": int(args.viz_seeds),
            },
        )

        # Create a compact “results bundle” for offline analysis.
        _compare_and_write_summaries(out_dir=cascade_out_dir, rows=rows)

        # Plot top modules by invariance difference.
        _plot_ranked(
            out_path=cascade_out_dir / "invariance_top_modules.png",
            modules=common,
            vals_a=mean_off,
            vals_b=mean_on,
            tag_a="perm_off",
            tag_b="perm_on",
            top_n=80,
        )

        # Save a short README of what was computed.
        (cascade_out_dir / "README.txt").write_text(
            "\n".join(
                [
                    "Joint-forward all-contrast permutation probe",
                    "",
                    f"cascade_idx={int(cascade_idx)}",
                    "",
                    "Computed tests:",
                    "  1) output_equiv_nmse: NMSE(y, unperm(y_perm)) in k-space",
                    "  2) repr_perm_cosdist: cosine distance between pooled module outputs (orig vs perm)",
                    "  3) knn_mixing_kK: kNN cross-run neighbor fraction (higher => more overlap)",
                    "  4) mmd2_rbf: MMD^2 between orig vs perm embedding distributions",
                    "  5) viz: PCA always; optional UMAP/tSNE if deps installed",
                    "",
                    f"exclude_stem={args.exclude_stem}",
                    f"pool={args.pool}",
                    f"n_perm={args.n_perm}",
                    f"max_batches={args.max_batches}",
                    f"reservoir_cap={args.reservoir_cap}",
                    f"knn_k={args.knn_k}",
                    f"viz={args.viz}",
                    f"viz_modules={len(viz_modules)}",
                ]
            )
            + "\n"
        )

        print("\nSaved results to:", cascade_out_dir)
        print("- metrics:", (cascade_out_dir / "metrics.csv").as_posix())
        print("- comparison:", (cascade_out_dir / "comparison_long.csv").as_posix())
        print("- summary:", (cascade_out_dir / "SUMMARY.txt").as_posix())
        print("- ranked invariance plot:", (cascade_out_dir / "invariance_top_modules.png").as_posix())
        if viz_modules:
            print("- viz modules (top |Δ|):")
            for m in viz_modules:
                print("  -", m)


if __name__ == "__main__":
    # Avoid tokenizer parallelism warnings in some environments.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
