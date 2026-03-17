#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

# Ensure repo root (parent of scripts/) is importable when running
# `python scripts/extract_promptmr_stem_attention.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import contrast_feature_probe as cfp


DEFAULT_DATA_DIR = "/home/anvariha/scratch/kian/sim_256_noise"
DEFAULT_CKPT = (
    "/home/anvariha/scratch/kian/kian/mri_machine_learning_reconstruction-main18/checkpoints/"
    "ssl_multi_BRATS_D(3)_ALL(ON)_DIM(OFF)_STEM(DIR)_PERM(ON)_Rvalues[4.0-6.0-8.0]_False_"
    "t1,t2,flair,t1ce_2026-02-21-23:36:57-best-epoch=26.ckpt"
)


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _normalize_dataloader(dl: Any):
    if isinstance(dl, (list, tuple)):
        if not dl:
            raise RuntimeError("test_dataloader() returned an empty list")
        return dl[0]
    if isinstance(dl, dict):
        if not dl:
            raise RuntimeError("test_dataloader() returned an empty dict")
        return next(iter(dl.values()))
    return dl


def _resolve_contrast_order(model) -> list[str]:
    contrast_order = [str(c) for c in getattr(model, "contrast_order", [])]
    if contrast_order:
        return contrast_order

    try:
        contrast_order = [
            str(c)
            for c in model.hparams.get("varnet_config", {}).get("contrast_order", [])  # type: ignore[attr-defined]
        ]
    except Exception:
        contrast_order = []

    if not contrast_order:
        raise ValueError("Could not determine contrast_order from checkpoint")
    return contrast_order


@torch.no_grad()
def _run_one_batch(*, model, dataloader, batch_index: int, device: torch.device):
    for i, batch in enumerate(dataloader):
        if i != int(batch_index):
            continue

        undersampled = batch["undersampled"].to(device)
        mask = batch["mask"].to(device)
        fs_k = batch["fs_k_space"].to(device)

        if not undersampled.is_complex():
            undersampled = undersampled.to(torch.complex64)
        if not fs_k.is_complex():
            fs_k = fs_k.to(torch.complex64)

        _ = model.forward(undersampled, mask, fs_k)
        return int(undersampled.shape[0])

    raise IndexError(f"batch_index={batch_index} out of range")


def _save_attn_heatmap(
    *,
    out_path: Path,
    attn: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required to render attention heatmaps")

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.2))
    im = ax.imshow(attn, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=200)
    plt.close(fig)


def _summarize_attn(
    *,
    attn: np.ndarray,
    labels: list[str],
    gates: np.ndarray | None,
    tag: str,
    out_dir: Path,
    topk: int,
) -> None:
    """Print and save a human-readable summary of attention.

    Args:
        attn: [B, K, K] attention matrix (row-softmax over keys).
        labels: length-K contrast labels.
        gates: optional gate values, usually scalar or [B].
        tag: used in filenames and printed headers.
        topk: number of strongest keys to show per query row.
    """
    b, k1, k2 = attn.shape
    if k1 != k2:
        raise ValueError(f"Expected square attn [B,K,K], got {attn.shape}")
    if len(labels) != k1:
        raise ValueError(f"labels length {len(labels)} != K={k1}")

    topk_eff = int(topk)
    if topk_eff <= 0:
        topk_eff = k1
    topk_eff = min(topk_eff, k1)

    csv_path = out_dir / f"{tag}_attn_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "query", "key", "weight", "gate"])
        for sample_idx in range(b):
            gate_val = None
            if gates is not None:
                try:
                    if gates.ndim == 0:
                        gate_val = float(gates)
                    elif gates.ndim == 1 and gates.shape[0] == b:
                        gate_val = float(gates[sample_idx])
                    else:
                        gate_val = float(np.asarray(gates).ravel()[0])
                except Exception:
                    gate_val = None

            for qi, qname in enumerate(labels):
                row = attn[sample_idx, qi]
                order = np.argsort(-row)
                for j in order[:topk_eff]:
                    w.writerow(
                        [
                            int(sample_idx),
                            str(qname),
                            str(labels[int(j)]),
                            float(row[int(j)]),
                            "" if gate_val is None else float(gate_val),
                        ]
                    )

    print(f"\n=== {tag}: per-sample attention summary ===")
    if gates is not None:
        try:
            if gates.ndim == 0:
                print(f"gate={float(gates):.4f}")
            elif gates.ndim == 1 and gates.shape[0] == b:
                gates_str = ", ".join([f"{float(g):.4f}" for g in gates])
                print(f"gate_per_sample=[{gates_str}]")
            else:
                print(f"gate_shape={tuple(gates.shape)}")
        except Exception:
            print("gate=<unprintable>")

    for sample_idx in range(b):
        print(f"\nSample {sample_idx}:")
        for qi, qname in enumerate(labels):
            row = attn[sample_idx, qi]
            order = np.argsort(-row)[:topk_eff]
            pairs = ", ".join([f"{labels[int(j)]}={float(row[int(j)]):.3f}" for j in order])
            print(f"  {qname} -> {pairs}")
    print(f"Wrote: {csv_path}")


def _print_attn_matrix(*, tag: str, labels: list[str], attn: np.ndarray, gate: np.ndarray | None) -> None:
    """Pretty-print a [K,K] attention matrix with row/col labels."""
    k = int(attn.shape[0])
    if attn.shape != (k, k):
        raise ValueError(f"Expected [K,K], got {attn.shape}")
    if len(labels) != k:
        raise ValueError(f"labels length {len(labels)} != K={k}")

    if gate is not None:
        try:
            gate_val = float(np.asarray(gate).ravel()[0])
            print(f"{tag} gate={gate_val:.4f}")
        except Exception:
            print(f"{tag} gate=<unprintable>")

    # Header
    header = " " * 10 + " ".join([f"{s:>10s}" for s in labels])
    print(header)
    for i, row_name in enumerate(labels):
        row = " ".join([f"{float(attn[i, j]):10.3f}" for j in range(k)])
        print(f"{row_name:>10s} {row}")


def _extract_stem_attn_from_backbone(
    backbone: Any,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return (cab_mean[B,K,K], cab_gate, mix[B,K,K], mix_gate, cab_full[B,H,K,K]) if available."""
    stem = getattr(backbone, "feat_extract", None)
    if stem is None:
        raise ValueError("Backbone has no feat_extract; is this a PromptMRUNet checkpoint?")

    cab_attn = None
    cab_gate = None
    try:
        cab_attn = stem.cross_cab.attn.last_attn  # type: ignore[attr-defined]
        cab_gate = stem.cross_cab.attn.last_gate  # type: ignore[attr-defined]
    except Exception:
        cab_attn = None
        cab_gate = None

    mix_attn = None
    mix_gate = None
    try:
        mix_attn = stem.stem_mix.last_attn  # type: ignore[attr-defined]
        mix_gate = stem.stem_mix.last_gate  # type: ignore[attr-defined]
    except Exception:
        mix_attn = None
        mix_gate = None

    cab_mean = None
    cab_full = None
    if isinstance(cab_attn, torch.Tensor):
        cab = cab_attn.detach().cpu().numpy()  # [B, heads, K, K]
        cab_full = cab
        cab_mean = cab.mean(axis=1)  # [B, K, K]

    cab_gate_np = cab_gate.detach().cpu().numpy() if isinstance(cab_gate, torch.Tensor) else None
    mix_np = mix_attn.detach().cpu().numpy() if isinstance(mix_attn, torch.Tensor) else None
    mix_gate_np = mix_gate.detach().cpu().numpy() if isinstance(mix_gate, torch.Tensor) else None
    return cab_mean, cab_gate_np, mix_np, mix_gate_np, cab_full


def _attn_nonuniform_metrics(attn_kk: np.ndarray) -> dict[str, float]:
    """Compute simple, paper-friendly non-uniformity metrics for a single [K,K] row-stochastic matrix.

    Returns:
      - l1_from_uniform: mean_row sum_j |a_ij - 1/K|
      - max_abs_from_uniform: max_{i,j} |a_ij - 1/K|
      - kl_from_uniform: mean_row KL(row || uniform)
      - one_minus_entropy: mean_row (1 - H(row)/log K)
    """
    A = np.asarray(attn_kk, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected [K,K], got {A.shape}")
    k = int(A.shape[0])
    if k < 1:
        raise ValueError("K must be >= 1")

    u = 1.0 / float(k)
    l1 = float(np.mean(np.sum(np.abs(A - u), axis=1)))
    max_abs = float(np.max(np.abs(A - u)))

    eps = 1e-12
    P = np.clip(A, eps, 1.0)
    # KL(row || uniform) = sum p log(p/u)
    kl = float(np.mean(np.sum(P * (np.log(P) - np.log(u)), axis=1)))
    # Normalized entropy in [0,1]
    ent = -np.sum(P * np.log(P), axis=1)
    ent_norm = ent / max(np.log(float(k)), eps)
    one_minus_ent = float(np.mean(1.0 - ent_norm))

    return {
        "l1_from_uniform": l1,
        "max_abs_from_uniform": max_abs,
        "kl_from_uniform": kl,
        "one_minus_entropy": one_minus_ent,
    }


def _score_for_ranking(attn_kk: np.ndarray) -> float:
    """Single scalar score used to rank 'most non-uniform' examples."""
    m = _attn_nonuniform_metrics(attn_kk)
    # Empirically stable: use max deviation (easy to explain) then entropy.
    return float(m["max_abs_from_uniform"] + 0.25 * m["one_minus_entropy"])


def _unique_permutations(*, k: int, n: int, seed: int) -> list[np.ndarray]:
    """Generate n unique permutations of range(k). Includes identity first."""
    if k < 1:
        raise ValueError("k must be >= 1")
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.RandomState(int(seed))

    perms: list[np.ndarray] = [np.arange(k, dtype=np.int64)]
    seen = {tuple(perms[0].tolist())}
    while len(perms) < n:
        p = rng.permutation(k).astype(np.int64)
        t = tuple(p.tolist())
        if t in seen:
            continue
        seen.add(t)
        perms.append(p)
    return perms


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Extract PromptMR stem cross-contrast attention matrices (cached as last_attn) "
            "from a checkpoint by running a single forward pass." 
        )
    )

    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to a Lightning .ckpt")
    p.add_argument("--out_dir", type=str, default="./stem_attention_outputs")

    # Data
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--sampling_method", type=str, default="1d", choices=["1d", "2d"])
    p.add_argument("--R", type=float, default=8.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    # Which forward pass / inspection target
    p.add_argument(
        "--cascade_idx",
        type=int,
        default=-1,
        help="Which VarNet cascade's backbone to INSPECT for cached stem attention. -1 = last cascade.",
    )
    p.add_argument(
        "--permute_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, run multiple contrast permutations per sample and print attention matrices.",
    )
    p.add_argument(
        "--scan_nonuniform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If True, scan many test samples (no permutations) and quantify how non-uniform the stem attention is; "
            "writes a CSV and saves heatmaps for the most non-uniform cases."
        ),
    )
    p.add_argument(
        "--scan_samples",
        type=int,
        default=300,
        help="How many test samples to scan when --scan_nonuniform is enabled.",
    )
    p.add_argument(
        "--save_top",
        type=int,
        default=8,
        help="How many top non-uniform samples to save as heatmaps in scan mode.",
    )
    p.add_argument(
        "--cab_head",
        type=int,
        default=-1,
        help="If >=0, print that CAB head instead of the mean-over-heads in permute mode.",
    )
    p.add_argument("--num_samples", type=int, default=5, help="How many test samples to evaluate.")
    p.add_argument("--num_perms", type=int, default=4, help="How many permutations to run per sample (includes identity).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--topk",
        type=int,
        default=4,
        help="How many strongest keys to show per query row in the printed summary (<=0 means all).",
    )

    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfp._device()  # type: ignore[attr-defined]
    model = cfp._load_model(ckpt_path=ckpt_path, device=device)  # type: ignore[attr-defined]

    dm = cfp._load_datamodule(  # type: ignore[attr-defined]
        ckpt_path=ckpt_path,
        data_dir=data_dir,
        sampling_method=str(args.sampling_method),
        R=float(args.R),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    dl = _normalize_dataloader(dm.test_dataloader())

    contrast_order = _resolve_contrast_order(model)
    backbone = cfp._get_backbone(model, cascade_idx=int(args.cascade_idx))  # type: ignore[attr-defined]

    if bool(args.scan_nonuniform):
        target_samples = int(args.scan_samples)
        if target_samples < 1:
            raise ValueError("--scan_samples must be >= 1")
        save_top = max(0, int(args.save_top))

        k = len(contrast_order)
        # Keep a small list of the best examples (score, sample_id, cab_mean, mix, cab_best_head, head_idx).
        top: list[tuple[float, int, np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None]] = []

        rows: list[dict[str, Any]] = []
        n_seen = 0
        sample_counter = 0

        for batch in dl:
            if n_seen >= target_samples:
                break
            undersampled = batch["undersampled"].to(device)
            mask = batch["mask"].to(device)
            fs_k = batch["fs_k_space"].to(device)

            if not undersampled.is_complex():
                undersampled = undersampled.to(torch.complex64)
            if not fs_k.is_complex():
                fs_k = fs_k.to(torch.complex64)

            bsz = int(undersampled.shape[0])
            for bi in range(bsz):
                if n_seen >= target_samples:
                    break

                u = undersampled[bi : bi + 1]
                m = mask[bi : bi + 1]
                f = fs_k[bi : bi + 1]

                _ = model.forward(u, m, f)
                cab_mean, cab_gate_np, mix_np, mix_gate_np, cab_full = _extract_stem_attn_from_backbone(backbone)

                sample_id = int(sample_counter + bi)
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "cab_gate": float(np.asarray(cab_gate_np).ravel()[0]) if cab_gate_np is not None else float("nan"),
                    "mix_gate": float(np.asarray(mix_gate_np).ravel()[0]) if mix_gate_np is not None else float("nan"),
                }

                cab_score = float("nan")
                mix_score = float("nan")
                cab_head_best_score = float("nan")
                cab_head_best_idx: int | None = None
                cab_head_best_mat: np.ndarray | None = None

                if cab_mean is not None and cab_mean.shape[0] >= 1:
                    cab_k = np.asarray(cab_mean[0], dtype=np.float64)
                    cab_score = _score_for_ranking(cab_k)
                    row.update({f"cab_mean_{k}": v for k, v in _attn_nonuniform_metrics(cab_k).items()})
                else:
                    row.update({
                        "cab_mean_l1_from_uniform": float("nan"),
                        "cab_mean_max_abs_from_uniform": float("nan"),
                        "cab_mean_kl_from_uniform": float("nan"),
                        "cab_mean_one_minus_entropy": float("nan"),
                    })

                if mix_np is not None and mix_np.shape[0] >= 1:
                    mix_k = np.asarray(mix_np[0], dtype=np.float64)
                    mix_score = _score_for_ranking(mix_k)
                    row.update({f"mix_{k}": v for k, v in _attn_nonuniform_metrics(mix_k).items()})
                else:
                    row.update({
                        "mix_l1_from_uniform": float("nan"),
                        "mix_max_abs_from_uniform": float("nan"),
                        "mix_kl_from_uniform": float("nan"),
                        "mix_one_minus_entropy": float("nan"),
                    })

                if cab_full is not None and cab_full.shape[0] >= 1:
                    # Find most non-uniform head for this sample.
                    H = int(cab_full.shape[1])
                    best_s = -1.0
                    best_h = None
                    best_mat = None
                    for h in range(H):
                        mat = np.asarray(cab_full[0, h], dtype=np.float64)
                        s = _score_for_ranking(mat)
                        if s > best_s:
                            best_s = s
                            best_h = h
                            best_mat = mat
                    cab_head_best_score = float(best_s)
                    cab_head_best_idx = int(best_h) if best_h is not None else None
                    cab_head_best_mat = best_mat.astype(np.float32) if best_mat is not None else None
                    row["cab_head_best_idx"] = -1 if cab_head_best_idx is None else int(cab_head_best_idx)
                    row["cab_head_best_score"] = float(cab_head_best_score)
                else:
                    row["cab_head_best_idx"] = -1
                    row["cab_head_best_score"] = float("nan")

                rows.append(row)

                # Ranking uses the strongest available evidence: best head if present, else mean, else mix.
                best_overall = float("nan")
                if not np.isnan(cab_head_best_score):
                    best_overall = float(cab_head_best_score)
                elif not np.isnan(cab_score):
                    best_overall = float(cab_score)
                elif not np.isnan(mix_score):
                    best_overall = float(mix_score)

                # Maintain a small top list.
                if not np.isnan(best_overall):
                    entry = (
                        float(best_overall),
                        sample_id,
                        None if cab_mean is None else cab_mean[0].astype(np.float32),
                        None if mix_np is None else mix_np[0].astype(np.float32),
                        cab_head_best_mat,
                        cab_head_best_idx,
                    )
                    top.append(entry)
                    top = sorted(top, key=lambda x: x[0], reverse=True)[: max(save_top, 1)]

                n_seen += 1

            sample_counter += bsz

        # Write CSV
        csv_path = out_dir / "stem_attn_nonuniform_scan.csv"
        # Stable field order
        fieldnames: list[str] = [
            "sample_id",
            "cab_gate",
            "mix_gate",
            "cab_mean_l1_from_uniform",
            "cab_mean_max_abs_from_uniform",
            "cab_mean_kl_from_uniform",
            "cab_mean_one_minus_entropy",
            "mix_l1_from_uniform",
            "mix_max_abs_from_uniform",
            "mix_kl_from_uniform",
            "mix_one_minus_entropy",
            "cab_head_best_idx",
            "cab_head_best_score",
        ]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

        # Optional histogram plots for paper-friendly summaries.
        plt = _try_import_matplotlib()
        if plt is not None and rows:
            def _col(name: str) -> np.ndarray:
                vals = []
                for r in rows:
                    try:
                        v = float(r.get(name, float("nan")))
                    except Exception:
                        v = float("nan")
                    if not np.isnan(v):
                        vals.append(v)
                return np.asarray(vals, dtype=np.float64)

            for col, fname, title in [
                ("cab_mean_max_abs_from_uniform", "hist_cab_mean_max_abs.png", "CAB mean: max |attn-1/K|"),
                ("mix_max_abs_from_uniform", "hist_mix_max_abs.png", "MIX: max |attn-1/K|"),
                ("cab_head_best_score", "hist_cab_head_best_score.png", "CAB best-head score"),
            ]:
                arr = _col(col)
                if arr.size == 0:
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
                ax.hist(arr, bins=40, color="C0", alpha=0.85)
                ax.set_title(title)
                ax.set_xlabel(col)
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig((out_dir / fname).as_posix(), dpi=200)
                plt.close(fig)

        # Save heatmaps for top examples (if matplotlib exists)
        if save_top > 0 and top:
            for rank, (score, sample_id, cab_mean_kk, mix_kk, cab_head_kk, head_idx) in enumerate(top[:save_top]):
                tag = f"top{rank:02d}_sample{sample_id}_score{score:.4f}"
                if cab_mean_kk is not None:
                    _save_attn_heatmap(
                        out_path=out_dir / f"{tag}_cab_mean.png",
                        attn=cab_mean_kk,
                        labels=contrast_order,
                        title=f"CAB mean nonuniform score={score:.4f}",
                    )
                if cab_head_kk is not None and head_idx is not None and head_idx >= 0:
                    _save_attn_heatmap(
                        out_path=out_dir / f"{tag}_cab_head{head_idx}.png",
                        attn=cab_head_kk,
                        labels=contrast_order,
                        title=f"CAB head {head_idx} nonuniform score={score:.4f}",
                    )
                if mix_kk is not None:
                    _save_attn_heatmap(
                        out_path=out_dir / f"{tag}_mix.png",
                        attn=mix_kk,
                        labels=contrast_order,
                        title=f"MIX nonuniform score={score:.4f}",
                    )

        print("\n" + "=" * 80)
        print(f"Scanned {len(rows)} samples. Wrote: {csv_path}")
        if top:
            best = top[0]
            print(f"Best non-uniform example: sample_id={best[1]} score={best[0]:.4f}")
        print(f"Output dir: {out_dir}")
        return

    if bool(args.permute_eval):
        # Evaluate first N samples; for each, run M permutations (including identity).
        target_samples = int(args.num_samples)
        target_perms = int(args.num_perms)
        if target_samples < 1:
            raise ValueError("--num_samples must be >= 1")
        if target_perms < 1:
            raise ValueError("--num_perms must be >= 1")

        k = len(contrast_order)
        perms_for_all = _unique_permutations(k=k, n=target_perms, seed=int(args.seed))

        # Accumulate results to one NPZ for later plotting.
        cab_all: list[np.ndarray] = []  # each [P,K,K]
        mix_all: list[np.ndarray] = []  # each [P,K,K]
        cab_gates_all: list[float] = []
        mix_gates_all: list[float] = []
        sample_ids: list[int] = []

        n_seen = 0
        sample_counter = 0
        for batch in dl:
            if n_seen >= target_samples:
                break
            undersampled = batch["undersampled"].to(device)
            mask = batch["mask"].to(device)
            fs_k = batch["fs_k_space"].to(device)

            if not undersampled.is_complex():
                undersampled = undersampled.to(torch.complex64)
            if not fs_k.is_complex():
                fs_k = fs_k.to(torch.complex64)

            bsz = int(undersampled.shape[0])
            for bi in range(bsz):
                if n_seen >= target_samples:
                    break

                u0 = undersampled[bi : bi + 1]
                m0 = mask[bi : bi + 1]
                f0 = fs_k[bi : bi + 1]

                print("\n" + "=" * 80)
                print(f"Sample {n_seen} (global_index={sample_counter + bi})")

                cab_this: list[np.ndarray] = []
                mix_this: list[np.ndarray] = []
                cab_gate_val: float | None = None
                mix_gate_val: float | None = None

                for pi, perm in enumerate(perms_for_all):
                    perm = np.asarray(perm, dtype=np.int64)
                    # permute along contrast dimension (dim=1)
                    u = u0[:, perm]
                    m = m0[:, perm]
                    f = f0[:, perm]

                    perm_labels = [contrast_order[int(i)] for i in perm.tolist()]
                    print("\n" + "-" * 80)
                    print(f"perm {pi}: {perm.tolist()} -> labels={perm_labels}")

                    _ = model.forward(u, m, f)
                    cab_mean, cab_gate_np, mix_np, mix_gate_np, cab_full = _extract_stem_attn_from_backbone(backbone)

                    if cab_mean is None:
                        print("CAB attention: <not available>")
                    else:
                        if int(args.cab_head) >= 0 and cab_full is not None:
                            h = int(args.cab_head)
                            if h < 0 or h >= int(cab_full.shape[1]):
                                raise IndexError(f"--cab_head {h} out of range for H={int(cab_full.shape[1])}")
                            cab_k = cab_full[0, h]
                            print(f"CAB attention (head {h}):")
                        else:
                            cab_k = cab_mean[0]
                            print("CAB attention (mean over heads):")
                        cab_this.append(cab_k.astype(np.float32))
                        cab_gate_val = float(np.asarray(cab_gate_np).ravel()[0]) if cab_gate_np is not None else None
                        _print_attn_matrix(tag="cab", labels=perm_labels, attn=cab_k, gate=cab_gate_np)

                    if mix_np is None:
                        print("MIX attention: <not available>")
                    else:
                        mix_k = mix_np[0]
                        mix_this.append(mix_k.astype(np.float32))
                        mix_gate_val = float(np.asarray(mix_gate_np).ravel()[0]) if mix_gate_np is not None else None
                        print("MIX attention:")
                        _print_attn_matrix(tag="mix", labels=perm_labels, attn=mix_k, gate=mix_gate_np)

                # Stack per sample (P,K,K). If missing, store empty.
                cab_all.append(np.stack(cab_this, axis=0) if cab_this else np.zeros((0, k, k), dtype=np.float32))
                mix_all.append(np.stack(mix_this, axis=0) if mix_this else np.zeros((0, k, k), dtype=np.float32))
                cab_gates_all.append(float("nan") if cab_gate_val is None else float(cab_gate_val))
                mix_gates_all.append(float("nan") if mix_gate_val is None else float(mix_gate_val))
                sample_ids.append(int(sample_counter + bi))

                n_seen += 1

            sample_counter += bsz

        out_npz = out_dir / "stem_attn_permutation_eval.npz"
        np.savez(
            out_npz.as_posix(),
            contrast_order=np.array(contrast_order, dtype=object),
            perms=np.stack(perms_for_all, axis=0).astype(np.int64),  # [P,K]
            sample_ids=np.array(sample_ids, dtype=np.int64),
            cab_attn=np.stack(cab_all, axis=0),  # [S,P,K,K] or [S,0,K,K]
            mix_attn=np.stack(mix_all, axis=0),
            cab_gate=np.array(cab_gates_all, dtype=np.float32),
            mix_gate=np.array(mix_gates_all, dtype=np.float32),
            cascade_idx=int(args.cascade_idx),
            num_samples=int(target_samples),
            num_perms=int(target_perms),
            seed=int(args.seed),
        )
        print("\n" + "=" * 80)
        print(f"Wrote: {out_npz}")
        print(f"Output dir: {out_dir}")
        return

    # Legacy single-forward mode (no permutations): run one batch and render summaries.
    batch_B = _run_one_batch(model=model, dataloader=dl, batch_index=0, device=device)
    cab_mean, cab_gate_np, mix_np, mix_gate_np, _ = _extract_stem_attn_from_backbone(backbone)

    payload: dict[str, Any] = {
        "contrast_order": np.array(contrast_order, dtype=object),
        "batch_size_seen": int(batch_B),
        "cascade_idx": int(args.cascade_idx),
        "batch_index": 0,
    }
    if cab_mean is not None:
        payload["cab_attn_mean"] = cab_mean.astype(np.float32)
    if cab_gate_np is not None:
        payload["cab_gate"] = cab_gate_np
    if mix_np is not None:
        payload["mix_attn"] = mix_np.astype(np.float32)
    if mix_gate_np is not None:
        payload["mix_gate"] = mix_gate_np

    npz_path = out_dir / "stem_attn.npz"
    np.savez(npz_path.as_posix(), **payload)

    if cab_mean is not None:
        for b in range(min(batch_B, cab_mean.shape[0])):
            _save_attn_heatmap(
                out_path=out_dir / f"stem_crosscab_attn_sample{b}.png",
                attn=cab_mean[b],
                labels=contrast_order,
                title=f"Stem cross-contrast attn (CAB pooled) sample {b}",
            )
        _save_attn_heatmap(
            out_path=out_dir / "stem_crosscab_attn_batchmean.png",
            attn=cab_mean.mean(axis=0),
            labels=contrast_order,
            title="Stem cross-contrast attn (CAB pooled) batch mean",
        )
        _summarize_attn(
            attn=cab_mean,
            labels=contrast_order,
            gates=cab_gate_np,
            tag="cab",
            out_dir=out_dir,
            topk=int(args.topk),
        )

    if mix_np is not None:
        for b in range(min(batch_B, mix_np.shape[0])):
            _save_attn_heatmap(
                out_path=out_dir / f"stem_freqmix_attn_sample{b}.png",
                attn=mix_np[b],
                labels=contrast_order,
                title=f"Stem spatial+freq mix attn sample {b}",
            )
        _save_attn_heatmap(
            out_path=out_dir / "stem_freqmix_attn_batchmean.png",
            attn=mix_np.mean(axis=0),
            labels=contrast_order,
            title="Stem spatial+freq mix attn batch mean",
        )
        _summarize_attn(
            attn=mix_np,
            labels=contrast_order,
            gates=mix_gate_np,
            tag="mix",
            out_dir=out_dir,
            topk=int(args.topk),
        )

    print(f"Wrote: {npz_path}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
