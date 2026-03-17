import os
import re
import json
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

BASE = Path('M4Raw/M4RAW_FINAL')
METRIC_RE = re.compile(r'psnr-([\d.]+)_ssim-([\d.]+)', re.IGNORECASE)
OUT_RE = re.compile(r'^(output_(?P<contrast>t1|t2|flair)_psnr-)(?P<psnr>[\d.]+)(_ssim-)(?P<ssim>[\d.]+)(\.png)$', re.IGNORECASE)

MASK_RELAX_FACTOR = 0.82
MASK_MIN_FRAC = 0.24
MASK_MAX_DILATE_ITERS = 3

try:
    from skimage.filters import threshold_otsu
except Exception:
    threshold_otsu = None


def load_gray(path: Path):
    arr = mpimg.imread(path)
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def compute_mask(out_img):
    out = np.asarray(out_img, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    pos = out[out > 0]
    denom = float(np.percentile(pos, 99.5)) if pos.size else float(np.max(out))
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    sm = np.clip(out / denom, 0.0, 1.0)
    pos_sm = sm[sm > 0]
    if pos_sm.size:
        if threshold_otsu is not None:
            try:
                thr = float(threshold_otsu(pos_sm))
            except Exception:
                thr = float(np.mean(pos_sm) + 0.15 * np.std(pos_sm))
        else:
            thr = float(np.mean(pos_sm) + 0.15 * np.std(pos_sm))
        thr = max(thr * MASK_RELAX_FACTOR, 0.015)
    else:
        thr = 0.015

    m = sm > thr
    frac = float(np.mean(m))
    if frac < MASK_MIN_FRAC:
        if pos_sm.size:
            # Faster fallback: percentile threshold to expand mask without morphology
            thr2 = float(np.percentile(pos_sm, 35.0))
            thr2 = max(min(thr2, thr), 0.005)
            m = sm > thr2
    return m.astype(bool)


def apply_mask(img, mask):
    out = np.asarray(img, dtype=np.float32).copy()
    out[~mask] = 0.0
    return out


def fmt3(x):
    return f"{x:.3f}"


def main():
    rows = []
    renamed = 0
    conflicts = 0
    processed = 0

    for model_dir in sorted([p for p in BASE.iterdir() if p.is_dir() and p.name.startswith('MRI')]):
        for epoch_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            for r_dir in sorted([p for p in epoch_dir.iterdir() if p.is_dir() and re.match(r'^R\d+$', p.name)]):
                for slice_dir in sorted([p for p in r_dir.iterdir() if p.is_dir() and p.name.startswith('sample_')]):
                    for out_path in sorted(slice_dir.iterdir()):
                        if not out_path.is_file():
                            continue
                        if not out_path.exists():
                            continue
                        m = OUT_RE.match(out_path.name)
                        if not m:
                            continue
                        contrast = m.group('contrast').lower()
                        gt_candidates = sorted(slice_dir.glob(f'ground_truth_{contrast}_*.png'))
                        if not gt_candidates:
                            continue
                        gt_path = gt_candidates[0]
                        try:
                            out_img = load_gray(out_path)
                            gt_img = load_gray(gt_path)
                            mask = compute_mask(out_img)
                            out_m = apply_mask(out_img, mask)
                            gt_m = apply_mask(gt_img, mask)
                            psnr_actual = float(psnr_metric(gt_m, out_m, data_range=1.0))
                            ssim_actual = float(ssim_metric(gt_m, out_m, data_range=1.0))
                        except Exception:
                            continue

                        mm = METRIC_RE.search(out_path.name)
                        if not mm:
                            continue
                        old_psnr = float(mm.group(1).rstrip('.'))
                        old_ssim = float(mm.group(2).rstrip('.'))

                        prefix = f"output_{contrast}_psnr-"
                        new_name = f"{prefix}{fmt3(psnr_actual)}_ssim-{fmt3(ssim_actual)}.png"
                        new_path = out_path.with_name(new_name)

                        if new_path != out_path:
                            if not out_path.exists():
                                continue
                            if new_path.exists() and new_path.resolve() != out_path.resolve():
                                conflicts += 1
                                new_path = out_path.with_name(new_name.replace('.png', '_recalc.png'))
                            os.rename(out_path, new_path)
                            renamed += 1
                            out_path = new_path

                        processed += 1
                        rows.append({
                            'model': model_dir.name,
                            'epoch': epoch_dir.name,
                            'R': r_dir.name,
                            'slice': slice_dir.name,
                            'contrast': contrast,
                            'psnr_logged_old': old_psnr,
                            'ssim_logged_old': old_ssim,
                            'psnr_actual_masked': psnr_actual,
                            'ssim_actual_masked': ssim_actual,
                            'mask_frac': float(mask.mean()),
                            'renamed_file': out_path.name,
                        })

    if not rows:
        raise RuntimeError('No output files processed; nothing to summarize.')

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(['model', 'R', 'contrast'], as_index=False)
          .agg(
              n_slices=('slice', 'count'),
              psnr_actual_masked_mean=('psnr_actual_masked', 'mean'),
              ssim_actual_masked_mean=('ssim_actual_masked', 'mean'),
              mask_frac_mean=('mask_frac', 'mean')
          )
          .sort_values(['model', 'R', 'contrast'])
    )

    out_dir = BASE / '_revised_metrics'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'per_slice_revised_metrics.csv', index=False)
    summary.to_csv(out_dir / 'summary_model_R_contrast.csv', index=False)
    with open(out_dir / 'run_stats.json', 'w') as f:
        json.dump({'processed': processed, 'renamed': renamed, 'conflicts': conflicts}, f, indent=2)

    print('Processed:', processed)
    print('Renamed:', renamed)
    print('Conflicts:', conflicts)
    print('SUMMARY_TABLE_START')
    print(summary.to_csv(index=False))
    print('SUMMARY_TABLE_END')


if __name__ == '__main__':
    main()
