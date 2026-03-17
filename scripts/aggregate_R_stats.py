#!/usr/bin/env python3
"""Aggregate mean and std of PSNR/SSIM per model and R-value.

Reads: M4Raw/M4RAW_FINAL/_revised_metrics/per_slice_revised_metrics.csv
Writes: M4Raw/M4RAW_FINAL/_revised_metrics/aggregate_model_R_stats.csv

Output columns: model,R,contrast,metric,mean,std,n
"""
from pathlib import Path
import pandas as pd


def aggregate(in_csv: Path, out_csv: Path) -> None:
    df = pd.read_csv(in_csv)

    # Ensure expected columns exist
    for c in ["model", "R"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c} in {in_csv}")

    # Metrics to aggregate
    metrics = [("psnr_actual_masked", "psnr"), ("ssim_actual_masked", "ssim")]

    rows = []

    # Emit rows grouped by model -> R -> contrasts, and then add an ALL row
    # immediately after the contrast-specific rows for that model/R.
    models = sorted(df["model"].unique())
    for m in models:
        df_m = df[df["model"] == m]
        # sort R values naturally (R4, R6, R8 etc.)
        Rs = sorted(df_m["R"].unique(), key=lambda x: str(x))
        for r in Rs:
            df_mr = df_m[df_m["R"] == r]

            # get contrast order (keep natural order if possible)
            contrasts = list(df_mr["contrast"].dropna().unique())

            # per-contrast rows
            for c in contrasts:
                g = df_mr[df_mr["contrast"] == c]
                for col, short in metrics:
                    if col in g.columns:
                        vals = pd.to_numeric(g[col], errors="coerce").dropna()
                        if len(vals) == 0:
                            continue
                        rows.append({
                            "model": m,
                            "R": r,
                            "contrast": c,
                            "metric": short,
                            "mean": float(vals.mean()),
                            "std": float(vals.std(ddof=0)),
                            "n": int(len(vals)),
                        })

            # overall across contrasts for this model/R
            for col, short in metrics:
                if col in df_mr.columns:
                    vals = pd.to_numeric(df_mr[col], errors="coerce").dropna()
                    if len(vals) == 0:
                        continue
                    rows.append({
                        "model": m,
                        "R": r,
                        "contrast": "ALL",
                        "metric": short,
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)),
                        "n": int(len(vals)),
                    })

    out = pd.DataFrame(rows)
    out = out[["model", "R", "contrast", "metric", "mean", "std", "n"]]
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out)} rows)")


def main():
    repo = Path(__file__).resolve().parents[1]
    in_csv = repo / "M4Raw" / "M4RAW_FINAL" / "_revised_metrics" / "per_slice_revised_metrics.csv"
    out_csv = repo / "M4Raw" / "M4RAW_FINAL" / "_revised_metrics" / "aggregate_model_R_stats.csv"
    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")
    aggregate(in_csv, out_csv)


if __name__ == "__main__":
    main()
