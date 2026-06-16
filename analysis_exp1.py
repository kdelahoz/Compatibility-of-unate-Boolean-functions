# -*- coding: utf-8 -*-
"""
Post-analysis for Experiment 1.

Reads:
- Results/raw_exp1.csv

Produces:
- Results/summary_exp1.csv:
  Summary by (n, m). Stores run counts, seeds, sizes of F0/F1, number of
  discrepancies, time statistics, restart statistics, coverage-vector weight
  statistics, and solution rates.

- Results/table_exp1_by_m.csv:
  Compact table by m. Stores the number of discrepancies, the range of mean
  coverage-vector weights over n, the maximum time coefficient of variation,
  and the minimum solution rate.
"""


from pathlib import Path

import pandas as pd


# ---------------------------
# Paths
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"
RAW_INPUT = RESULTS_DIR / "raw_exp1.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "summary_exp1.csv"
TABLE_OUTPUT = RESULTS_DIR / "table_exp1_by_m.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Load raw data
# ---------------------------

raw = pd.read_csv(RAW_INPUT)

required_columns = {
    "n", "m", "tb", "seed",
    "F0_size", "F1_size",
    "d", "time_sec", "restarts",
    "solution_found", "sigma_weight",
}
missing = required_columns - set(raw.columns)
if missing:
    raise ValueError(f"Missing required columns in raw CSV: {sorted(missing)}")


# ---------------------------
# summary_exp1.csv
# ---------------------------

summary = (
    raw.groupby(["n", "m"], as_index=False)
    .agg(
        tb_count=("tb", "count"),
        seed_min=("seed", "min"),
        seed_max=("seed", "max"),

        F0_size=("F0_size", "first"),
        F1_size=("F1_size", "first"),
        d=("d", "first"),

        time_mean=("time_sec", "mean"),
        time_std=("time_sec", "std"),
        time_median=("time_sec", "median"),
        time_min=("time_sec", "min"),
        time_max=("time_sec", "max"),

        restarts_mean=("restarts", "mean"),
        restarts_std=("restarts", "std"),
        restarts_median=("restarts", "median"),
        restarts_min=("restarts", "min"),
        restarts_max=("restarts", "max"),

        sigma_weight_mean=("sigma_weight", "mean"),
        sigma_weight_std=("sigma_weight", "std"),
        sigma_weight_median=("sigma_weight", "median"),
        sigma_weight_min=("sigma_weight", "min"),
        sigma_weight_max=("sigma_weight", "max"),

        num_found=("solution_found", "sum"),
    )
)

summary["num_found"] = summary["num_found"].astype(int)
summary["num_not_found"] = summary["tb_count"] - summary["num_found"]

summary["time_cv"] = summary["time_std"] / summary["time_mean"]
summary["found_rate"] = summary["num_found"] / summary["tb_count"]

summary = summary[
    [
        "n", "m", "tb_count", "seed_min", "seed_max",
        "F0_size", "F1_size", "d",
        "time_mean", "time_std", "time_median", "time_min", "time_max", "time_cv",
        "restarts_mean", "restarts_std", "restarts_median", "restarts_min", "restarts_max",
        "sigma_weight_mean", "sigma_weight_std", "sigma_weight_median", "sigma_weight_min", "sigma_weight_max",
        "num_found", "num_not_found", "found_rate",
    ]
].sort_values(["n", "m"], ignore_index=True)

summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.4f")

print(f"Saved: {SUMMARY_OUTPUT}")

# ---------------------------
# table_exp1_by_m.csv
# ---------------------------

summary_by_m = (
    summary.groupby("m", as_index=False)
    .agg(
        d=("d", "first"),
        sigma_weight_mean_min=("sigma_weight_mean", "min"),
        sigma_weight_mean_max=("sigma_weight_mean", "max"),
        time_cv_max=("time_cv", "max"),
        found_rate_min=("found_rate", "min"),
        tb_count_min=("tb_count", "min"),
        tb_count_max=("tb_count", "max"),
    )
)
summary_by_m.to_csv(TABLE_OUTPUT, index=False, float_format="%.4f")

print(f"Saved: {TABLE_OUTPUT}")

