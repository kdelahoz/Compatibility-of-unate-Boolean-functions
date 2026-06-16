# -*- coding: utf-8 -*-
"""
Post-analysis for Experiment 3.

Reads:
- Results/raw_exp3_drosophila.csv

Produces:
- Results/summary_exp3_drosophila.csv:
  Summary by (k, m). Stores run counts, node counts, sizes of F0/F1, number
  of discrepancies, support recovery statistics, signed recovery statistics,
  false-positive and false-negative statistics, coverage-vector weight
  statistics, weight-ratio statistics, restart statistics, sign-error
  diagnostics, and solution rates.
"""

from pathlib import Path
import ast

import numpy as np
import pandas as pd


# ---------------------------
# Paths
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"

RAW_INPUT = RESULTS_DIR / "raw_exp3_drosophila.csv"

SUMMARY_OUTPUT = RESULTS_DIR / "summary_exp3_drosophila.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Helpers
# ---------------------------

def parse_list(value):
    """Parse a list stored as a string in the raw CSV."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    return ast.literal_eval(str(value))


def safe_div(num, den):
    """Return num / den, or NaN if den is zero or missing."""
    if pd.isna(den) or den == 0:
        return np.nan
    return num / den


def q(prob):
    """Quantile aggregation helper for pandas groupby."""
    return lambda x: x.quantile(prob)


def as_bool(series):
    """Convert a boolean-like pandas Series to bool."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def add_recovery_metrics(raw):
    """
    Add support and signed recovery metrics.

    For each run:
    - support metrics evaluate whether true regulator coordinates were recovered;
    - signed metrics additionally require the correct sign on recovered true coordinates.
    """
    metrics_rows = []

    for row in raw.itertuples(index=False):
        support_true = parse_list(row.support_true)
        signs_true = parse_list(row.signs_true)
        support_pred = parse_list(row.support_predicted)
        signs_pred = parse_list(row.signs_predicted)

        true_sign = {int(i): int(s) for i, s in zip(support_true, signs_true)}
        pred_sign = {int(i): int(s) for i, s in zip(support_pred, signs_pred)}

        true_support = set(true_sign)
        pred_support = set(pred_sign)

        k = int(row.k)
        found = bool(row.solution_found)

        if found:
            if pd.isna(row.sigma_weight):
                sigma_weight = len(pred_support)
            else:
                sigma_weight = int(row.sigma_weight)

            support_tp = len(true_support & pred_support)

            signed_tp = sum(
                1
                for i in true_support & pred_support
                if pred_sign[i] == true_sign[i]
            )

            false_positives = sigma_weight - support_tp
            false_negatives = k - support_tp
            sign_errors = support_tp - signed_tp

            support_recall = safe_div(support_tp, k)
            support_precision = safe_div(support_tp, sigma_weight)

            signed_recall = safe_div(signed_tp, k)
            signed_precision = safe_div(signed_tp, sigma_weight)

            sign_accuracy_conditional = safe_div(signed_tp, support_tp)

            exact_support = (support_tp == k and sigma_weight == k)
            exact_signed = (signed_tp == k and sigma_weight == k)

            weight_ratio = safe_div(sigma_weight, k)

        else:
            sigma_weight = np.nan

            support_tp = np.nan
            signed_tp = np.nan

            false_positives = np.nan
            false_negatives = np.nan
            sign_errors = np.nan

            support_recall = np.nan
            support_precision = np.nan

            signed_recall = np.nan
            signed_precision = np.nan

            sign_accuracy_conditional = np.nan

            exact_support = False
            exact_signed = False

            weight_ratio = np.nan

        metrics_rows.append(
            {
                "support_tp": support_tp,
                "signed_tp": signed_tp,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "sign_errors": sign_errors,

                "support_recall": support_recall,
                "support_precision": support_precision,
                "signed_recall": signed_recall,
                "signed_precision": signed_precision,
                "sign_accuracy_conditional": sign_accuracy_conditional,

                "exact_support": exact_support,
                "exact_signed": exact_signed,

                "weight_ratio": weight_ratio,
            }
        )

    return pd.concat(
        [raw.reset_index(drop=True), pd.DataFrame(metrics_rows)],
        axis=1,
    )


# ---------------------------
# Load raw data
# ---------------------------

raw = pd.read_csv(RAW_INPUT)

required_columns = {
    "node",
    "n",
    "k",
    "m",
    "tb",
    "dataset_id",
    "dataset_seed",
    "F0_size",
    "F1_size",
    "d",
    "support_true",
    "signs_true",
    "sigma_weight",
    "support_predicted",
    "signs_predicted",
    "restarts",
    "solution_found",
}

missing = required_columns - set(raw.columns)
if missing:
    raise ValueError(
        f"Missing required columns in {RAW_INPUT.name}: {sorted(missing)}"
    )

raw = raw.copy()
raw["solution_found"] = as_bool(raw["solution_found"])

numeric_cols = [
    "n",
    "k",
    "m",
    "tb",
    "dataset_id",
    "dataset_seed",
    "F0_size",
    "F1_size",
    "d",
    "sigma_weight",
    "restarts",
]

for col in numeric_cols:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")


# ---------------------------
# Run-level metrics
# ---------------------------

metrics = add_recovery_metrics(raw)


# ---------------------------
# Node-level averages
# ---------------------------
# First averaging step reported in the article:
# for each fixed node and m, average over the 20 observation sets.

valid = metrics[metrics["solution_found"]].copy()

summary_by_node = (
    valid.groupby(["node", "k", "m"], as_index=False)
    .agg(
        tb_count=("tb", "count"),
        n=("n", "first"),

        F0_size_mean=("F0_size", "mean"),
        F1_size_mean=("F1_size", "mean"),
        d_mean=("d", "mean"),

        support_recall_mean=("support_recall", "mean"),
        support_precision_mean=("support_precision", "mean"),
        signed_recall_mean=("signed_recall", "mean"),
        signed_precision_mean=("signed_precision", "mean"),

        false_positives_mean=("false_positives", "mean"),
        false_negatives_mean=("false_negatives", "mean"),

        sigma_weight_mean=("sigma_weight", "mean"),
        sigma_weight_median=("sigma_weight", "median"),
        sigma_weight_max=("sigma_weight", "max"),

        weight_ratio_mean=("weight_ratio", "mean"),
        weight_ratio_median=("weight_ratio", "median"),
        weight_ratio_max=("weight_ratio", "max"),

        exact_support_rate=("exact_support", "mean"),
        exact_signed_rate=("exact_signed", "mean"),

        restarts_mean=("restarts", "mean"),
        restarts_median=("restarts", "median"),
        restarts_q90=("restarts", q(0.90)),
        restarts_max=("restarts", "max"),
    )
)


# ---------------------------
# Degree-level summary
# ---------------------------
# Second averaging step reported in the article:
# average the node-level values across nodes with the same degree k.

summary = (
    summary_by_node.groupby(["k", "m"], as_index=False)
    .agg(
        n_nodes=("node", "nunique"),
        tb_total=("tb_count", "sum"),
        n=("n", "first"),

        F0_size_mean=("F0_size_mean", "mean"),
        F1_size_mean=("F1_size_mean", "mean"),
        d_mean=("d_mean", "mean"),

        support_recall_mean=("support_recall_mean", "mean"),
        support_recall_sd_nodes=("support_recall_mean", "std"),

        support_precision_mean=("support_precision_mean", "mean"),
        support_precision_sd_nodes=("support_precision_mean", "std"),

        signed_recall_mean=("signed_recall_mean", "mean"),
        signed_recall_sd_nodes=("signed_recall_mean", "std"),

        signed_precision_mean=("signed_precision_mean", "mean"),
        signed_precision_sd_nodes=("signed_precision_mean", "std"),

        false_positives_mean=("false_positives_mean", "mean"),
        false_negatives_mean=("false_negatives_mean", "mean"),

        sigma_weight_mean=("sigma_weight_mean", "mean"),
        sigma_weight_sd_nodes=("sigma_weight_mean", "std"),
        sigma_weight_median=("sigma_weight_median", "median"),
        sigma_weight_max=("sigma_weight_max", "max"),

        weight_ratio_mean=("weight_ratio_mean", "mean"),
        weight_ratio_sd_nodes=("weight_ratio_mean", "std"),
        weight_ratio_median=("weight_ratio_median", "median"),
        weight_ratio_max=("weight_ratio_max", "max"),

        exact_support_rate=("exact_support_rate", "mean"),
        exact_signed_rate=("exact_signed_rate", "mean"),

        restarts_mean=("restarts_mean", "mean"),
        restarts_median=("restarts_median", "median"),
        restarts_q90=("restarts_q90", "median"),
        restarts_max=("restarts_max", "max"),
    )
)

for metric in [
    "support_recall",
    "support_precision",
    "signed_recall",
    "signed_precision",
    "sigma_weight",
    "weight_ratio",
]:
    sd_col = f"{metric}_sd_nodes"
    se_col = f"{metric}_se_nodes"
    summary[se_col] = summary[sd_col] / np.sqrt(summary["n_nodes"])


# ---------------------------
# Run-level support information
# ---------------------------
# These columns support the claims that every run returned a coverage vector
# and that no recovered true coordinate received an incorrect sign.

run_summary = (
    metrics.groupby(["k", "m"], as_index=False)
    .agg(
        run_count=("tb", "count"),
        found_count=("solution_found", "sum"),

        support_tp_total=("support_tp", "sum"),
        signed_tp_total=("signed_tp", "sum"),

        sign_errors_total=("sign_errors", "sum"),
        sign_error_runs=("sign_errors", lambda x: (x > 0).sum()),

        exact_support_count=("exact_support", "sum"),
        exact_signed_count=("exact_signed", "sum"),
    )
)

run_summary["found_count"] = run_summary["found_count"].astype(int)
run_summary["not_found_count"] = (
    run_summary["run_count"] - run_summary["found_count"]
)
run_summary["found_rate"] = (
    run_summary["found_count"] / run_summary["run_count"]
)

run_summary["sign_accuracy_global"] = (
    run_summary["signed_tp_total"]
    / run_summary["support_tp_total"].replace(0, np.nan)
)

run_summary["sign_error_run_rate"] = (
    run_summary["sign_error_runs"] / run_summary["run_count"]
)

run_summary["exact_support_but_not_signed"] = (
    run_summary["exact_support_count"] - run_summary["exact_signed_count"]
)

summary = summary.merge(
    run_summary,
    on=["k", "m"],
    how="left",
)

summary = summary[
    [
        "k",
        "m",
        "n",
        "n_nodes",
        "tb_total",
        "run_count",
        "found_count",
        "not_found_count",
        "found_rate",

        "F0_size_mean",
        "F1_size_mean",
        "d_mean",

        "support_recall_mean",
        "support_recall_sd_nodes",
        "support_recall_se_nodes",

        "support_precision_mean",
        "support_precision_sd_nodes",
        "support_precision_se_nodes",

        "signed_recall_mean",
        "signed_recall_sd_nodes",
        "signed_recall_se_nodes",

        "signed_precision_mean",
        "signed_precision_sd_nodes",
        "signed_precision_se_nodes",

        "support_tp_total",
        "signed_tp_total",
        "sign_accuracy_global",
        "sign_errors_total",
        "sign_error_runs",
        "sign_error_run_rate",

        "exact_support_count",
        "exact_signed_count",
        "exact_support_but_not_signed",
        "exact_support_rate",
        "exact_signed_rate",

        "false_positives_mean",
        "false_negatives_mean",

        "sigma_weight_mean",
        "sigma_weight_sd_nodes",
        "sigma_weight_se_nodes",
        "sigma_weight_median",
        "sigma_weight_max",

        "weight_ratio_mean",
        "weight_ratio_sd_nodes",
        "weight_ratio_se_nodes",
        "weight_ratio_median",
        "weight_ratio_max",

        "restarts_mean",
        "restarts_median",
        "restarts_q90",
        "restarts_max",
    ]
].sort_values(["k", "m"], ignore_index=True)

summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.6f")
print(f"Saved: {SUMMARY_OUTPUT}")
