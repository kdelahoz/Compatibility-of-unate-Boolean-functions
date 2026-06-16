# -*- coding: utf-8 -*-
"""
Post-analysis for Experiment 2.

Reads:
- Results/raw_exp2.csv
- Results/raw_exp2_kmax10.csv

Produces:
- Results/sign_run_summary_exp2.csv:
  Summary by regime. Stores how many runs recovered at least one true
  coordinate and, among them, how many assigned all recovered true coordinates
  the correct sign.

- Results/summary_exp2.csv:
  Main summary by (regime, n, k, m). Stores support and signed recovery
  metrics, false positives/negatives, exact recovery counts, sign-error
  counts, coverage-vector weights, weight ratios, restart statistics, and
  solution rates.

- Results/comparison_exp2.csv:
  Paired comparison between unrestricted and kmax10 runs by (n, k, m).
  Stores average differences in recovery metrics, weights, false positives,
  false negatives, and restarts.

- Results/sign_errors_exp2.csv:
  Diagnostic table with only the individual runs where at least one recovered
  true coordinate received the wrong sign. Stores the true and predicted
  supports/signs and the corresponding sign-error counts.
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

RAW_UNRESTRICTED = RESULTS_DIR / "raw_exp2.csv"
RAW_KMAX10 = RESULTS_DIR / "raw_exp2_kmax10.csv"

SUMMARY_OUTPUT = RESULTS_DIR / "summary_exp2.csv"
COMPARISON_OUTPUT = RESULTS_DIR / "comparison_exp2.csv"
SIGN_ERRORS_OUTPUT = RESULTS_DIR / "sign_errors_exp2.csv"

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
    """Return num / den, or NaN if den is zero."""
    if den == 0:
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


def load_raw(path, regime_name, k_max_value):
    """Load one raw file, validate expected columns, and add regime labels."""
    raw = pd.read_csv(path)

    required_columns = {
        "n", "k", "m", "f_id", "tb", 
        "function_seed", "embedding_seed", "dataset_seed",
        "F0_size", "F1_size", "d",
        "support_true", "signs_true",
        "antichain_masks", "antichain_sets",
        "sigma_weight", "support_predicted", "signs_predicted",
        "restarts", "solution_found",
    }

    missing = required_columns - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {sorted(missing)}"
        )

    raw = raw.copy()
    raw["solution_found"] = as_bool(raw["solution_found"])
    raw["regime"] = regime_name
    raw["k_max"] = k_max_value

    return raw


def add_recovery_metrics(raw):
    """
    Add support and signed recovery metrics.

    For each run:
    - support metrics evaluate whether true influencer coordinates were recovered.
    - signed metrics additionally require the correct sign on recovered coordinates.
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

        exact_support_recovery = (support_tp == k and sigma_weight == k)
        exact_signed_recovery = (signed_tp == k and sigma_weight == k)

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

                "exact_support_recovery": exact_support_recovery,
                "exact_signed_recovery": exact_signed_recovery,

                "weight_ratio": safe_div(sigma_weight, k),
                "m_over_n": safe_div(row.m, row.n),
                "m_over_nk": safe_div(row.m, row.n * row.k),
            }
        )

    return pd.concat(
        [raw.reset_index(drop=True), pd.DataFrame(metrics_rows)],
        axis=1,
    )


# ---------------------------
# Load raw data 
# ---------------------------

raw_unrestricted = load_raw(
    RAW_UNRESTRICTED,
    regime_name="unrestricted",
    k_max_value=np.inf,
)

raw_kmax10 = load_raw(
    RAW_KMAX10,
    regime_name="kmax10",
    k_max_value=10,
)


metrics = pd.concat(
    [raw_unrestricted, raw_kmax10],
    ignore_index=True,
)

metrics = add_recovery_metrics(metrics)

# ---------------------------
# Conditional sign recovery rate
# ---------------------------

metrics["selected_true_coordinate"] = metrics["support_tp"] > 0
metrics["all_recovered_true_signs_correct"] = (
    metrics["selected_true_coordinate"] & (metrics["sign_errors"] == 0)
)

sign_run_summary = (
    metrics.groupby("regime", as_index=False)
    .agg(
        runs_with_true_coordinate=("selected_true_coordinate", "sum"),
        runs_all_recovered_true_signs_correct=("all_recovered_true_signs_correct", "sum"),
    )
)

sign_run_summary["sign_run_accuracy_conditional"] = (
    sign_run_summary["runs_all_recovered_true_signs_correct"]
    / sign_run_summary["runs_with_true_coordinate"]
)

sign_run_summary.to_csv(
    RESULTS_DIR / "sign_run_summary_exp2.csv",
    index=False,
    float_format="%.6f",
)

print("Saved:", RESULTS_DIR / "sign_run_summary_exp2.csv")

# ---------------------------
# summary_exp2.csv
# ---------------------------

function_group_cols = ["regime", "n", "k", "m", "f_id"]
config_group_cols = ["regime", "n", "k", "m"]

summary_by_function = (
    metrics.groupby(function_group_cols, as_index=False)
    .agg(
        tb_count=("tb", "count"),

        function_seed=("function_seed", "first"),
        embedding_seed=("embedding_seed", "first"),
        dataset_seed_min=("dataset_seed", "min"),
        dataset_seed_max=("dataset_seed", "max"),

        F0_size_mean=("F0_size", "mean"),
        F1_size_mean=("F1_size", "mean"),
        d_mean=("d", "mean"),

        support_recall_mean=("support_recall", "mean"),
        support_precision_mean=("support_precision", "mean"),
        signed_recall_mean=("signed_recall", "mean"),
        signed_precision_mean=("signed_precision", "mean"),

        support_tp_sum=("support_tp", "sum"),
        signed_tp_sum=("signed_tp", "sum"),
        sign_errors_sum=("sign_errors", "sum"),
        false_positives_mean=("false_positives", "mean"),
        false_negatives_mean=("false_negatives", "mean"),

        sigma_weight_mean=("sigma_weight", "mean"),
        sigma_weight_median=("sigma_weight", "median"),
        sigma_weight_max=("sigma_weight", "max"),

        weight_ratio_mean=("weight_ratio", "mean"),
        weight_ratio_median=("weight_ratio", "median"),
        weight_ratio_max=("weight_ratio", "max"),

        restarts_mean=("restarts", "mean"),
        restarts_median=("restarts", "median"),
        restarts_max=("restarts", "max"),

        found_count=("solution_found", "sum"),
        exact_support_count=("exact_support_recovery", "sum"),
        exact_signed_count=("exact_signed_recovery", "sum"),

        m_over_n=("m_over_n", "first"),
        m_over_nk=("m_over_nk", "first"),
    )
)

summary_by_function["found_count"] = summary_by_function["found_count"].astype(int)
summary_by_function["found_rate"] = (
    summary_by_function["found_count"] / summary_by_function["tb_count"]
)

# Function-level means summarized across functions.
summary_function_level = (
    summary_by_function.groupby(config_group_cols, as_index=False)
    .agg(
        f_count=("f_id", "count"),
        tb_total=("tb_count", "sum"),

        F0_size_mean=("F0_size_mean", "mean"),
        F1_size_mean=("F1_size_mean", "mean"),
        d_mean=("d_mean", "mean"),

        support_recall_mean=("support_recall_mean", "mean"),
        support_recall_sd_functions=("support_recall_mean", "std"),
        support_precision_mean=("support_precision_mean", "mean"),
        support_precision_sd_functions=("support_precision_mean", "std"),

        signed_recall_mean=("signed_recall_mean", "mean"),
        signed_recall_sd_functions=("signed_recall_mean", "std"),
        signed_precision_mean=("signed_precision_mean", "mean"),
        signed_precision_sd_functions=("signed_precision_mean", "std"),

        false_positives_mean=("false_positives_mean", "mean"),
        false_negatives_mean=("false_negatives_mean", "mean"),

        sigma_weight_mean=("sigma_weight_mean", "mean"),
        sigma_weight_sd_functions=("sigma_weight_mean", "std"),
        sigma_weight_median=("sigma_weight_median", "median"),
        sigma_weight_max=("sigma_weight_max", "max"),

        weight_ratio_mean=("weight_ratio_mean", "mean"),
        weight_ratio_median=("weight_ratio_median", "median"),
        weight_ratio_max=("weight_ratio_max", "max"),

        found_rate_mean=("found_rate", "mean"),
        m_over_n=("m_over_n", "first"),
        m_over_nk=("m_over_nk", "first"),
    )
)

for metric in [
    "support_recall", "support_precision",
    "signed_recall", "signed_precision",
    "sigma_weight",
]:
    sd_col = f"{metric}_sd_functions"
    se_col = f"{metric}_se_functions"
    summary_function_level[se_col] = (
        summary_function_level[sd_col]
        / np.sqrt(summary_function_level["f_count"])
    )

# Run-level totals and heavy-tail statistics, useful for diagnostics and cost.
summary_run_level = (
    metrics.groupby(config_group_cols, as_index=False)
    .agg(
        run_count=("tb", "count"),
        found_count=("solution_found", "sum"),

        support_tp_total=("support_tp", "sum"),
        signed_tp_total=("signed_tp", "sum"),
        sign_errors_total=("sign_errors", "sum"),
        sign_error_runs=("sign_errors", lambda x: (x > 0).sum()),

        exact_support_count=("exact_support_recovery", "sum"),
        exact_signed_count=("exact_signed_recovery", "sum"),

        restarts_mean=("restarts", "mean"),
        restarts_median=("restarts", "median"),
        restarts_q90=("restarts", q(0.90)),
        restarts_q95=("restarts", q(0.95)),
        restarts_q99=("restarts", q(0.99)),
        restarts_max=("restarts", "max"),

        weight_ratio_run_median=("weight_ratio", "median"),
        weight_ratio_run_max=("weight_ratio", "max"),
    )
)

summary_run_level["found_count"] = summary_run_level["found_count"].astype(int)
summary_run_level["not_found_count"] = (
    summary_run_level["run_count"] - summary_run_level["found_count"]
)
summary_run_level["found_rate"] = (
    summary_run_level["found_count"] / summary_run_level["run_count"]
)
summary_run_level["sign_accuracy_global"] = (
    summary_run_level["signed_tp_total"]
    / summary_run_level["support_tp_total"].replace(0, np.nan)
)
summary_run_level["sign_error_run_rate"] = (
    summary_run_level["sign_error_runs"] / summary_run_level["run_count"]
)
summary_run_level["exact_support_but_not_signed"] = (
    summary_run_level["exact_support_count"]
    - summary_run_level["exact_signed_count"]
)

summary = summary_function_level.merge(
    summary_run_level,
    on=config_group_cols,
    how="left",
    suffixes=("", "_run"),
)

summary = summary[
    [
        "regime", "n", "k", "m", "m_over_n", "m_over_nk",
        "f_count", "tb_total", "run_count",
        "F0_size_mean", "F1_size_mean", "d_mean",

        "support_recall_mean", "support_recall_sd_functions", "support_recall_se_functions",
        "support_precision_mean", "support_precision_sd_functions", "support_precision_se_functions",
        "signed_recall_mean", "signed_recall_sd_functions", "signed_recall_se_functions",
        "signed_precision_mean", "signed_precision_sd_functions", "signed_precision_se_functions",

        "support_tp_total", "signed_tp_total", "sign_accuracy_global",
        "sign_errors_total", "sign_error_runs", "sign_error_run_rate",
        "exact_support_count", "exact_signed_count", "exact_support_but_not_signed",

        "false_positives_mean", "false_negatives_mean",
        "sigma_weight_mean", "sigma_weight_sd_functions", "sigma_weight_se_functions",
        "sigma_weight_median", "sigma_weight_max",
        "weight_ratio_mean", "weight_ratio_median", "weight_ratio_max",
        "weight_ratio_run_median", "weight_ratio_run_max",

        "restarts_mean", "restarts_median", "restarts_q90", "restarts_q95", "restarts_q99", "restarts_max",
        "found_count", "not_found_count", "found_rate",
    ]
].sort_values(["regime", "k", "n", "m"], ignore_index=True)

summary.to_csv(SUMMARY_OUTPUT, index=False, float_format="%.6f")
print(f"Saved: {SUMMARY_OUTPUT}")


# ---------------------------
# comparison_exp2.csv
# ---------------------------

pair_keys = ["n", "k", "m", "f_id", "tb"]

u = metrics[metrics["regime"] == "unrestricted"].copy()
b = metrics[metrics["regime"] == "kmax10"].copy()

paired = u.merge(
    b,
    on=pair_keys,
    how="inner",
    suffixes=("_unrestricted", "_kmax10"),
)

paired["pair_id"] = np.arange(len(paired))

paired_metrics = [
    "support_recall", "support_precision",
    "signed_recall", "signed_precision",
    "sigma_weight", "weight_ratio",
    "false_positives", "false_negatives",
    "sign_errors", "restarts",
]

for col in paired_metrics:
    paired[f"delta_{col}"] = paired[f"{col}_kmax10"] - paired[f"{col}_unrestricted"]

paired["support_recall_improved"] = paired["delta_support_recall"] > 0
paired["support_recall_worsened"] = paired["delta_support_recall"] < 0
paired["support_recall_unchanged"] = paired["delta_support_recall"] == 0

paired["support_precision_improved"] = paired["delta_support_precision"] > 0
paired["support_precision_worsened"] = paired["delta_support_precision"] < 0
paired["support_precision_unchanged"] = paired["delta_support_precision"] == 0

# First average paired differences within each function.
paired_by_function = (
    paired.groupby(["n", "k", "m", "f_id"], as_index=False)
    .agg(
        pair_count=("pair_id", "count"),

        delta_support_recall_mean=("delta_support_recall", "mean"),
        delta_support_precision_mean=("delta_support_precision", "mean"),
        delta_signed_recall_mean=("delta_signed_recall", "mean"),
        delta_signed_precision_mean=("delta_signed_precision", "mean"),

        delta_sigma_weight_mean=("delta_sigma_weight", "mean"),
        delta_weight_ratio_mean=("delta_weight_ratio", "mean"),
        delta_false_positives_mean=("delta_false_positives", "mean"),
        delta_false_negatives_mean=("delta_false_negatives", "mean"),
        delta_restarts_mean=("delta_restarts", "mean"),

        support_recall_improved_rate=("support_recall_improved", "mean"),
        support_recall_unchanged_rate=("support_recall_unchanged", "mean"),
        support_recall_worsened_rate=("support_recall_worsened", "mean"),

        support_precision_improved_rate=("support_precision_improved", "mean"),
        support_precision_unchanged_rate=("support_precision_unchanged", "mean"),
        support_precision_worsened_rate=("support_precision_worsened", "mean"),
    )
)

comparison = (
    paired_by_function.groupby(["n", "k", "m"], as_index=False)
    .agg(
        f_count=("f_id", "count"),
        pair_count=("pair_count", "sum"),

        delta_support_recall_mean=("delta_support_recall_mean", "mean"),
        delta_support_recall_sd_functions=("delta_support_recall_mean", "std"),
        delta_support_precision_mean=("delta_support_precision_mean", "mean"),
        delta_support_precision_sd_functions=("delta_support_precision_mean", "std"),

        delta_signed_recall_mean=("delta_signed_recall_mean", "mean"),
        delta_signed_precision_mean=("delta_signed_precision_mean", "mean"),

        delta_sigma_weight_mean=("delta_sigma_weight_mean", "mean"),
        delta_weight_ratio_mean=("delta_weight_ratio_mean", "mean"),
        delta_false_positives_mean=("delta_false_positives_mean", "mean"),
        delta_false_negatives_mean=("delta_false_negatives_mean", "mean"),
        delta_restarts_mean=("delta_restarts_mean", "mean"),

        support_recall_improved_rate=("support_recall_improved_rate", "mean"),
        support_recall_unchanged_rate=("support_recall_unchanged_rate", "mean"),
        support_recall_worsened_rate=("support_recall_worsened_rate", "mean"),

        support_precision_improved_rate=("support_precision_improved_rate", "mean"),
        support_precision_unchanged_rate=("support_precision_unchanged_rate", "mean"),
        support_precision_worsened_rate=("support_precision_worsened_rate", "mean"),
    )
)

comparison["delta_support_recall_se_functions"] = (
    comparison["delta_support_recall_sd_functions"]
    / np.sqrt(comparison["f_count"])
)
comparison["delta_support_precision_se_functions"] = (
    comparison["delta_support_precision_sd_functions"]
    / np.sqrt(comparison["f_count"])
)
comparison["m_over_n"] = comparison["m"] / comparison["n"]
comparison["m_over_nk"] = comparison["m"] / (comparison["n"] * comparison["k"])

comparison = comparison[
    [
        "n", "k", "m", "m_over_n", "m_over_nk",
        "f_count", "pair_count",

        "delta_support_recall_mean", "delta_support_recall_sd_functions", "delta_support_recall_se_functions",
        "delta_support_precision_mean", "delta_support_precision_sd_functions", "delta_support_precision_se_functions",
        "delta_signed_recall_mean", "delta_signed_precision_mean",

        "delta_sigma_weight_mean", "delta_weight_ratio_mean",
        "delta_false_positives_mean", "delta_false_negatives_mean",
        "delta_restarts_mean",

        "support_recall_improved_rate", "support_recall_unchanged_rate", "support_recall_worsened_rate",
        "support_precision_improved_rate", "support_precision_unchanged_rate", "support_precision_worsened_rate",
    ]
].sort_values(["k", "n", "m"], ignore_index=True)

comparison.to_csv(COMPARISON_OUTPUT, index=False, float_format="%.6f")
print(f"Saved: {COMPARISON_OUTPUT}")


# ---------------------------
# sign_errors_exp2.csv
# ---------------------------

sign_error_runs = metrics[metrics["sign_errors"] > 0].copy()

sign_error_cols = [
    "regime", "n", "k", "m", "f_id", "tb",
    "support_true", "signs_true",
    "support_predicted", "signs_predicted",
    "support_tp", "signed_tp", "sign_errors",
]

sign_error_runs[sign_error_cols].to_csv(
    SIGN_ERRORS_OUTPUT,
    index=False,
    float_format="%.6f",
)
print(f"Saved: {SIGN_ERRORS_OUTPUT}")

