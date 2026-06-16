# -*- coding: utf-8 -*-
"""
Reproducible experimental runner for Experiment 2 with k_max = 10.

This variant constrains Algorithm 1 to produce coverage vectors of weight
at most 10, reflecting the standard biological prior that gene regulatory
in-degrees are typically below this threshold (cf. Kadelka & Hari, 2025).

Compared to the unconstrained run (run_exp2.py with k_max=None), this run
may exhibit:
  - non-zero restarts when the greedy procedure exceeds the weight bound,
  - instances where no coverage vector is found within the stop condition
    (Sigma=None, solution_found=False).

Both outcomes are recorded in the raw output for downstream analysis.

Outputs go to Results/raw_exp2_kmax10.csv.
"""

import csv
import os
import json
import numpy as np
from pathlib import Path

from src.io_utils import validate_observations
from src.coverage import coverage_algorithm

# ---------------------------
# Parameters
# ---------------------------

K_MAX = 10  
STOP_FACTOR = 5.0

N_VALUES = [10, 50, 100]
K_VALUES = [2, 3, 4, 5, 6]

M_BY_K = {
    2: [2, 4, 8],
    3: [4, 8, 16],
    4: [8, 16, 32],
    5: [16, 32, 64],
    6: [32, 64, 128],
}

N_FUNCTIONS = 15
TB = 5

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "exp2"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT = RESULTS_DIR / "raw_exp2_kmax10.csv"

# ---------------------------
# Helper
# ---------------------------

def check_coverage(D, Sigma, n):
    """
    Verify externally that Sigma covers all discrepancies in D.
    """
    if Sigma is None:
        return False

    for delta in D:
        covered = False
        for j in range(n):
            if Sigma[j] != "0" and delta[j] == Sigma[j]:
                covered = True
                break
        if not covered:
            return False
    return True


# ---------------------------
# Initialize output
# ---------------------------

with open(RAW_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "n",
        "k",
        "m",
        "f_id",
        "tb",
        "function_seed",
        "embedding_seed",
        "dataset_seed",
        "k_max_used",
        "F0_size",
        "F1_size",
        "d",
        "support_true",
        "signs_true",
        "antichain_masks",
        "antichain_sets",
        "sigma_weight",
        "support_predicted",
        "signs_predicted",
        "restarts",
        "solution_found",
    ])

# ---------------------------
# Main loop
# ---------------------------

for k in K_VALUES:
    for f_id in range(1, N_FUNCTIONS + 1):
        for n in N_VALUES:
            for m in M_BY_K[k]:
                for tb in range(1, TB + 1):

                    data_file = DATA_DIR / f"exp2_n{n}_k{k}_m{m}_f{f_id:02d}_tb{tb:02d}.npz"

                    if not os.path.exists(data_file):
                        print(f"[WARN] Missing file {data_file}, skipping.")
                        continue

                    try:
                        # Load dataset
                        data = np.load(data_file, allow_pickle=True)
                        X = data["X"]
                        y = data["y"]

                        support_true = [int(i) for i in data["support"]]
                        signs_true = [int(s) for s in data["signs"]]
                        antichain = [int(a) for a in data["antichain"]]

                        f_seed = int(data["function_seed"])
                        e_seed = int(data["embedding_seed"])
                        d_seed = int(data["seed"])

                        F0 = X[y == 0]
                        F1 = X[y == 1]

                        F0_size = len(F0)
                        F1_size = len(F1)

                        # Validate mathematical assumptions on (F0, F1)
                        validate_observations(F0, F1, n=n, m=m)

                        # Algorithm 1 with k_max constraint
                        D, Sigma, info = coverage_algorithm(
                            F0, F1, n,
                            k_max=K_MAX,
                            seed=d_seed,
                            prune=True,
                            stop_params={"factor": STOP_FACTOR},
                        )

                        # External verification (only when a Sigma was returned)
                        if Sigma is not None and not check_coverage(D, Sigma, n):
                            raise RuntimeError(
                                f"Coverage check failed for n={n}, k={k}, m={m}, "
                                f"f={f_id}, tb={tb}"
                            )

                        # Decompose Sigma into support and signs (handles None case)
                        if Sigma is not None:
                            support_pred = [i for i in range(n) if Sigma[i] != "0"]
                            signs_pred = [1 if Sigma[i] == "+" else -1 for i in support_pred]
                        else:
                            support_pred = []
                            signs_pred = []

                        # Decode antichain bitmasks into readable index sets
                        antichain_sets = [
                            [j for j in range(k) if (mask >> j) & 1]
                            for mask in antichain
                        ]

                        d = info.get("d", None)
                        restarts = info.get("restarts", None)
                        found = info.get("found", False)
                        sigma_weight = info.get("sigma_weight", None)

                        row = [
                            n,
                            k,
                            m,
                            f_id,
                            tb,
                            f_seed,
                            e_seed,
                            d_seed,
                            K_MAX,
                            F0_size,
                            F1_size,
                            d,
                            json.dumps(support_true, separators=(",", ":")),
                            json.dumps(signs_true, separators=(",", ":")),
                            json.dumps(antichain, separators=(",", ":")),
                            json.dumps(antichain_sets, separators=(",", ":")),
                            sigma_weight,
                            json.dumps(support_pred, separators=(",", ":")),
                            json.dumps(signs_pred, separators=(",", ":")),
                            restarts,
                            found,
                        ]

                        with open(RAW_OUTPUT, "a", newline="") as fcsv:
                            writer = csv.writer(fcsv)
                            writer.writerow(row)

                        status = "OK " if found else "NOT"
                        print(
                            f"[{status}] n={n}, k={k}, m={m}, f={f_id}, tb={tb} | "
                            f"w(Sigma)={sigma_weight} | restarts={restarts} | "
                            f"found={found}"
                        )

                    except Exception as e:
                        print(
                            f"[ERROR] n={n}, k={k}, m={m}, f={f_id}, tb={tb} -> {e}"
                        )
                        continue

print(f"\nRaw results saved incrementally to {RAW_OUTPUT}")
