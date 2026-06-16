# -*- coding: utf-8 -*-
"""
Reproducible experimental runner for Experiment 1.

This script:
- reads datasets from Data/exp1,
- runs Algorithm 1 on each dataset,
- writes only raw per-instance results to Results/raw_exp1.csv.
"""

import csv
import os
import time
import numpy as np
from pathlib import Path

from src.io_utils import validate_observations
from src.coverage import coverage_algorithm

# ---------------------------
# Parameters
# ---------------------------

N_VALUES = [500, 1000, 2000, 3000, 4000, 5000]
M_VALUES = [50, 100, 150, 200, 250, 300]
TB = 10

PROJECT_ROOT = Path(__file__).resolve().parent 
DATA_DIR = PROJECT_ROOT / "Data" / "exp1" 
RESULTS_DIR = PROJECT_ROOT / "Results" 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT = RESULTS_DIR / "raw_exp1.csv" 

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
        "m",
        "tb",
        "seed",
        "F0_size",
        "F1_size",
        "d",
        "time_sec",
        "restarts",
        "solution_found",
        "sigma_weight",
    ])

# ---------------------------
# Main loop
# ---------------------------

for n in N_VALUES:
    for m in M_VALUES:
        for tb in range(1, TB + 1):

            data_file = DATA_DIR / f"exp1_n{n}_m{m}_tb{tb:02d}.npz"

            if not os.path.exists(data_file):
                print(f"[WARN] Missing file {data_file}, skipping.")
                continue

            try:
                # Load dataset
                data = np.load(data_file)
                X = data["X"]
                y = data["y"]
                seed = int(data["seed"])

                F0 = X[y == 0]
                F1 = X[y == 1]
                
                F0_size = len(F0)
                F1_size = len(F1)

                # Validate mathematical assumptions on (F0, F1)
                validate_observations(F0, F1, n=n, m=m)

                # Timed execution: only Algorithm 1
                start = time.perf_counter()
                D, Sigma, info = coverage_algorithm(
                    F0, F1, n, k_max=None, seed=seed, prune=False
                )
                end = time.perf_counter()

                elapsed = end - start

                # External verification (only when a Sigma was returned)
                if Sigma is not None and not check_coverage(D, Sigma, n):
                    raise RuntimeError(
                        f"Coverage check failed for n={n}, m={m}, tb={tb}"
                    )

                d = info.get("d", None)
                restarts = info.get("restarts", None)
                found = info.get("found", None)
                sigma_weight = info.get("sigma_weight", None)

                row = [
                    n,
                    m,
                    tb,
                    seed,
                    F0_size,
                    F1_size,
                    d,
                    round(elapsed, 6),
                    restarts,
                    found,
                    sigma_weight,
                ]

                with open(RAW_OUTPUT, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(
                    f"[OK] n={n}, m={m}, tb={tb} | "
                    f"time={elapsed:.4f}s | d={d} | found={found}"
                )

            except Exception as e:
                print(f"[ERROR] n={n}, m={m}, tb={tb} -> {e}")
                continue

print(f"\nRaw results saved incrementally to {RAW_OUTPUT}")
