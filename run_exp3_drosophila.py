# -*- coding: utf-8 -*-
"""
Reproducible experimental runner for Experiment 3: Drosophila.

This script:
- reads network-level datasets (X, f(X)) from Data/exp3_drosophila,
- splits them into node-level datasets (X, f_i(X)),
- runs Algorithm 1 on each nonconstant update rule,
- writes only raw per-instance results to Results/raw_exp3_drosophila.csv.
"""

import csv
import os
import json
import numpy as np
from pathlib import Path

from src.io_utils import validate_observations
from src.coverage import coverage_algorithm
from src.drosophila_network import compiled_network, node_metadata, nonconstant_nodes

# ---------------------------
# Parameters
# ---------------------------

M_VALUES = [4, 8, 16, 32]
TB = 20

K_MAX = 10
STOP_PARAMS = {"factor": 5}
PRUNE = True

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "exp3_drosophila"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT = RESULTS_DIR / "raw_exp3_drosophila.csv"

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
# Network metadata
# ---------------------------

network = compiled_network()
nodes = network["nodes"]
n = len(nodes)
target_nodes = nonconstant_nodes(network)

metadata = {
    node: node_metadata(network, node)
    for node in target_nodes
}

# ---------------------------
# Initialize output
# ---------------------------

with open(RAW_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
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
    ])

# ---------------------------
# Main loop
# ---------------------------

for m in M_VALUES:
    for tb in range(1, TB + 1):

        data_file = DATA_DIR / f"exp3_drosophila_m{m}_tb{tb:02d}.npz"

        if not os.path.exists(data_file):
            print(f"[WARN] Missing file {data_file}, skipping.")
            continue

        try:
            # Load network-level dataset
            data = np.load(data_file, allow_pickle=True)
            X = data["X"]
            Y = data["Y"]

            dataset_id = int(data["dataset_id"])
            dataset_seed = int(data["seed"])

        except Exception as e:
            print(f"[ERROR] Could not load {data_file} -> {e}")
            continue

        for node in target_nodes:

            node_id = network["index"][node]
            meta = metadata[node]

            y = Y[:, node_id]

            F0 = X[y == 0]
            F1 = X[y == 1]

            F0_size = len(F0)
            F1_size = len(F1)

            try:
                # Validate mathematical assumptions on (F0, F1)
                validate_observations(F0, F1, n=n, m=m)

                # Algorithm seed is computed inside the run, from the dataset
                # seed and the target node index.
                algorithm_seed = dataset_seed + 100_000 * node_id

                # Algorithm 1
                D, Sigma, info = coverage_algorithm(
                    F0,
                    F1,
                    n,
                    k_max=K_MAX,
                    seed=algorithm_seed,
                    stop_params=STOP_PARAMS,
                    prune=PRUNE,
                )

                # External verification
                if Sigma is not None and not check_coverage(D, Sigma, n):
                    raise RuntimeError(
                        f"Coverage check failed for node={node}, m={m}, tb={tb}"
                    )

                # Decompose Sigma into predicted support and signs
                if Sigma is not None:
                    support_pred = [i for i in range(n) if Sigma[i] != "0"]
                    signs_pred = [
                        1 if Sigma[i] == "+" else -1
                        for i in support_pred
                    ]
                else:
                    support_pred = []
                    signs_pred = []

                # True support and signs
                support_true = meta["support"]
                signs_true = meta["signs"]
                k = meta["true_k"]

                d = info.get("d", None)
                restarts = info.get("restarts", None)
                found = info.get("found", None)
                sigma_weight = info.get("sigma_weight", None)

                row = [
                    node,
                    n,
                    k,
                    m,
                    tb,
                    dataset_id,
                    dataset_seed,
                    F0_size,
                    F1_size,
                    d,
                    json.dumps(support_true, separators=(",", ":")),
                    json.dumps(signs_true, separators=(",", ":")),
                    sigma_weight,
                    json.dumps(support_pred, separators=(",", ":")),
                    json.dumps(signs_pred, separators=(",", ":")),
                    restarts,
                    found,
                ]

                with open(RAW_OUTPUT, "a", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(row)

                print(
                    f"[OK] node={node}, m={m}, tb={tb} | "
                    f"k={k} | w(Sigma)={sigma_weight} | "
                    f"restarts={restarts} | found={found}"
                )

            except Exception as e:
                print(f"[ERROR] node={node}, m={m}, tb={tb} -> {e}")
                continue

print(f"\nRaw results saved incrementally to {RAW_OUTPUT}")
