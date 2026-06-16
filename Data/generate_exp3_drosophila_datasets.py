# -*- coding: utf-8 -*-
"""
Data generator for Experiment 3: Drosophila Boolean network.

This script:
- loads the Drosophila Boolean network from src/biological_networks.py,
- samples global input states x in {0,1}^n,
- evaluates the complete network successor f(x),
- saves network-level datasets (X, Y) in Data/exp3_drosophila.

Each saved dataset contains paired observations (x, f(x)). The run script
will later split each dataset into node-level observations (x, f_i(x)).
"""

import sys
import numpy as np
from pathlib import Path


# ---------------------------
# Paths
# ---------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# If this script is inside Data/, the project root is one level above.
# Otherwise, assume the script is already at the project root.
if SCRIPT_DIR.name.lower() == "data":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drosophila_network import compiled_network, eval_network_state, nonconstant_nodes

# -----------------------
# Parameters
# -----------------------

NETWORK_NAME = "drosophila"
M_VALUES = [4, 8, 16, 32]
TB = 20

# ---------------------------
# Reproducibility
# ---------------------------

SEED = 1234
OUT_DIR = PROJECT_ROOT / "Data" / "exp3_drosophila"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dataset_seed(base, m, tb):
    return base + 1000 * m + tb


def evaluate_network_rows(network, X):
    Y = np.zeros((X.shape[0], len(network["nodes"])), dtype=np.uint8)
    for i in range(X.shape[0]):
        Y[i, :] = np.array(eval_network_state(network, X[i]), dtype=np.uint8)
    return Y


def all_targets_non_degenerate(Y, target_indices):
    for idx in target_indices:
        y = Y[:, idx]
        if not (np.any(y == 0) and np.any(y == 1)):
            return False
    return True


def generate_dataset(network, target_indices, m, seed, max_tries=10000):
    rng = np.random.default_rng(seed)
    n = len(network["nodes"])
    last_X = None
    last_Y = None

    for _ in range(max_tries):
        X = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        Y = evaluate_network_rows(network, X)

        if all_targets_non_degenerate(Y, target_indices):
            return X, Y, False

        last_X, last_Y = X, Y

    return last_X, last_Y, True


def main():
    network = compiled_network()
    nodes = network["nodes"]
    target_nodes = nonconstant_nodes(network)
    target_indices = [network["index"][node] for node in target_nodes]

    dataset_id = 0

    for m in M_VALUES:
        for tb in range(1, TB + 1):
            seed = dataset_seed(SEED, m, tb)

            X, Y, degenerate_after_retries = generate_dataset(
                network=network,
                target_indices=target_indices,
                m=m,
                seed=seed,
            )

            fname = f"exp3_drosophila_m{m}_tb{tb:02d}.npz"
            fpath = OUT_DIR / fname

            F0_sizes = []
            F1_sizes = []
            for idx in target_indices:
                y = Y[:, idx]
                F0_sizes.append(int(np.sum(y == 0)))
                F1_sizes.append(int(np.sum(y == 1)))

            np.savez(
                fpath,
                X=X,
                Y=Y,
                n=len(nodes),
                m=m,
                tb=tb,
                seed=seed,
                dataset_id=dataset_id,
                network_name=NETWORK_NAME,
                nodes=np.array(nodes, dtype=object),
                target_nodes=np.array(target_nodes, dtype=object),
                target_indices=np.array(target_indices, dtype=np.int64),
                F0_sizes=np.array(F0_sizes, dtype=np.int64),
                F1_sizes=np.array(F1_sizes, dtype=np.int64),
                degenerate_after_retries=int(degenerate_after_retries),
            )

            dataset_id += 1
            flag = " degenerate_after_retries" if degenerate_after_retries else ""
            print(f"[OK] {fname}{flag}")

    print("\nExperiment 3 Drosophila datasets generated.")


if __name__ == "__main__":
    main()
