# -*- coding: utf-8 -*-
"""
Data generate experiment 1

This script: 
- generates balanced random observation sets (F0, F1), 
- ensures that all sampled binary rows are unique, 
- saves datasets in Data/exp1 as .npz files.
"""

import numpy as np
from pathlib import Path

# -----------------------
# Parameters
# -----------------------

N_VALUES = [500, 1000, 2000, 3000, 4000, 5000]
M_VALUES = [50, 100, 150, 200, 250, 300]
TB = 10

# ---------------------------
# Reproducibility
# ---------------------------

SEED = 1234
DATA_ROOT = Path(__file__).resolve().parent 
OUT_DIR = DATA_ROOT / "exp1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Main functions
# -----------------------

def generate_unique_binary_rows(num_rows, n, existing_hashes=None, rng=None):
    """
    Generates num_rows unique binary rows of length n,
    avoiding collisions with existing_hashes if provided.
    """
    if rng is None:
        rng = np.random.default_rng()

    if existing_hashes is None:
        existing_hashes = set()

    rows = []
    hashes = set(existing_hashes)

    while len(rows) < num_rows:
        x = rng.integers(0, 2, size=n, dtype=np.uint8)
        h = x.tobytes()
        if h not in hashes:
            hashes.add(h)
            rows.append(x)

    return np.array(rows, dtype=np.uint8), hashes


def generate_dataset(n, m, seed):
    rng = np.random.default_rng(seed)

    m0 = m // 2
    m1 = m - m0

    # Generate F0
    F0, hashes = generate_unique_binary_rows(num_rows=m0, n=n, existing_hashes=None, rng=rng)

    # Generate F1
    F1, _ = generate_unique_binary_rows(num_rows=m1, n=n, existing_hashes=hashes, rng=rng)

    # Concatenate
    X = np.vstack([F0, F1])
    y = np.array([0] * m0 + [1] * m1, dtype=np.uint8)

    return X, y


def main():
    dataset_id = 0

    for n in N_VALUES:
        for m in M_VALUES:
            for tb in range(1, TB + 1):

                seed = SEED + 100000 * n + 1000 * m + tb

                X, y = generate_dataset(n, m, seed)

                fname = f"exp1_n{n}_m{m}_tb{tb:02d}.npz"
                fpath = OUT_DIR / fname

                np.savez(fpath, X=X, y=y, n=n, m=m, seed=seed, dataset_id=dataset_id)

                dataset_id += 1

                print(f"[OK] {fname}")

    print("\nGeneración de datasets E1 completa.")


if __name__ == "__main__":
    main()



