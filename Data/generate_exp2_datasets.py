# -*- coding: utf-8 -*-
"""
Data generate experiment 2

This script:
- generates random unate Boolean functions with exactly k relevant variables,
- embeds each function into ambient spaces of size n by fixing a support,
- samples observation datasets of size m from the embedded function,
- produces multiple independent datasets (TB repetitions) per configuration,
- saves datasets in Data/exp2 as .npz files.
"""

import numpy as np
import math
from pathlib import Path

# ---------------------------
# Parameters
# ---------------------------

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

# ---------------------------
# Reproducibility
# ---------------------------

SEED = 1234
DATA_ROOT = Path(__file__).resolve().parent 
OUT_DIR = DATA_ROOT / "exp2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Antichain utilities
# ---------------------------

ANTICHAINS_K2 = [
    [1, 2],  # {{0}, {1}}
    [3],     # {{0,1}}
]

ANTICHAINS_K3 = [ 
    [1,2,4], # {{0},{1},{2}} 
    [1,6], # {{0},{1,2}} 
    [2,5], # {{1},{0,2}} 
    [4,3], # {{2},{0,1}} 
    [3,5], # {{0,1},{0,2}} 
    [3,6], # {{0,1},{1,2}} 
    [5,6], # {{0,2},{1,2}} 
    [3,5,6], # {{0,1},{0,2},{1,2}} 
    [7], # {{0,1,2}} 
]


def comparable(a, b):
    """
    True if a is subset of b or b is subset of a.
    """
    return (a & b) == a or (b & a) == b


def covers_all(A, k):
    """
    True if the union of the elements of A covers all variables {0,...,k-1}.
    """
    full = (1 << k) - 1
    cover = 0
    for m in A:
        cover |= m
    return cover == full


def max_antichain_size(k):
    """
    Maximum possible size of an antichain in P([k]).
    """
    return math.comb(k, k // 2)


def sample_antichain(k, rng):
    """
    Random greedy antichain generator.
    """
    t_max = max_antichain_size(k)
    t = rng.integers(1, t_max + 1)

    U = list(range(1, 1 << k))
    A = []
    alive = U[:]

    while alive:
        i = rng.integers(0, len(alive))
        s = alive.pop(i)

        if all(not comparable(s, a) for a in A):
            A.append(s)
            alive = [x for x in alive if not comparable(x, s)]

            if covers_all(A, k) and (len(A) >= t or not alive):
                break
    return A

# ---------------------------
# Function construction
# ---------------------------

def monotone_function_from_antichain(A):
    """ 
    Returns g(x_mask)=1 iff some a in A is subset of x_mask. 
    """ 
    def g(x_mask):
        for a in A:
            if (x_mask & a) == a:
                return 1
        return 0
    return g


def generate_base_function(k, seed):
    """
    Samples a random monotone Boolean function on k variables via an antichain,
    and assigns random signs to obtain a unate function.
    """
    rng = np.random.default_rng(seed)

    if k == 2:
        A = ANTICHAINS_K2[rng.integers(0, len(ANTICHAINS_K2))]
    elif k == 3:
        A = ANTICHAINS_K3[rng.integers(0, len(ANTICHAINS_K3))]
    else:
        A = sample_antichain(k, rng)

    # signs in {+1, -1}
    signs = rng.choice([-1, 1], size=k)

    g = monotone_function_from_antichain(A)

    return A, signs, g


def embed_base_function(n, k, seed):
    """
    Samples a support of size k in [n], embedding the k relevant variables
    into an n-dimensional space.
    """
    rng = np.random.default_rng(seed)
    support = rng.choice(n, size=k, replace=False)
    return support

# ---------------------------
# Evaluation
# ---------------------------

def eval_unate_row(x, support, signs, g):
    """
    Evaluates the embedded unate function on input x by restricting to the support,
    applying signs, and evaluating the base monotone function g.
    """
    mask = 0
    for idx, var in enumerate(support):
        bit = x[var]

        if signs[idx] == -1:
            bit = 1 - bit

        if bit == 1:
            mask |= (1 << idx)

    return g(mask)

# ---------------------------
# Dataset generation
# ---------------------------

def generate_dataset(n, m, support, signs, g, seed, max_tries=20):
    """
    Generates m random binary samples in {0,1}^n and labels them using
    the embedded unate function defined by (support, signs, g).
    """
    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        X = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        y = np.zeros(m, dtype=np.uint8)

        for i in range(m):
            y[i] = eval_unate_row(X[i], support, signs, g)

        if np.any(y == 0) and np.any(y == 1):
            F0_size = int(np.sum(y == 0))
            F1_size = int(np.sum(y == 1))
            return X, y, F0_size, F1_size

    # fallback: aceptar igual (caso extremadamente raro)
    F0_size = int(np.sum(y == 0))
    F1_size = int(np.sum(y == 1))
    return X, y, F0_size, F1_size


# ---------------------------
# Seed helpers
# ---------------------------

def function_seed(base, k, f):
    return base + 10000 * k + f


def embedding_seed(base, k, f, n):
    return base + 1_000_000 * k + 10_000 * f + n


def dataset_seed(base, k, f, n, m, tb):
    return base + 100_000_000 * k + 1_000_000 * f + 10_000 * n + 100 * m + tb


# ---------------------------
# Main
# ---------------------------

def main():
    dataset_id = 0
    
    summary_rows = []

    for k in K_VALUES:

        for f in range(1, N_FUNCTIONS + 1):

            f_seed = function_seed(SEED, k, f)
            A, signs, g = generate_base_function(k, f_seed)

            for n in N_VALUES:

                e_seed = embedding_seed(SEED, k, f, n)
                support = embed_base_function(n, k, e_seed)

                for m in M_BY_K[k]:

                    for tb in range(1, TB + 1):

                        d_seed = dataset_seed(SEED, k, f, n, m, tb)

                        X, y, F0_size, F1_size = generate_dataset(
                            n, m, support, signs, g, d_seed
                        )
                        
                        summary_rows.append([
                            k,
                            f,
                            n,
                            m,
                            tb,
                            F0_size,
                            F1_size,
                        ])

                        fname = f"exp2_n{n}_k{k}_m{m}_f{f:02d}_tb{tb:02d}.npz"
                        fpath = OUT_DIR / fname

                        np.savez(
                            fpath,
                            X=X,
                            y=y,
                            n=n,
                            k=k,
                            m=m,
                            f_id=f,
                            tb=tb,
                            seed=d_seed,
                            function_seed=f_seed,
                            embedding_seed=e_seed,
                            support=support,
                            signs=signs,
                            antichain=A,
                            F0_size=F0_size,
                            F1_size=F1_size,
                            dataset_id=dataset_id,
                        )

                        dataset_id += 1

                        print(f"[OK] {fname}")

    print("\nExp2 dataset generation complete.")


if __name__ == "__main__":
    main()
