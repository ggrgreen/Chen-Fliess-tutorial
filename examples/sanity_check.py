"""
sanity_check.py
==================
A ground-truth bilinear system with true state dimension 4, alphabet
{0,1}, and strictly upper-triangular generators (nilpotent of index <=4,
i.e. order K=3), constructed with random Gaussian entries. Its exact
coefficients for |w|<=3 are computed directly from this system, then
handed to the graded construction with no further information about the
true dimension, to check both exact reconstruction and how close the
recovered dimension comes to the true minimum.


@author: gary green
last edit 19-10-2026
"""
import itertools
import numpy as np

from cfs_tutorial import graded_nilpotent_realize, verify_exact, verify_nilpotent

SEED = 0
N_TRUE = 4
ORDER = 3


def make_ground_truth(seed=SEED, n=N_TRUE):
    rng = np.random.default_rng(seed)
    A0 = np.triu(rng.normal(size=(n, n)), k=1)
    A1 = np.triu(rng.normal(size=(n, n)), k=1)
    lam = rng.normal(size=n)
    mu = rng.normal(size=n)
    return A0, A1, lam, mu


def true_coeffs(A0, A1, lam, mu, order):
    A = {0: A0, 1: A1}
    coef = {}
    for L in range(0, order + 1):
        for w in itertools.product([0, 1], repeat=L):
            x = lam.copy()
            for a in w:
                x = A[a] @ x
            coef[w] = float(mu @ x)
    return coef


if __name__ == '__main__':
    A0, A1, lam, mu = make_ground_truth()
    coef = true_coeffs(A0, A1, lam, mu, ORDER)

    word_tree_dim = 2 ** (ORDER + 1) - 1
    realiz = graded_nilpotent_realize(coef, m=1, order=ORDER, tol=1e-8)
    err = verify_exact(realiz, coef, m=1, order=ORDER)
    nilp = verify_nilpotent(realiz, order=ORDER, m=1)

    print(f"True state dimension: {N_TRUE}")
    print(f"Word-tree (unreduced) dimension: {word_tree_dim}")
    print(f"Graded realisation dimension: {realiz['dim_total']}")
    print(f"Reconstruction error: {err:.3e}")
    print(f"Nilpotency: {nilp}")
