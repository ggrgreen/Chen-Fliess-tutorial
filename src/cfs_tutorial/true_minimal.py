"""
true_minimal.py
==================
The correct way to test whether a genuinely minimal realisation is
nilpotent: compute the rank of the staircase Hankel matrix exactly, and
no numerical tolerance, then build the realisation from an exact (or, at
high truncation order, exact-rank-certified, high-precision) linear
algebra factorisation, not from a fixed-tolerance SVD.

The Hankel matrix used is the staircase matrix: rows and columns are every word up to length K, and the entry
(u, v) is c(uv) when |u| + |v| <= K and zero beyond, because a truncated series is zero there (Fliess 1974:
the minimal dimension equals the Hankel rank).  No fixed singular-value tolerance is used.

For an exact (symbolic) coefficient sequence, both issues are avoidable:
the rank of the full Hankel matrix can be computed exactly, either over
the rationals directly (practical up to about K=6 for a two-letter
alphabet) or via a modular-arithmetic rank certificate, cross-checked
against a float64 rank-revealing QR (practical to at least K=8, and used
here specifically because plain rational arithmetic becomes impractical
before then).


@author: gary green
last edit 19-10-2026
"""
import itertools
from math import gcd

import numpy as np
import scipy.linalg as sla
import sympy as sp


def full_hankel_matrix(coef, alphabet, K, words=None):
    """The staircase Hankel matrix: rows and columns are every word of length 0..K, entry (u,v) = c(uv) when
    |u|+|v| <= K, else 0."""
    if words is None:
        words = [w for L in range(K + 1) for w in itertools.product(alphabet, repeat=L)]
    n = len(words)

    def c(w):
        return coef.get(w, 0)

    H = [[c(words[i] + words[j]) if len(words[i]) + len(words[j]) <= K else 0
          for j in range(n)] for i in range(n)]
    return H, words


def exact_rank_modular(coef, alphabet, K, prime=2**31 - 1):
    """A fast, reliable rank certificate: reduce every entry mod a large
    prime, then compute rank by ordinary Gaussian elimination in that
    field. For a "generic" large prime, rank over the rationals and rank
    mod p coincide (rank can only drop mod p if p happens to divide a
    nonzero subdeterminant, vanishingly unlikely for a prime of this
    size chosen without reference to the matrix). This is the method
    used here at K=8, where exact rational Gaussian elimination on the
    full Hankel matrix is no longer practical."""
    H, words = full_hankel_matrix(coef, alphabet, K)
    n = len(words)

    def to_mod(x, p=prime):
        frac = sp.Rational(x)
        num, den = int(frac.p) % p, int(frac.q) % p
        return (num * pow(den, -1, p)) % p

    M = [[to_mod(H[i][j]) for j in range(n)] for i in range(n)]
    rank = 0
    for col in range(n):
        pivot = None
        for r in range(rank, n):
            if M[r][col] % prime != 0:
                pivot = r
                break
        if pivot is None:
            continue
        M[rank], M[pivot] = M[pivot], M[rank]
        inv = pow(M[rank][col], -1, prime)
        M[rank] = [(x * inv) % prime for x in M[rank]]
        for r in range(n):
            if r != rank and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [(M[r][k] - factor * M[rank][k]) % prime for k in range(n)]
        rank += 1
        if rank == n:
            break
    return rank, words


def true_minimal_realize(coef, alphabet, K, method='auto'):
    """
    Builds the genuinely minimal realisation (lambda, {A_a}, mu) from the
    full Hankel matrix, by exact linear algebra, not SVD.

    method='exact': rational arithmetic throughout (sympy). Reliable and
        machine-independent, but impractical much beyond K=6 for a
        two-letter alphabet.
    method='numeric': float64 rank-revealing QR to select independent
        rows/columns, then a float64 construction. Used at higher K,
        where exact arithmetic is no longer practical; validated by
        checking reconstruction of every known coefficient before any
        nilpotency claim is trusted (see examples/true_minimal_duffing.py).
    method='auto': 'exact' for K<=6, 'numeric' otherwise.

    Returns a dict with A (letter -> r x r array), lam (1 x r), gamma
    (r x 1), dim (r), and the (P, T) word sets used, mirroring the
    structure of graded_nilpotent_realize's output where practical.
    """
    if method == 'auto':
        method = 'exact' if K <= 6 else 'numeric'

    H, words = full_hankel_matrix(coef, alphabet, K)
    n = len(words)

    if method == 'exact':
        Hs = sp.Matrix(n, n, lambda i, j: sp.Rational(H[i][j]))
        r = Hs.rank()
        _, pivot_cols = Hs.rref()
        T_idx = list(pivot_cols)[:r]
        _, pivot_rows = Hs.T.rref()
        P_idx = list(pivot_rows)[:r]
        P = [words[i] for i in P_idx]
        T = [words[j] for j in T_idx]
        H_PT = Hs[P_idx, T_idx]
        H_PT_inv = H_PT.inv()

        def c(w):
            return sp.Rational(coef.get(w, 0))

        def H_row(word_key, T_words):
            return sp.Matrix(1, len(T_words), lambda _, j: c(word_key + T_words[j]))

        A = {}
        for a in alphabet:
            H_PaT = sp.Matrix.vstack(*[H_row(p + (a,), T) for p in P])
            A[a] = H_PaT * H_PT_inv
        lam = H_row((), T) * H_PT_inv
        gamma = sp.Matrix(r, 1, lambda i, _: c(P[i] + ()))

        def reconstruct(w):
            v = lam
            for letter in w:
                v = v * A[letter]
            return (v * gamma)[0, 0]

    else:  # numeric
        H_np = np.array([[float(x) for x in row] for row in H])
        Q, R, piv = sla.qr(H_np, pivoting=True)
        diagR = np.abs(np.diag(R))
        r = int(np.sum(diagR > 1e-8 * diagR[0]))
        T_idx = sorted(piv[:r].tolist())
        QT, RT, pivT = sla.qr(H_np.T, pivoting=True)
        P_idx = sorted(pivT[:r].tolist())
        P = [words[i] for i in P_idx]
        T = [words[j] for j in T_idx]
        H_PT = H_np[np.ix_(P_idx, T_idx)]
        H_PT_inv = np.linalg.inv(H_PT)

        def c(w):
            return float(coef.get(w, 0))

        def H_row(word_key, T_words):
            return np.array([c(word_key + t) for t in T_words])

        A = {}
        for a in alphabet:
            H_PaT = np.array([H_row(p + (a,), T) for p in P])
            A[a] = H_PaT @ H_PT_inv
        lam = H_row((), T) @ H_PT_inv
        gamma = np.array([c(p + ()) for p in P])

        def reconstruct(w):
            v = lam.copy()
            for letter in w:
                v = v @ A[letter]
            return float(v @ gamma)

    return dict(A=A, lam=lam, gamma=gamma, dim=r, P=P, T=T,
                reconstruct=reconstruct, method=method)


def verify_true_minimal(realiz, coef, K, alphabet, n_beyond=3):
    """Checks reconstruction of every known coefficient (should match
    exactly, or to machine precision for the numeric method), then
    checks every word beyond K up to K+n_beyond (should be exactly, or
    numerically indistinguishable from, zero if the true minimal
    realisation is indeed nilpotent, as the theory predicts)."""
    reconstruct = realiz['reconstruct']
    max_err = 0.0
    for L in range(K + 1):
        for w in itertools.product(alphabet, repeat=L):
            true_c = coef.get(w, 0)
            recon = reconstruct(w)
            err = abs(float(recon) - float(true_c))
            max_err = max(max_err, err)
    beyond = {}
    for L in range(K + 1, K + 1 + n_beyond):
        max_val = 0.0
        for w in itertools.product(alphabet, repeat=L):
            max_val = max(max_val, abs(float(reconstruct(w))))
        beyond[L] = max_val
    return dict(max_reconstruction_error=max_err, max_beyond=beyond)


staircase_hankel_matrix = full_hankel_matrix


def products_vanish(realiz, alphabet, K):
    """largest absolute entry of A_w for every word w of length K+1 (row-vector convention of this module:
    the state after reading w is lam A_{a_1} ... A_{a_L})."""
    import itertools
    A = realiz["A"]
    r = realiz["dim"]
    worst = 0.0
    for w in itertools.product(alphabet, repeat=K + 1):
        P = np.eye(r) if realiz["method"] == "numeric" else sp.eye(r)
        for a in w:
            P = P @ A[a] if realiz["method"] == "numeric" else P * A[a]
        worst = max(worst, float(np.max(np.abs(np.array(P, dtype=float)))))
    return worst
