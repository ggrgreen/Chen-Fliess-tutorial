"""
hankel.py
=========
The realisation of a truncated Chen-Fliess series read off its staircase Hankel matrix by singular value
decomposition, for comparison with the graded construction of realization.py.

Convention (as in realization.py): x_() = lam, x_{w+(a,)} = A_a x_w, c_w = mu . x_w; the Hankel matrix has entry
H[u, v] = c(u v), the word u read first.  With H = P Q (P = U sqrt(S), Q = sqrt(S) V^T, rank r chosen by the
relative tolerance tau), A_a = (P^+ H_a Q^+)^T, lam = P[(), :], mu = Q[:, ()], where H_a[u, v] = c(u a v).

Staircase Hankel matrix: rows and columns are every word of length 0..K; the entry (u, v) is c(uv) when
|u| + |v| <= K and zero otherwise, because a truncated series is zero beyond K.  The non-zero entries form a
staircase: a row word of length l has non-zero entries only in columns of length <= K - l.


@author: gary green
last edit 19-10-2026
"""
import itertools
import numpy as np


def _words(m, L):
    return [w for l in range(L + 1) for w in itertools.product(range(m + 1), repeat=l)]


def _realise(coef, m, K, rows, cols, tau):
    c = lambda w: float(coef.get(w, 0.0)) if len(w) <= K else 0.0
    H = np.array([[c(u + v) for v in cols] for u in rows])
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    r = int(np.sum(S > tau * S[0])) if S.size and S[0] > 0 else 0
    if r == 0:
        return dict(A=[np.zeros((0, 0))] * (m + 1), lam=np.zeros(0), mu=np.zeros(0), dim_total=0, sv=S)
    P = U[:, :r] * np.sqrt(S[:r]); Q = np.sqrt(S[:r])[:, None] * Vt[:r]
    Pp, Qp = np.linalg.pinv(P), np.linalg.pinv(Q)
    A = []
    for a in range(m + 1):
        Ha = np.array([[c(u + (a,) + v) for v in cols] for u in rows])
        A.append((Pp @ Ha @ Qp).T)
    return dict(A=A, lam=P[rows.index(())].copy(), mu=Q[:, cols.index(())].copy(), dim_total=r, sv=S)


def staircase_realize(coef, m, K, tau):
    W = _words(m, K)
    return _realise(coef, m, K, W, W, tau)


def coefficient(real, w):
    x = np.asarray(real["lam"], float).copy()
    for a in w:
        x = real["A"][a] @ x
    return float(np.asarray(real["mu"], float) @ x)


def max_coef_error(real, coef, m, K):
    return max(abs(coefficient(real, w) - float(coef.get(w, 0.0))) for w in _words(m, K))


def nilpotency_residual(real, m, K):
    """largest operator norm of a product of K + 1 generators (zero for an exactly nilpotent realisation)."""
    n = real["dim_total"]
    if n == 0:
        return 0.0
    best = 0.0
    for combo in itertools.product(range(m + 1), repeat=K + 1):
        P = np.eye(n)
        for a in combo:
            P = real["A"][a] @ P
        best = max(best, np.linalg.norm(P, 2))
    return best


def beyond_K(real, m, K, L=2):
    """largest |coefficient| the realisation assigns to words of length K+1..K+L (zero for the truncated series)."""
    return max(abs(coefficient(real, w)) for l in range(K + 1, K + L + 1) for w in itertools.product(range(m + 1), repeat=l))
