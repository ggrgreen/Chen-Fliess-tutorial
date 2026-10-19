"""
lyndon.py -- Lyndon words and the two places the Lyndon basis enters a truncated Chen-Fliess series.

Words are tuples over the alphabet {0 (drift), 1, ..., m}, ordered 0 < 1 < ... < m, first letter earliest in time.

1. Lie-Hankel matrix (rows indexed by Lie polynomials).
   Each Lyndon word l has a standard bracketing P_l, a Lie polynomial (a combination of words of length |l|).
   The P_l, l Lyndon, form a basis of the free Lie algebra (Lyndon / Chen-Fox-Lyndon / Viennot).
   With c(P v) = sum_w P_l(w) c(w v), the matrix with rows P_l and columns all words v is the Lie-Hankel matrix.
   For c the series of a system x' = f_0(x) + sum u_a f_a(x), y = h(x), c(P v) = (dL_v h)(x0) . F_P(x0),
   where F_P is the bracket of vector fields named by P, so its rank is at most dim x (Fliess 1983).
   Truncation (entries with |l| + |v| > K set to 0) changes this, which is what lie_hankel_matrix measures.

2. Radford (shuffle) basis of the regression.
   Every word w factorises uniquely as w = l_1^{n_1} ... l_k^{n_k}, l_1 > ... > l_k Lyndon (Chen-Fox-Lyndon).
   For iterated integrals of a path the shuffle identity I_u I_v = sum_{w in u sh v} I_w holds, so
       R_w = I_{l_1}^{n_1} ... I_{l_k}^{n_k} / (n_1! ... n_k!)
   is a polynomial in Lyndon iterated integrals and equals I_w plus a combination of other words of the same
   length (Radford 1979; Reutenauer 1993, Thm 6.1).  The R_w span exactly the same functions as the I_w.


@author: gary green
last edit 19-10-2026
"""
import itertools
from math import factorial
from collections import defaultdict
import numpy as np


def is_lyndon(w):
    """True if w is non-empty and strictly smaller than each of its proper rotations."""
    n = len(w)
    if n == 0:
        return False
    return all(w < w[i:] + w[:i] for i in range(1, n))


def lyndon_words(m, K, min_len=1):
    """Lyndon words over {0..m} of length min_len..K, by length then lexicographically."""
    return [w for L in range(min_len, K + 1) for w in itertools.product(range(m + 1), repeat=L) if is_lyndon(w)]


def standard_factorisation(l):
    """l = (u, v) with v the longest proper suffix of l that is Lyndon; then u is Lyndon too."""
    for i in range(1, len(l)):
        if is_lyndon(l[i:]):
            return l[:i], l[i:]
    raise ValueError("a single letter has no standard factorisation")


def _mul(p, q):
    out = defaultdict(float)
    for a, x in p.items():
        for b, y in q.items():
            out[a + b] += x * y
    return out


def bracket_polynomial(l, _cache={}):
    """Standard bracketing P_l of a Lyndon word, as {word: coefficient}.  P_a = a; P_l = [P_u, P_v]."""
    if l in _cache:
        return _cache[l]
    if len(l) == 1:
        p = {l: 1.0}
    else:
        u, v = standard_factorisation(l)
        pu, pv = bracket_polynomial(u), bracket_polynomial(v)
        p = _mul(pu, pv)
        for w, x in _mul(pv, pu).items():
            p[w] -= x
        p = {w: x for w, x in p.items() if x != 0.0}
    _cache[l] = p
    return p


def lie_hankel_matrix(coef, m, K, rows_max=None, cols_max=None):
    """Rows: P_l for Lyndon l with |l| <= rows_max (default K).  Columns: all words v with |v| <= cols_max
    (default K).  Entry c(P_l v), set to 0 when |l| + |v| > K (the staircase convention)."""
    rows_max = K if rows_max is None else rows_max
    cols_max = K if cols_max is None else cols_max
    rows = lyndon_words(m, rows_max)
    cols = [v for L in range(cols_max + 1) for v in itertools.product(range(m + 1), repeat=L)]
    H = np.zeros((len(rows), len(cols)))
    for i, l in enumerate(rows):
        P = bracket_polynomial(l)
        for j, v in enumerate(cols):
            if len(l) + len(v) <= K:
                H[i, j] = sum(x * coef.get(w + v, 0.0) for w, x in P.items())
    return H, rows, cols


def lyndon_factorisation(w):
    """Duval's algorithm: w = l_1 l_2 ... l_k, l_1 >= l_2 >= ... >= l_k Lyndon."""
    out, i, n = [], 0, len(w)
    while i < n:
        j, k = i + 1, i
        while j < n and w[k] <= w[j]:
            k = i if w[k] < w[j] else k + 1
            j += 1
        while i <= k:
            out.append(w[i:i + j - k])
            i += j - k
    return out


def radford_features(E, words):
    """Given iterated integrals E (rows indexed by `words`, which must contain every Lyndon factor needed),
    return R with R[w] = prod I_{l_i}^{n_i} / n_i! over the Lyndon factorisation of w.  The empty word gives 1."""
    idx = {w: i for i, w in enumerate(words)}
    R = np.empty_like(E)
    for i, w in enumerate(words):
        if len(w) == 0:
            R[i] = 1.0
            continue
        f = lyndon_factorisation(w)
        r = np.ones(E.shape[1:])
        for l in set(f):
            n = f.count(l)
            r = r * E[idx[l]] ** n / factorial(n)
        R[i] = r
    return R


def shuffle(u, v):
    """Shuffle product of two words, as {word: multiplicity}."""
    if not u:
        return {v: 1}
    if not v:
        return {u: 1}
    out = defaultdict(int)
    for w, n in shuffle(u[:-1], v).items():
        out[w + u[-1:]] += n
    for w, n in shuffle(u, v[:-1]).items():
        out[w + v[-1:]] += n
    return dict(out)


def radford_matrix(words):
    """T with R_w = sum_v T[w, v] I_v (exact shuffle expansion).  T is block diagonal by length and
    triangular in the order of the Lyndon factorisation; R = T @ E for iterated integrals E of a path."""
    idx = {w: i for i, w in enumerate(words)}
    T = np.zeros((len(words), len(words)))
    for i, w in enumerate(words):
        if len(w) == 0:
            T[i, idx[()]] = 1.0
            continue
        f = lyndon_factorisation(w)
        poly = {(): 1.0}
        for l in f:
            new = defaultdict(float)
            for a, x in poly.items():
                for b, n in shuffle(a, l).items():
                    new[b] += x * n
            poly = new
        d = 1
        for l in set(f):
            d *= factorial(f.count(l))
        for v, x in poly.items():
            T[i, idx[v]] += x / d
    return T
