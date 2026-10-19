"""
volterra_kernel.py
=====================
Extracts explicit Volterra kernels from a truncated Chen-Fliess coefficient
set, for the single-input, drift-bearing convention (alphabet {0,1}; 0 is
drift, dxi_0=dt; 1 is the input, dxi_1=u(t)dt).

A word with exactly n occurrences of letter 1, split by its drift-letter
run-lengths (j_0, j_1, ..., j_n) -- j_0 drift letters before the first
input letter, j_i between the i-th and (i+1)-th, j_n after the last --
contributes to the exact, closed-form n-th order kernel

    K_w(t; s_1,...,s_n) = s_1^{j_0}/j_0! * (s_2-s_1)^{j_1}/j_1! * ...
                           * (t-s_n)^{j_n}/j_n!,           0 <= s_1<=...<=s_n<=t

so that I_w(t) = integral_{0<s_1<...<s_n<t} K_w(t;s) u(s_1)...u(s_n) ds,
and the full n-th order Volterra term is the sum of c_w K_w over every
word of degree n. This follows directly from the same recursive
definition used in signature.py (letters appended in time order, drift
integrates dt, input integrates u(t)dt); checked against
I_(0,1)(t)=int_0^t s u(s) ds and I_(1,0)(t)=int_0^t (t-s) u(s) ds.


@author: gary green
last edit 19-10-2026

"""
import itertools
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import factorial

Word = Tuple[int, ...]


def words_of_degree(n: int, max_length: int) -> List[Tuple[int, ...]]:
    """All (j_0,...,j_n) with j_i >= 0 and n + sum(j_i) <= max_length --
    equivalently, every drift-letter run-length pattern for a degree-n
    word within the truncation order."""
    budget = max_length - n
    if budget < 0:
        return []
    out = []
    for total in range(budget + 1):
        for js in _compositions(total, n + 1):
            out.append(js)
    return out


def _compositions(total: int, parts: int):
    """All ways to write `total` as an ordered sum of `parts` non-negative
    integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def word_from_runs(js: Tuple[int, ...]) -> Word:
    """(j_0,...,j_n) -> the word (0^j0, 1, 0^j1, 1, ..., 1, 0^jn)."""
    w = list((0,) * js[0])
    for j in js[1:]:
        w.append(1)
        w.extend([0] * j)
    return tuple(w)


def kernel_value(js: Tuple[int, ...], t: float, s: np.ndarray) -> np.ndarray:
    """K_w(t; s_1,...,s_n) for one run-length pattern, vectorised over
    an array of (s_1,...,s_n) points: s has shape (..., n)."""
    n = len(js) - 1
    if n == 0:
        return np.full(s.shape[:-1] if s.ndim > 0 else (), t**js[0] / factorial(js[0]))
    edges = np.concatenate([np.zeros(s.shape[:-1] + (1,)), s,
                             np.full(s.shape[:-1] + (1,), t)], axis=-1)
    gaps = np.diff(edges, axis=-1)
    return np.prod(gaps**np.array(js) / factorial(js), axis=-1)


def extract_kernel_terms(coef: Dict[Word, float], n: int, max_length: int):
    """Every (run-length pattern, coefficient) pair contributing to the
    n-th order kernel, skipping words with zero fitted/exact coefficient."""
    terms = []
    for js in words_of_degree(n, max_length):
        w = word_from_runs(js)
        c = coef.get(w, 0.0)
        if c != 0.0:
            terms.append((js, c))
    return terms


def evaluate_kernel(terms, t: float, s: np.ndarray) -> np.ndarray:
    """Sum_w c_w K_w(t;s) over the given terms, at one or many (s_1,...,s_n)
    points (s of shape (..., n))."""
    total = np.zeros(s.shape[:-1])
    for js, c in terms:
        total = total + c * kernel_value(js, t, s)
    return total


def simplex_quadrature(n, t, n_nodes=40):
    """Gauss-Legendre rule on the simplex 0 < s_1 < ... < s_n < t, by the collapsed (Duffy) map
    s_n = t x_n, s_{k} = s_{k+1} x_k, with Jacobian t^n prod_k x_k^(k-1) (k = 1..n, x in [0,1]^n).
    Returns nodes (P, n) ordered s_1 < ... < s_n and weights (P,)."""
    g, wg = np.polynomial.legendre.leggauss(n_nodes)
    x1 = 0.5 * (g + 1.0); w1 = 0.5 * wg
    X = np.array(np.meshgrid(*([x1] * n), indexing="ij")).reshape(n, -1).T       # (P, n): x_1..x_n
    W = np.prod(np.array(np.meshgrid(*([w1] * n), indexing="ij")).reshape(n, -1).T, axis=1)
    S = np.empty_like(X)
    S[:, n - 1] = t * X[:, n - 1]
    for k in range(n - 2, -1, -1):
        S[:, k] = S[:, k + 1] * X[:, k]
    jac = t ** n * np.prod([X[:, k] ** k for k in range(n)], axis=0)
    return S, W * jac


def kernel_contribution(coef: Dict[Word, float], n: int, max_length: int, t: float, u_func, n_nodes=40) -> float:
    """The degree-n part of y(t) - y(0), computed from the kernel: integral over the simplex of
    sum_w c_w K_w(t; s) u(s_1)...u(s_n), by Gauss-Legendre quadrature on the simplex."""
    terms = extract_kernel_terms(coef, n, max_length)
    if not terms:
        return 0.0
    if n == 0:
        return float(sum(c * kernel_value(js, t, np.zeros((0,))) for js, c in terms))
    S, W = simplex_quadrature(n, t, n_nodes)
    k = evaluate_kernel(terms, t, S)
    uu = np.prod(np.array([np.asarray(u_func(S[:, j])).reshape(-1) for j in range(n)]), axis=0)
    return float(np.sum(W * k * uu))


def symmetric_kernel(terms, t: float, s: np.ndarray) -> np.ndarray:
    """The symmetric n-th order Volterra kernel at any points s (shape (..., n)) of [0, t]^n:
    h_n(t; s) = (1/n!) sum_w c_w K_w(t; sorted s).  Then the order-n part of y(t) - y(0) is the
    integral of h_n(t; s) u(s_1)...u(s_n) over the whole cube [0, t]^n."""
    s = np.sort(np.asarray(s, float), axis=-1)
    n = s.shape[-1]
    return evaluate_kernel(terms, t, s) / factorial(n)


def cube_contribution(coef: Dict[Word, float], n: int, max_length: int, t: float, u_func, n_nodes=40) -> float:
    """The order-n part of y(t) - y(0) from the symmetric kernel, integrated over the whole cube [0, t]^n.
    The cube is split into its n! ordered pieces (the kernel has a corner on each diagonal); each piece is
    integrated by the simplex rule with its coordinates permuted, and the symmetric kernel is evaluated at
    the unsorted points."""
    terms = extract_kernel_terms(coef, n, max_length)
    if not terms or n == 0:
        return kernel_contribution(coef, n, max_length, t, u_func, n_nodes)
    S, W = simplex_quadrature(n, t, n_nodes)
    total = 0.0
    for p in itertools.permutations(range(n)):
        Sp = S[:, list(p)]
        k = symmetric_kernel(terms, t, Sp)
        uu = np.prod(np.array([np.asarray(u_func(Sp[:, j])).reshape(-1) for j in range(n)]), axis=0)
        total += float(np.sum(W * k * uu))
    return total
