"""
word_tree.py
==============
A bilinear realization that is exactly nilpotent of index k+1: any
product of k+1 or more generator matrices annihilates every state, by
construction, so its generating series is exactly zero for all words
longer than k. Forward-simulating this system is mathematically
identical to evaluating the truncated signature series directly (up to
ODE-integration numerical error).

State space: every word of length 0..k (the full, unreduced word count,
not minimal-rank, but guaranteed correct). Each letter's action extends
the current word by appending that letter at the end (matching
signature.py's own recursive convention: a word's value is built by
integrating the newest letter against the existing prefix's value); if
the extended word would exceed length k, the action is zero.


@author: gary green
last edit 19-10-2026
"""
import numpy as np
import itertools


def build_word_tree_system(coef_dict, m, order):
    """
    coef_dict: {word_tuple: value}. m: number of non-drift letters (so
    alphabet size = m+1, letters 0..m). order: truncation k.
    Returns M0, M1, ..., Mm (each (dim,dim)), mu (dim,), x0 (dim,), words.
    """
    alphabet = list(range(m + 1))
    words = [w for L in range(order + 1) for w in itertools.product(alphabet, repeat=L)]
    idx = {w: i for i, w in enumerate(words)}
    dim = len(words)

    Ms = [np.zeros((dim, dim)) for _ in alphabet]
    for w in words:
        i = idx[w]
        for a in alphabet:
            w_ext = w + (a,)
            if len(w_ext) <= order:
                j = idx[w_ext]
                Ms[a][j, i] = 1.0

    mu = np.array([coef_dict.get(w, 0.0) for w in words])
    x0 = np.zeros(dim)
    x0[idx[()]] = 1.0
    return Ms, mu, x0, words


def verify_nilpotent(Ms, order, tol=1e-10):
    """Check every length-(order+1) product of generators is exactly zero."""
    dim = Ms[0].shape[0]
    m_let = len(Ms)
    max_norm = 0.0
    for combo in itertools.product(range(m_let), repeat=order + 1):
        P = np.eye(dim)
        for a in combo:
            P = Ms[a] @ P
        max_norm = max(max_norm, np.linalg.norm(P))
    return max_norm, max_norm < tol
