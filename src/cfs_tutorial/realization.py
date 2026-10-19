"""
graded_nilpotent.py
====================
Minimal, EXACTLY nilpotent bilinear realization of a truncated (order-K)
Chen-Fliess series, built by degree-respecting recursive Hankel reduction.

Word convention (matches signature.py / word_tree.py):
  - alphabet = {0, 1, ..., m}  (size s = m+1; letter 0 = drift, 1..m = inputs)
  - a word w = (a1, ..., aL) is built by APPENDING letters at the end
  - c(w) is the coefficient such that y(t) = sum_w c_w * I_w(t), with
      I_() = 1
      I_{w+(a,)}(t) = int_0^t I_w(s) dxi_a(s)          (xi_0(t) = t)
  - a bilinear realization (A_0..A_m, lam, mu) represents this via
      x_() = lam,   x_{w+(a,)} = A_a x_w,   c_w = mu . x_w
    i.e. reading word w = (a1,...,aL) left-to-right in TIME order, the
    state evolves x <- A_{a1} x, then A_{a2}, ..., last A_{aL}:
      x_w = A_{aL} ... A_{a1} lam,   c_w = mu^T A_{aL} ... A_{a1} lam

Algorithm ("graded / degree-respecting recursive Hankel reduction"):
  Process word length ell = K, K-1, ..., 0. At each ell, stack the "local
  future vector" of every raw length-ell word:
      F_w = [ c(w),  coord(w+(0,)),  coord(w+(1,)), ..., coord(w+(m,)) ]
  where coord(u) is u's coordinate vector in the ALREADY-REDUCED basis at
  degree ell+1 (built in the previous, deeper step; empty/absent for ell=K).
  SVD-factorise the stack of F_w's: this gives (a) a reduced basis of
  dimension r_ell for degree ell, and (b) for every raw length-ell word, its
  coordinate vector in that reduced basis (needed by degree ell-1).

  Because A_a is only ever assembled between a degree-ell block and the
  degree-(ell+1) block (columns of the degree-ell reduced-basis matrix), the
  full system, stacked over ell=0..K, is STRICTLY BLOCK SUPER-TRIANGULAR by
  degree. That makes it exactly nilpotent of index K+1 by construction --
  an algebraic identity, not a numerical coincidence -- independent of
  whatever SVD tolerance was used to pick each r_ell.
  

@author: gary green
last edit 19-10-2026
"""
import itertools
import numpy as np
from typing import Dict, Tuple

Word = Tuple[int, ...]


def graded_nilpotent_realize(coef: Dict[Word, float], m: int, order: int,
                              tol: float = 1e-10) -> Dict:
    """
    m     : number of non-drift letters (alphabet = {0,...,m}, size s=m+1)
    order : truncation K. coef is assumed (or will be treated as) zero for
            |w| > order.
    tol   : relative singular-value cutoff used at EVERY degree block.

    Returns dict with:
      dims        : list [r_0, r_1, ..., r_K]  (block sizes by degree)
      dim_total   : sum(dims)  -- total realized state dimension
      A           : list of (dim_total, dim_total) matrices, one per letter
      lam, mu     : (dim_total,) initial-state and readout vectors
      block_slices: list of python slice objects, one per degree, giving
                    each degree's index range within the full state vector
      words_by_degree, coord_by_degree : bookkeeping (for diagnostics)
    """
    s = m + 1
    alphabet = list(range(s))
    K = order

    def c(w: Word) -> float:
        return float(coef.get(w, 0.0))

    # bottom-up: degree K down to 0
    steps = {}             # ell -> (stacked local future vectors Q_ell, singular values) for inspection
    reduced_basis = {}     # ell -> (r_ell, feat_dim_ell) matrix
    coord_of = {}           # ell -> dict word -> coordinate vector (r_ell,)
    r = {K + 1: 0}           # degree K+1 doesn't exist; "0-dimensional"

    for ell in range(K, -1, -1):
        words_ell = list(itertools.product(alphabet, repeat=ell))
        r_next = r[ell + 1]
        feat_dim = 1 + s * r_next

        Fmat = np.zeros((len(words_ell), feat_dim))
        for i, w in enumerate(words_ell):
            Fmat[i, 0] = c(w)
            if r_next > 0:
                for a in alphabet:
                    w_ext = w + (a,)
                    coord = coord_of[ell + 1].get(w_ext)
                    if coord is not None:
                        Fmat[i, 1 + a * r_next: 1 + (a + 1) * r_next] = coord

        if Fmat.size == 0:
            r_ell = 0
            RB = np.zeros((0, feat_dim))
            C = np.zeros((len(words_ell), 0))
        else:
            U, S, Vt = np.linalg.svd(Fmat, full_matrices=False)
            if S.size == 0 or S[0] <= 0:
                r_ell = 0
            else:
                r_ell = int(np.sum(S > tol * S[0]))
            r_ell = max(r_ell, 0)
            RB = (np.diag(np.sqrt(S[:r_ell])) @ Vt[:r_ell, :]) if r_ell > 0 \
                else np.zeros((0, feat_dim))
            C = (U[:, :r_ell] * np.sqrt(S[:r_ell])[np.newaxis, :]) if r_ell > 0 \
                else np.zeros((len(words_ell), 0))

        reduced_basis[ell] = RB
        steps[ell] = dict(Q=Fmat.copy(), sv=(S.copy() if Fmat.size else np.zeros(0)), words=words_ell)
        r[ell] = r_ell
        coord_of[ell] = {w: C[i, :] for i, w in enumerate(words_ell)}

    dims = [r[ell] for ell in range(0, K + 1)]
    dim_total = sum(dims)

    # index offsets for each degree block within the full stacked state
    offsets = {}
    off = 0
    slices = []
    for ell in range(0, K + 1):
        offsets[ell] = off
        slices.append(slice(off, off + dims[ell]))
        off += dims[ell]

    A = [np.zeros((dim_total, dim_total)) for _ in alphabet]
    mu = np.zeros(dim_total)
    for ell in range(0, K + 1):
        RB = reduced_basis[ell]
        r_ell = dims[ell]
        r_next = r[ell + 1] if ell < K else 0
        if r_ell == 0:
            continue
        mu[offsets[ell]: offsets[ell] + r_ell] = RB[:, 0]
        if r_next > 0:
            for a in alphabet:
                block = RB[:, 1 + a * r_next: 1 + (a + 1) * r_next]  # (r_ell, r_next)
                # A_a acts: x_new = A_a @ x_old ; we want a state in degree-ell
                # basis (coefficient vector v in R^{r_ell}) to map to a state
                # in degree-(ell+1) basis (coefficients block^T @ v).
                A[a][offsets[ell + 1]: offsets[ell + 1] + r_next,
                     offsets[ell]: offsets[ell] + r_ell] = block.T

    # initial state: degree-0 block, coordinate of the single raw empty word
    lam = np.zeros(dim_total)
    if dims[0] > 0:
        lam[offsets[0]: offsets[0] + dims[0]] = coord_of[0][()]

    return dict(dims=dims, dim_total=dim_total, A=A, lam=lam, mu=mu,
                block_slices=slices, offsets=offsets, steps=steps, reduced_basis=reduced_basis)


def verify_exact(realiz: Dict, coef: Dict[Word, float], m: int, order: int) -> float:
    """Max |predicted - target| over every word of length 0..order."""
    A, lam, mu = realiz['A'], realiz['lam'], realiz['mu']
    max_err = 0.0
    alphabet = list(range(m + 1))
    for L_ in range(0, order + 1):
        for w in itertools.product(alphabet, repeat=L_):
            x = lam.copy()
            for a in w:
                x = A[a] @ x
            pred = float(mu @ x)
            target = float(coef.get(w, 0.0))
            max_err = max(max_err, abs(pred - target))
    return max_err


def verify_nilpotent(realiz: Dict, order: int, m: int, n_probe: int = None) -> float:
    """
    Max operator norm over all length-(order+1) products of generators.
    Since dim can be large, if n_probe is given, only that many random
    products are checked (with a fixed seed) instead of the full
    (m+1)^(order+1) exhaustive set.
    """
    A = realiz['A']
    dim = realiz['dim_total']
    alphabet = list(range(m + 1))
    combos = itertools.product(alphabet, repeat=order + 1)
    max_norm = 0.0
    if n_probe is not None:
        rng = np.random.default_rng(0)
        all_combos = list(combos)
        idx = rng.choice(len(all_combos), min(n_probe, len(all_combos)), replace=False)
        combos = [all_combos[i] for i in idx]
    for combo in combos:
        P = np.eye(dim)
        for a in combo:
            P = A[a] @ P
        max_norm = max(max_norm, np.linalg.norm(P))
    return max_norm


def simulate_bilinear(realiz: Dict, t: np.ndarray, u) -> np.ndarray:
    """
    Forward-simulate dx/dt = A_0 x + sum_{i>=1} u_i(t) A_i x, y = mu . x, x(0) = lam, by the eighth-order
    Dormand-Prince method (DOP853, relative tolerance 1e-12).  u is a function of time, or samples at t
    (interpolated by a cubic spline).
    """
    from .integrate import simulate_realisation
    from .signature import as_function
    return simulate_realisation(realiz['A'], realiz['lam'], realiz['mu'], np.asarray(t, float), as_function(t, u))
