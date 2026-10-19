"""
signature.py
============
Iterated integrals for the drift-bearing Chen-Fliess convention: alphabet {0, 1, ..., m}, letter 0 is the drift
(dxi_0 = dt), letters 1..m are inputs (dxi_i = u_i(t) dt).

A word w = (a_1, ..., a_L) is built by appending letters in time order, so a_1 acts first:

    I_() = 1,   I_{w+(a,)}(t) = int_0^t I_w(s) dxi_a(s).

Two methods, chosen by what is known about the input (project rule):

  * input known as a formula (a callable): the exact word-tree ODE, d/dt I_{w+(a,)} = u_a(t) I_w, integrated by the
    eighth-order Dormand-Prince method with a tolerance scaled for each word (integrate.iterated_integrals);
  * input known only as samples (data, and every regression fit): iisignature on the sampled path
    X(t_k) = (t_k - t_0, U_1(t_k), ..., U_m(t_k)), U_i = int u_i, where U_i at the samples is the exact integral of the
    cubic-spline interpolant of the samples of u_i.  iisignature gives the exact signature of the piecewise-linear
    path through these points; its difference from the smooth path is of second order in the sampling step and is
    reported by the cross-check in examples/run_all.py.


@author: gary green
last edit 19-10-2026
"""
import numpy as np
from scipy.interpolate import CubicSpline

from .integrate import iterated_integrals, word_list


def as_function(t, u):
    """u: callable s -> (m,) array, or samples of shape (N,) or (N, m) at the times t."""
    if callable(u):
        return u
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = u[:, None]
    spl = CubicSpline(t, u, axis=0)
    return lambda s: np.atleast_1d(spl(s))


def sampled_iterated_integrals(t, u_samples, K, drift=True):
    """Iterated integrals at every sample time from sampled inputs, by iisignature.
    t: (N,), u_samples: (N,) or (N, m).  Returns (E, words) in the same layout as integrate.iterated_integrals."""
    import iisignature
    t = np.asarray(t, float)
    u = np.asarray(u_samples, float)
    if u.ndim == 1:
        u = u[:, None]
    m = u.shape[1]
    U = CubicSpline(t, u, axis=0).antiderivative()(t)
    U = U - U[0]
    path = np.column_stack([t - t[0], U]) if drift else U        # driftless: the path is the channels alone
    S = iisignature.sig(path, K, 2)                       # signatures of every prefix, (N-1, n_words)
    E = np.vstack([np.zeros((1, S.shape[1])), S]).T
    words = word_list(m if drift else m - 1, K)          # same order as iisignature: by length, then dictionary
    return E, words


def signature_matrix(t, u, order, m=None, drift=True):
    """Returns (Emat, words): Emat[i, n] = I_{words[i]}(t[n]) (from t[0]); words lists every word over {0..m} of
    length 1..order (the empty word, I = 1, is not included).  A callable input uses the exact ODE; samples use
    iisignature (see the module docstring)."""
    t = np.asarray(t, float)
    if callable(u):
        if m is None:
            m = len(np.atleast_1d(u(t[0])))
        return iterated_integrals(t, u, m if drift else m - 1, order, drift=drift)
    return sampled_iterated_integrals(t, u, order, drift=drift)
