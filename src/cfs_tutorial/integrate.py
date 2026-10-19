"""
integrate.py
============
High-order integration for the drift-bearing Chen-Fliess convention, 

Alphabet {0, 1, ..., m}: letter 0 is the drift (dxi_0 = dt), letters 1..m are inputs (dxi_i = u_i(t) dt).
A word w = (a_1, ..., a_L) is built by appending letters in time order:

    I_() = 1,   I_{w+(a,)}(t) = int_0^t I_w(s) u_a(s) ds,   u_0 = 1.

iterated_integrals: every I_w with |w| <= K is one component of a linear ODE (the word tree),
    d/dt I_{w+(a,)} = u_a(t) I_w,
integrated with the 8th-order Dormand-Prince method (DOP853) at tight tolerances, with the input evaluated as the
analytic function it is.  The result is accurate to the integrator tolerance, not to the sampling step.

simulate_realisation: the bilinear system dx/dt = (A_0 + sum_i u_i(t) A_i) x, x(0) = lam, y = mu . x, integrated in
the same way.

@author: gary green
last edit 19-10-2026
"""
import itertools
import numpy as np
from scipy.integrate import solve_ivp

RTOL, ATOL = 1e-13, 1e-15


def word_list(m, K, with_empty=False):
    words = [w for L in range(1, K + 1) for w in itertools.product(range(m + 1), repeat=L)]
    return ([()] + words) if with_empty else words


def iterated_integrals(t, u_func, m, K, rtol=RTOL, atol=ATOL, drift=True):
    """t: (N,) sample times starting at t[0]; u_func(s) -> array of m input values at time s.
    Returns (E, words): E[i, n] = I_{words[i]}(t[n]) measured from t[0]; words exclude the empty word.

    The integral of a word of length L with n input letters is at most (T^L / L!) U^n, where T is the window and U the
    largest input size, so long words are tiny.
    drift=True: letter 0 is time and letters 1..m are the m inputs.  drift=False (path signature): the m+1 letters
    0..m are the m+1 channels returned by u_func, with no letter for time.  Each component is integrated in units of that bound, so that the
    absolute tolerance of the integrator is a relative tolerance for every word, however long."""
    from math import factorial
    words = word_list(m, K, with_empty=True)
    idx = {w: i for i, w in enumerate(words)}
    parent = np.array([idx[w[:-1]] if w else -1 for w in words])
    last = np.array([w[-1] if w else -1 for w in words])
    nz = np.arange(1, len(words))
    T = float(t[-1] - t[0])
    grid = np.linspace(t[0], t[-1], 2001)
    U = max(float(np.max(np.abs(np.array([np.atleast_1d(u_func(g)) for g in grid])))), 1e-300)
    scale = np.array([T ** len(w) / factorial(len(w)) * U ** sum(1 for a in w if (a != 0 or not drift)) for w in words])
    ratio = np.ones(len(words)); ratio[nz] = scale[parent[nz]] / scale[nz]

    def rhs(s, y):
        uu = np.concatenate([[1.0], np.atleast_1d(u_func(s))]) if drift else np.atleast_1d(u_func(s))
        dy = np.zeros_like(y)
        dy[nz] = uu[last[nz]] * ratio[nz] * y[parent[nz]]
        return dy

    y0 = np.zeros(len(words)); y0[0] = 1.0
    sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="DOP853", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return sol.y[1:] * scale[1:, None], words[1:]


def simulate_realisation(A, lam, mu, t, u_func, rtol=1e-12, atol=1e-14, drift=True):
    """A: list of (n, n) generators [A_0, A_1, ..., A_m]; returns y(t) = mu . x(t)."""
    A = [np.asarray(a, float) for a in A]

    def rhs(s, x):
        uu = np.atleast_1d(u_func(s))
        if drift:
            G = A[0].copy()
            for i, ui in enumerate(uu):
                G += ui * A[i + 1]
        else:
            G = sum(ui * A[i] for i, ui in enumerate(uu))
        return G @ x

    sol = solve_ivp(rhs, (t[0], t[-1]), np.asarray(lam, float), t_eval=t, method="DOP853", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return np.asarray(mu, float) @ sol.y


def simulate_realisation_sampled(A, lam, mu, t, u_samples, drift=True):
    """The bilinear system driven by the same sampled path that iisignature uses: on each sample interval the input
    is held at its mean, (U(t_{k+1}) - U(t_k)) / (t_{k+1} - t_k), with U the exact integral of the cubic spline through
    the samples.  The state is advanced exactly by the matrix exponential on each interval."""
    from scipy.interpolate import CubicSpline
    from scipy.linalg import expm
    A = [np.asarray(a, float) for a in A]
    u = np.asarray(u_samples, float)
    if u.ndim == 1:
        u = u[:, None]
    U = CubicSpline(t, u, axis=0).antiderivative()(t)
    x = np.asarray(lam, float).copy(); out = [float(np.asarray(mu, float) @ x)]
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]; ub = (U[k + 1] - U[k]) / dt
        if drift:
            G = A[0].copy()
            for i, ui in enumerate(ub):
                G += ui * A[i + 1]
        else:
            G = sum(ui * A[i] for i, ui in enumerate(ub))
        x = expm(G * dt) @ x
        out.append(float(np.asarray(mu, float) @ x))
    return np.array(out)
