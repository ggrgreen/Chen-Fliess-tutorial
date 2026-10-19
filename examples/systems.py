"""
systems.py -- the driftless test systems, their exact Chen-Fliess (signature) coefficients and their true responses.

Coefficients: c_w = (L_{f_{a1}} ... L_{f_{aL}} h)(x0), first letter earliest in time, computed symbolically with the
parameters and the starting state as exact fractions.  Letters 0..m are the input channels (no letter for time).

  driftless Duffing (2 letters):  q1' = u1,  q2' = u2 + beta q1^3 u1 - delta q2 u1,  y = q2
  Heisenberg / Brockett integrator (2 letters):  q1' = u1,  q2' = u2,  q3' = q1 u2 - q2 u1,  y = q3
  four-letter test system:  q1' = u1 + sin(q3) u2, q2' = u2 + cos(q4) u3, q3' = u3 + sin(q1) u4,
                            q4' = u4 + cos(q2) u1,  y = q1 q2 + sin(q3) + q4^2


@author: gary green
last edit 19-10-2026
"""
import itertools
import sympy as sp
from scipy.integrate import solve_ivp

R_ = lambda x: sp.Rational(str(x))


def series_coeffs(fields, h, syms, x0, order):
    """fields: list of sympy column vectors (one per letter); h: output expression; x0: starting state."""
    subs0 = {s: R_(v) for s, v in zip(syms, x0)}
    cache = {(): h}

    def L(g, f):
        return sum(sp.diff(g, s) * f[i] for i, s in enumerate(syms))

    def get(w):
        if w not in cache:
            cache[w] = sp.expand(L(get(w[1:]), fields[w[0]]))
        return cache[w]

    coef = {}
    for Ln in range(order + 1):
        for w in itertools.product(range(len(fields)), repeat=Ln):
            coef[w] = float(get(w).subs(subs0))
    return coef


# ---------------------------------------------------------------- driftless Duffing
DD_DELTA, DD_BETA, DD_X0 = 0.3, 1.0, (0.5, 0.3)
_q1, _q2 = sp.symbols("q1 q2")


def dduffing_coeffs(order):
    f = [sp.Matrix([1, R_(DD_BETA) * _q1 ** 3 - R_(DD_DELTA) * _q2]), sp.Matrix([0, 1])]
    c = series_coeffs(f, _q2, (_q1, _q2), DD_X0, order)
    c[()] = 0.0
    return c


def dduffing_true(t, u):
    """u(s) -> (u1, u2).  Returns y(t) - y(0)."""
    def rhs(s, q):
        u1, u2 = u(s)
        return [u1, u2 + DD_BETA * q[0] ** 3 * u1 - DD_DELTA * q[1] * u1]
    sol = solve_ivp(rhs, (t[0], t[-1]), list(DD_X0), t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[1] - DD_X0[1]


# ---------------------------------------------------------------- Heisenberg
HB_X0 = (0.3, 0.5, 0.0)
_h1, _h2, _h3 = sp.symbols("h1 h2 h3")


def heisenberg_coeffs(order):
    f = [sp.Matrix([1, 0, -_h2]), sp.Matrix([0, 1, _h1])]
    c = series_coeffs(f, _h3, (_h1, _h2, _h3), HB_X0, order)
    c[()] = 0.0
    return c


def heisenberg_true(t, u):
    def rhs(s, q):
        u1, u2 = u(s)
        return [u1, u2, q[0] * u2 - q[1] * u1]
    sol = solve_ivp(rhs, (t[0], t[-1]), list(HB_X0), t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[2] - HB_X0[2]


# ---------------------------------------------------------------- four letters
F4_X0 = (0.3, 0.5, -0.2, 0.4)
_p = sp.symbols("p1 p2 p3 p4")


def four_coeffs(order):
    p1, p2, p3, p4 = _p
    f = [sp.Matrix([1, 0, 0, sp.cos(p2)]), sp.Matrix([sp.sin(p3), 1, 0, 0]),
         sp.Matrix([0, sp.cos(p4), 1, 0]), sp.Matrix([0, 0, sp.sin(p1), 1])]
    h = p1 * p2 + sp.sin(p3) + p4 ** 2
    c = series_coeffs(f, h, _p, F4_X0, order)
    c[()] = 0.0
    return c
