"""
duffing_symbolic.py
====================
Exact (symbolic) Chen-Fliess generating-series coefficients for the driven
Duffing oscillator:

    dq1/dt = q2
    dq2/dt = -delta*q2 - alpha*q1 - beta*q1^3 + u(t)
    y(t)   = q1(t)

Vector fields (alphabet: 0=drift, 1=input):
    f0 = (q2, -delta*q2 - alpha*q1 - beta*q1^3)
    f1 = (0, 1)
    h  = q1

Convention (matches signature.py / realization.py):
  word w = (a1,...,aL) built by appending letters in TIME order (a1 first).
  The matching coefficient is

      c_w = ( L_{f_{a1}} L_{f_{a2}} ... L_{f_{aL}} h )(x0)

  i.e. apply Lie derivatives in REVERSE word order (rightmost/last letter
  applied to h FIRST, leftmost/first letter applied LAST), then evaluate
  at the initial state x0.


@author: gary green
last edit 19-10-2026
"""
import itertools
import sympy as sp

q1, q2 = sp.symbols('q1 q2')


def duffing_fields(delta, alpha, beta):
    f0 = sp.Matrix([q2, -delta * q2 - alpha * q1 - beta * q1**3])
    f1 = sp.Matrix([0, 1])
    return [f0, f1]


def lie_derivative(g, f):
    """L_f g = grad(g) . f,  g scalar, f vector field over (q1,q2)."""
    return sp.diff(g, q1) * f[0] + sp.diff(g, q2) * f[1]


def duffing_coeffs(order, delta=0.3, alpha=-1.0, beta=1.0,
                    q1_0=0.5, q2_0=0.3):
    """
    Returns dict {word: float} for every word over alphabet {0,1},
    length 0..order, of the EXACT (untruncated system, truncated series)
    Duffing generating-series coefficients.
    """
    R_ = lambda x: sp.Rational(str(x))          # parameters as exact rationals, so the coefficients are exact
    fields = duffing_fields(R_(delta), R_(alpha), R_(beta))
    h = q1
    subs0 = {q1: R_(q1_0), q2: R_(q2_0)}

    # cache of symbolic functions g_w(q1,q2) := L_{f_{aL}} ... L_{f_{a1}} h
    # built by prepending new letters (i.e. applying L_{f_{a}} at the FRONT
    # each time a letter is appended to the word), consistent with
    # c_w = (L_{a1} L_{a2} ... L_{aL} h)(x0):
    #   g_() = h
    #   g_{w+(a,)} = L_{f_a}( g_w ) applied FIRST at the innermost position,
    #   i.e. we build g by prepending the NEW letter's Lie derivative
    #   in front of the operator chain: g_{w+(a,)} = L_{f_{a1}}...L_{f_{aL}} L_{f_a} h
    #   -- easiest to just recompute directly via recursion on the *operator
    #   ordering*, so build bottom-up over word length using the recursive
    #   relation:  g_{(a1,...,aL)} = L_{f_{a1}} ( g_{(a2,...,aL)} )
    cache = {(): h}

    def get_g(w):
        if w in cache:
            return cache[w]
        a1 = w[0]
        rest = w[1:]
        g_rest = get_g(rest)
        g = sp.expand(lie_derivative(g_rest, fields[a1]))
        cache[w] = g
        return g

    coef = {}
    for L_ in range(0, order + 1):
        for w in itertools.product([0, 1], repeat=L_):
            g = get_g(w)
            coef[w] = float(g.subs(subs0))
    return coef
