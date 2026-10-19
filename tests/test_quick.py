"""Quick checks: realisations reproduce their series, with and without a letter for time.


@author: gary green
last edit 19-10-2026
"""
import itertools
import numpy as np
import cfs_tutorial as G


def _random_series(m, K, seed=0):
    rng = np.random.default_rng(seed)
    c = {w: float(rng.normal()) for L in range(1, K + 1) for w in itertools.product(range(m + 1), repeat=L)}
    c[()] = 0.0
    return c


def test_staircase_and_graded_reproduce_series():
    c = _random_series(1, 4)
    for R in (G.staircase_realize(c, 1, 4, 1e-10),):
        assert G.max_coef_error(R, c, 1, 4) < 1e-10
    g = G.graded_nilpotent_realize(c, m=1, order=4, tol=1e-12)
    gr = dict(A=g["A"], lam=g["lam"], mu=g["mu"], dim_total=g["dim_total"])
    assert G.max_coef_error(gr, c, 1, 4) < 1e-10 and G.nilpotency_residual(gr, 1, 4) == 0.0


def test_sampled_realisation_matches_series():
    t = np.arange(0, 0.15 + 5e-4, 1e-3)
    for drift, U in ((True, 0.3 * np.sin(6 * np.pi * t)), (False, np.column_stack([np.sin(6 * np.pi * t), np.cos(4 * np.pi * t)]))):
        E, W = G.sampled_iterated_integrals(t, U, 4, drift=drift)
        c = _random_series(1, 4, seed=1)
        R = G.staircase_realize(c, 1, 4, 1e-10)
        y = G.simulate_realisation_sampled(R["A"], R["lam"], R["mu"], t, U, drift=drift)
        assert np.max(np.abs(y - np.array([c[w] for w in W]) @ E)) < 1e-10


def test_exact_and_sampled_integrals_agree():
    u = lambda s: np.array([0.3 * np.sin(6 * np.pi * s)])
    t = np.arange(0, 0.15 + 5e-5, 1e-4)
    E1, W1 = G.iterated_integrals(t, u, 1, 3)
    E2, W2 = G.sampled_iterated_integrals(t, np.array([u(s)[0] for s in t]), 3)
    assert list(W1) == list(W2)
    assert np.max(np.abs(E1 - E2)) < 1e-8
