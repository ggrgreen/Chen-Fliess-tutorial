"""
check_beyond_K.py
=================
Do the bilinear realisations, used as solutions, add terms beyond the truncation order K?

For each realisation (graded; staircase Hankel SVD) of the Duffing coefficients, exact and regression-fitted (1 % and
10 % noise), at a tolerance that discards nothing (1e-6) and one that discards (1e-2):
  1. the largest coefficient the realisation assigns to any word of length K+1 .. K+4 (zero for the truncated series);
  2. the largest operator norm of a product of K+1 generators (zero if and only if exactly nilpotent, for a
     realisation of this size), computed for every such product;
  3. the solution check: the bilinear system is integrated (DOP853) with an input, and compared with the truncated
     series sum_{|w|<=K} c_w I_w(t) of the same realisation, evaluated with the same input.  Any difference is the
     contribution of words longer than K.  Reported over the fitting window (0.15 s) and over 1 s and 3 s.
Results: results/check_beyond_K.json.


@author: gary green
last edit 19-10-2026
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from cfs_tutorial import graded_nilpotent_realize
from cfs_tutorial.signature import sampled_iterated_integrals
from cfs_tutorial.integrate import iterated_integrals, simulate_realisation
from cfs_tutorial.hankel import staircase_realize, coefficient, nilpotency_residual
import compare_constructions as CC
from duffing_symbolic import duffing_coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    out = []
    u = lambda s: np.array([0.3 * np.sin(2 * np.pi * 3 * s) + 0.1 * np.sin(2 * np.pi * 7 * s)])
    for K in (4, 6):
        T = 0.15
        t = np.arange(0, T + CC.DT / 2, CC.DT)
        rng = np.random.default_rng(100 + K)
        inputs = [CC.make_input(rng) for _ in range(CC.N_TRIALS)]
        E, words = zip(*[sampled_iterated_integrals(t, np.array([ui(s)[0] for s in t]), K) for ui in inputs]); words = words[0]   # sampled data: iisignature
        Y = [CC.duffing(t, ui) for ui in inputs]
        sdy = np.std(np.concatenate(Y[:CC.TRAIN]))
        exact = dict(duffing_coeffs(order=K, delta=CC.DELTA, alpha=CC.ALPHA, beta=CC.BETA, q1_0=CC.X0[0], q2_0=CC.X0[1])); exact[()] = 0.0
        sources = [("exact", exact)]
        for nz in (0.01, 0.1):
            nrng = np.random.default_rng(int(nz * 1e4))
            Yn = [(lambda z: z - z[0])(y + nz * sdy * nrng.standard_normal(len(y))) for y in Y]   # each noisy window shifted to start at zero
            c, *_ = CC.fit([E[i] for i in range(CC.TRAIN)], [Yn[i] for i in range(CC.TRAIN)],
                           [E[i] for i in range(CC.TRAIN, CC.TRAIN + CC.VAL)], [Yn[i] for i in range(CC.TRAIN, CC.TRAIN + CC.VAL)])
            coef = {(): 0.0}; coef.update({w: float(v) for w, v in zip(words, c)})
            sources.append((f"fit {nz:g}", coef))
        for name, coef in sources:
            for tau in (1e-6, 1e-2):
                g = graded_nilpotent_realize(coef, m=1, order=K, tol=tau)
                reals = dict(graded=dict(A=g["A"], lam=g["lam"], mu=g["mu"], dim_total=g["dim_total"]),
                             staircase=staircase_realize(coef, 1, K, tau))
                for mname, R in reals.items():
                    beyond = {L: max(abs(coefficient(R, w)) for w in itertools.product((0, 1), repeat=L)) for L in range(K + 1, K + 5)}
                    nilp = nilpotency_residual(R, 1, K)
                    sol = {}
                    for Th in (0.15, 1.0, 3.0):
                        th = np.arange(0, Th + CC.DT / 2, CC.DT)
                        y_bil = simulate_realisation(R["A"], R["lam"], R["mu"], th, u)
                        Eh, wh = iterated_integrals(th, u, 1, K)
                        cvec = np.array([coefficient(R, w) for w in wh])
                        y_ser = cvec @ Eh
                        sol[Th] = dict(max_abs_diff=float(np.max(np.abs(y_bil - y_ser))),
                                       rel_diff=float(np.max(np.abs(y_bil - y_ser)) / max(np.max(np.abs(y_ser)), 1e-300)),
                                       max_abs_y=float(np.max(np.abs(y_ser))))
                    rec = dict(K=K, source=name, tau=tau, method=mname, dim=R["dim_total"], nilp=nilp, coef_max=float(max(abs(v) for v in coef.values())),
                               beyond={str(k): v for k, v in beyond.items()}, solution={str(k): v for k, v in sol.items()})
                    out.append(rec)
                    print(f"K={K} {name:9s} tau={tau:.0e} {mname:6s} dim {R['dim_total']:2d}  nilp {nilp:.1e}  "
                          f"beyond K: " + " ".join(f"L{L}:{v:.1e}" for L, v in beyond.items()) + "  |  bilinear - series: "
                          + " ".join(f"{Th}s {sol[Th]['max_abs_diff']:.1e} (rel {sol[Th]['rel_diff']:.1e})" for Th in sol), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "check_beyond_K.json"), "w"), indent=1)
