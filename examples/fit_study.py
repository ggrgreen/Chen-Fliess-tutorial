"""
fit_study.py -- fitting truncated series to sampled data, with and without a letter for time.
Writes results/fit_study.json.

Both systems are fitted in the same way: 24 trials of 0.15 s from one starting state (12 training, 4 validation,
8 test), each with its own random multi-sine input on every channel, sampled every 1 ms; iterated integrals of the
sampled path by iisignature; each noisy window shifted by its own first sample, with one free offset per window;
ridge penalty chosen on the validation trials.  Reported:
  * R^2 on the test trials (median over noise draws),
  * coefficient errors on the normalised scale c_w T^L / L!, relative to the largest normalised exact coefficient,
  * sizes of the staircase and graded realisations of the fitted series.


@author: gary green
last edit 19-10-2026
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import factorial
import numpy as np

import cfs_tutorial as G
from cfs_tutorial.hankel import staircase_realize
from compare_constructions import duffing, r2, fit as ridge_fit, X0, DT, N_TRIALS, TRAIN, VAL
from duffing_symbolic import duffing_coeffs
from systems import dduffing_coeffs, dduffing_true

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 0.15
SEEDS = 20
NOISES = (0.0, 0.001, 0.01, 0.03, 0.1)


def multisine(rng, sd=0.3):
    f = rng.uniform(1.0, 15.0, 3); ph = rng.uniform(0, 2 * np.pi, 3); a = rng.normal(0, 1, 3)
    a *= sd / np.sqrt(0.5 * np.sum(a ** 2))
    return lambda s: np.sum(a * np.sin(2 * np.pi * f * np.asarray(s, float)[..., None] + ph), axis=-1)


def system(name, K):
    t = np.arange(0, T + DT / 2, DT)
    rng = np.random.default_rng(300 + K)
    if name == "duffing":                                  # one input, letter 0 is time
        ins = [multisine(rng) for _ in range(N_TRIALS)]
        uf = [(lambda s, g=g: np.array([g(s)])) for g in ins]
        Y = [duffing(t, u) for u in uf]
        U = [g(t) for g in ins]
        E = [G.sampled_iterated_integrals(t, Ui, K)[0] for Ui in U]
        W = list(G.sampled_iterated_integrals(t, U[0], K)[1])
        exact = dict(duffing_coeffs(order=K, q1_0=X0[0], q2_0=X0[1]))
    else:                                                  # two channels, no letter for time
        ins = [(multisine(rng), multisine(rng)) for _ in range(N_TRIALS)]
        uf = [(lambda s, g=g: np.array([g[0](s), g[1](s)])) for g in ins]
        Y = [dduffing_true(t, u) for u in uf]
        U = [np.column_stack([g[0](t), g[1](t)]) for g in ins]
        E = [G.sampled_iterated_integrals(t, Ui, K, drift=False)[0] for Ui in U]
        W = list(G.sampled_iterated_integrals(t, U[0], K, drift=False)[1])
        exact = dduffing_coeffs(K)
    exact[()] = 0.0
    return t, E, W, Y, exact


def norm_rel_err(c, ce, W, K):
    sc = np.array([T ** len(w) / factorial(len(w)) for w in W])
    ref = np.max(np.abs(ce * sc))
    return {L: float(max(abs(c[i] - ce[i]) * sc[i] for i, w in enumerate(W) if len(w) == L) / ref) for L in range(1, K + 1)}


if __name__ == "__main__":
    t0 = time.time(); out = dict(runs=[])
    for name in ("duffing", "dduffing"):
        for K in (4, 6):
            t, E, W, Y, exact = system(name, K)
            ce = np.array([exact.get(w, 0.0) for w in W])
            tr, va, te = range(TRAIN), range(TRAIN, TRAIN + VAL), range(TRAIN + VAL, N_TRIALS)
            sdy = float(np.std(np.concatenate([Y[i] for i in tr])))
            rec = dict(system=name, K=K, n_words=len(W), noise={})
            for nz in NOISES:
                r2s, errs, sizes, nooff, ols = [], [], [], [], []
                for sd in range(SEEDS if nz > 0 else 1):
                    nrng = np.random.default_rng(1000 * sd + int(nz * 1e4))
                    Yn = [(lambda z: z - z[0])(y + nz * sdy * nrng.standard_normal(len(y))) for y in Y]
                    c = ridge_fit([E[i] for i in tr], [Yn[i] for i in tr], [E[i] for i in va], [Yn[i] for i in va])[0]
                    r2s.append(np.mean([r2(c @ E[i], Y[i]) for i in te]))
                    errs.append(norm_rel_err(c, ce, W, K))
                    cd = {(): 0.0}; cd.update(dict(zip(W, c)))
                    sizes.append((int(staircase_realize(cd, 1, K, 1e-6)["dim_total"]),
                                  int(G.graded_nilpotent_realize(cd, m=1, order=K, tol=1e-6)["dim_total"])))
                    if nz == 0.01:
                        cn = ridge_fit([E[i] for i in tr], [Yn[i] for i in tr], [E[i] for i in va], [Yn[i] for i in va], offsets=False)[0]
                        nooff.append(np.mean([r2(cn @ E[i], Y[i]) for i in te]))
                        co = ridge_fit([E[i] for i in tr], [Yn[i] for i in tr], [E[i] for i in va], [Yn[i] for i in va])  # ridge path
                        X = np.hstack([e - e.mean(axis=1, keepdims=True) for e in (E[i] for i in tr)]).T
                        yv = np.concatenate([(Yn[i] - Yn[i].mean()) for i in tr])
                        sc = np.linalg.norm(X, axis=0); cl = np.linalg.lstsq(X / sc, yv, rcond=None)[0] / sc
                        ols.append(np.mean([r2(cl @ E[i], Y[i]) for i in te]))
                rec["noise"][f"{nz:g}"] = dict(r2_median=float(np.median(r2s)), r2_lo=float(np.percentile(r2s, 5)),
                                               coef_rel_err={L: float(np.median([e[L] for e in errs])) for L in range(1, K + 1)},
                                               sizes=dict(staircase=sorted(set(a for a, b in sizes)), graded=sorted(set(b for a, b in sizes)),
                                                          n=len(sizes)),
                                               r2_no_offset_median=float(np.median(nooff)) if nooff else None,
                                               r2_ols_median=float(np.median(ols)) if ols else None)
            # exact series on the test trials (truncation only)
            rec["r2_truncated_series"] = float(np.mean([r2(ce @ E[i], Y[i]) for i in te]))
            out["runs"].append(rec)
            print(name, K, {k: (round(v["r2_median"], 6), v["sizes"], v["r2_no_offset_median"], v["r2_ols_median"]) for k, v in rec["noise"].items()},
                  {k: v["coef_rel_err"] for k, v in rec["noise"].items() if k in ("0", "0.01")}, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(ROOT, "results", "fit_study.json"), "w"), indent=1)
