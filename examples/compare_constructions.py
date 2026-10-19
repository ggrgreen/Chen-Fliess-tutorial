"""
compare_constructions.py
========================
Graded construction against the realisation read off the staircase Hankel matrix by SVD, with one shared rank rule (a relative singular-value tolerance tau, applied to every SVD), on the driven
Duffing oscillator, with exact and with regression-fitted coefficients.  Results: results/compare.json.

System: dq1/dt = q2, dq2/dt = -delta q2 - alpha q1 - beta q1^3 + u, y = q1, from x0 = (0.5, 0.3) (as in the paper).

Data: N_TRIALS trials of length T from the same initial state, each driven by its own random multi-sine input
(three components, frequencies 1-15 Hz, random phases, standard deviation SD_U), sampled at 1 ms.  Pooling trials
from one initial state is legitimate because the coefficients depend only on the initial state, not on the input.
Trials are split into training (TRAIN), validation (VAL) and test (TEST).  The target is the increment
y(t) - y(0) with white measurement noise of a given fraction of its standard deviation.

Each noisy window is shifted to start at zero by subtracting its own first sample; because that sample is noisy, one free
offset per window is fitted with the shared coefficients.
Fit: least squares of the increment on the iterated integrals (computed by the word-tree ODE, DOP853), with columns
scaled to unit norm and a ridge chosen on the validation trials from RIDGES (0 means ordinary least squares).

Assessment, on the test trials: every realisation is simulated by DOP853 with the test input and compared with
the true increment (R^2); the fitted series itself (sum of fitted c_w times the test iterated integrals) is
reported as the reference that no realisation of it can beat.


@author: gary green
last edit 19-10-2026
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp

from cfs_tutorial import graded_nilpotent_realize
from cfs_tutorial.signature import sampled_iterated_integrals
from cfs_tutorial.integrate import simulate_realisation_sampled
from cfs_tutorial.hankel import staircase_realize, max_coef_error, nilpotency_residual, beyond_K
from duffing_symbolic import duffing_coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA, ALPHA, BETA, X0 = 0.3, -1.0, 1.0, (0.5, 0.3)
DT = 1e-3
N_TRIALS, TRAIN, VAL = 24, 12, 4          # the remaining 8 are test trials
SD_U = 0.3
CONFIGS = [(4, 0.15), (4, 0.30), (6, 0.15), (6, 0.30)]
NOISES = (0.0, 0.001, 0.01, 0.03, 0.1)
SEEDS = int(os.environ.get("SEEDS", 50))
TAUS = (1e-10, 1e-6, 1e-4, 1e-2)
RIDGES = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)


def make_input(rng):
    f = rng.uniform(1.0, 15.0, 3); ph = rng.uniform(0, 2 * np.pi, 3); a = rng.normal(0, 1, 3)
    a *= SD_U / np.sqrt(0.5 * np.sum(a ** 2))
    return lambda s: np.array([np.sum(a * np.sin(2 * np.pi * f * s + ph))])


def duffing(t, u):
    rhs = lambda s, q: [q[1], -DELTA * q[1] - ALPHA * q[0] - BETA * q[0] ** 3 + u(s)[0]]
    sol = solve_ivp(rhs, (t[0], t[-1]), list(X0), t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[0] - X0[0]


def r2(p, y):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def demean(blocks):
    """Remove the mean of each window: equivalent to fitting one free offset (empty-word coefficient) per window."""
    return [b - b.mean(axis=-1, keepdims=True) for b in blocks]


def fit(E_tr, y_tr, E_va, y_va, offsets=True):
    """Each window carries its own offset: the starting value it was shifted by is measured with noise, so a free
    constant per window is fitted alongside the shared coefficients (by removing each window's mean)."""
    dm = demean if offsets else (lambda b: list(b))
    X = np.hstack(dm(E_tr)).T; y = np.concatenate(dm(y_tr))
    scale = np.linalg.norm(X, axis=0); scale[scale == 0] = 1.0
    Xs = X / scale
    s = np.linalg.svd(Xs, compute_uv=False)
    best = None
    for lam in RIDGES:
        if lam == 0:
            b = np.linalg.lstsq(Xs, y, rcond=None)[0]
        else:
            b = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
        c = b / scale
        e = np.mean([np.var(yv - c @ Ev) if offsets else np.mean((yv - c @ Ev) ** 2) for Ev, yv in zip(E_va, y_va)])
        if best is None or e < best[0]:
            best = (e, lam, c)
    cond_raw = np.linalg.cond(X); cond_scaled = s[0] / s[-1]
    return best[2], best[1], float(cond_raw), float(cond_scaled)


def to_real(g):
    return dict(A=g["A"], lam=g["lam"], mu=g["mu"], dim_total=g["dim_total"])


if __name__ == "__main__":
    t0 = time.time()
    out = dict(settings=dict(DELTA=DELTA, ALPHA=ALPHA, BETA=BETA, X0=X0, DT=DT, N_TRIALS=N_TRIALS, TRAIN=TRAIN,
                             VAL=VAL, SD_U=SD_U, CONFIGS=CONFIGS, NOISES=NOISES, SEEDS=SEEDS, TAUS=TAUS,
                             RIDGES=RIDGES), runs=[])
    for K, T in CONFIGS:
        t = np.arange(0, T + DT / 2, DT)
        rng = np.random.default_rng(100 + K)
        inputs = [make_input(rng) for _ in range(N_TRIALS)]
        usamp = [np.array([u(s)[0] for s in t]) for u in inputs]
        E, words = zip(*[sampled_iterated_integrals(t, us, K) for us in usamp]); words = words[0]   # sampled data: iisignature
        Y = [duffing(t, u) for u in inputs]
        exact = dict(duffing_coeffs(order=K, delta=DELTA, alpha=ALPHA, beta=BETA, q1_0=X0[0], q2_0=X0[1])); exact[()] = 0.0
        tr, va, te = range(TRAIN), range(TRAIN, TRAIN + VAL), range(TRAIN + VAL, N_TRIALS)
        sdy = np.std(np.concatenate([Y[i] for i in tr]))
        # truncation floor: the exact truncated series on the test trials
        cvec_exact = np.array([exact.get(w, 0.0) for w in words])
        r2_trunc = float(np.mean([r2(cvec_exact @ E[i], Y[i]) for i in te]))
        print(f"K={K} T={T}: exact truncated series, test R2 {r2_trunc:.6f}", flush=True)
        sources = [("exact", 0.0, 0, exact)]
        for nz in NOISES:
            for sd in range(SEEDS if nz > 0 else 1):
                nrng = np.random.default_rng(1000 * sd + int(nz * 1e4))
                Yn = [(lambda z: z - z[0])(y + nz * sdy * nrng.standard_normal(len(y))) for y in Y]   # each noisy window shifted to start at zero
                c, lam, cr, cs = fit([E[i] for i in tr], [Yn[i] for i in tr], [E[i] for i in va], [Yn[i] for i in va])
                coef = {(): 0.0}; coef.update({w: float(v) for w, v in zip(words, c)})
                sources.append(("fit", nz, sd, coef, lam, cr, cs, c))
        for src in sources:
            kind, nz, sd, coef = src[:4]
            cvec = np.array([coef.get(w, 0.0) for w in words])
            series_r2 = float(np.mean([r2(cvec @ E[i], Y[i]) for i in te]))
            coef_err = float(np.max(np.abs(cvec - cvec_exact)))
            for tau in TAUS:
                reals = dict(graded=to_real(graded_nilpotent_realize(coef, m=1, order=K, tol=tau)),
                             staircase=staircase_realize(coef, 1, K, tau))
                for name, R in reals.items():
                    rec = dict(K=K, T=T, source=kind, noise=nz, seed=sd, tau=tau, method=name, dim=R["dim_total"],
                               series_r2=series_r2, coef_err_vs_exact=coef_err)
                    if kind == "fit":
                        rec.update(ridge=src[4], cond_raw=src[5], cond_scaled=src[6])
                    if R["dim_total"] == 0:
                        rec.update(test_r2=float("nan"), recon_err=float("nan"), nilp=0.0, beyond=0.0)
                    else:
                        rec["recon_err"] = max_coef_error(R, coef, 1, K)
                        rec["nilp"] = nilpotency_residual(R, 1, K)
                        rec["beyond"] = beyond_K(R, 1, K)
                        try:
                            rec["test_r2"] = float(np.mean([r2(simulate_realisation_sampled(R["A"], R["lam"], R["mu"], t, usamp[i]), Y[i]) for i in te]))   # same sampled input as the features
                        except AssertionError:
                            rec["test_r2"] = float("nan")
                    out["runs"].append(rec)
            if kind == "exact" or sd == 0:
                rr = [r for r in out["runs"] if r["K"] == K and r["T"] == T and r["source"] == kind and r["noise"] == nz and r["seed"] == sd and r["tau"] == 1e-6]
                print(f"  {kind} noise {nz:g}: series R2 {series_r2:.6f} | " + " | ".join(f"{r['method']} dim {r['dim']} R2 {r['test_r2']:.6f}" for r in rr) + f"  ({time.time() - t0:.0f}s)", flush=True)
        # locality: the noiseless fit, used from a different initial state
        X1 = (0.6, 0.2)
        rhs1 = lambda s, q, u: [q[1], -DELTA * q[1] - ALPHA * q[0] - BETA * q[0] ** 3 + u(s)[0]]
        loc = []
        fit0 = [s_ for s_ in sources if s_[0] == "fit" and s_[1] == 0.0][0]
        c0 = np.array([fit0[3].get(w, 0.0) for w in words])
        for i in te:
            sol = solve_ivp(lambda s, q: rhs1(s, q, inputs[i]), (t[0], t[-1]), list(X1), t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15)
            loc.append(r2(c0 @ E[i], sol.y[0] - X1[0]))
        per_len = {}
        for s_ in sources:
            if s_[0] == "fit" and s_[2] == 0:
                per_len[f"{s_[1]:g}"] = {L: float(max(abs(s_[3].get(w, 0.0) - exact.get(w, 0.0)) for w in words if len(w) == L)) for L in range(1, K + 1)}
        out["runs_meta_" + f"{K}_{T}"] = dict(r2_trunc=r2_trunc, sdy=float(sdy), other_x0=X1, r2_other_x0=float(np.mean(loc)),
                                             coef_err_by_length=per_len,
                                             exact_max_by_length={L: float(max(abs(exact.get(w, 0.0)) for w in words if len(w) == L)) for L in range(1, K + 1)})
        json.dump(out, open(os.path.join(ROOT, "results", "compare.json"), "w"))
    out["time"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "results", "compare.json"), "w"))
