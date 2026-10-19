"""
driftless.py -- realisations of path signatures (no letter for time), with exact coefficients.
Writes results/driftless.json.

  1. driftless Duffing, K = 2..8: graded and staircase sizes, rank certificate, errors, nilpotency, Lie-Hankel ranks
  2. Heisenberg integrator: a finite series, realised exactly for all times
  3. four-letter system, K = 2..5: sizes
  4. driftless Duffing, K = 6: forward runs against the true system and the truncated series


@author: gary green
last edit 19-10-2026
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import cfs_tutorial as G
from cfs_tutorial.hankel import staircase_realize, max_coef_error, nilpotency_residual
from cfs_tutorial.true_minimal import exact_rank_modular
from cfs_tutorial.lyndon import lie_hankel_matrix
from systems import dduffing_coeffs, dduffing_true, heisenberg_coeffs, heisenberg_true, four_coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rank_gap(H, tol=1e-9):
    sv = np.linalg.svd(np.asarray(H, float), compute_uv=False)
    r = int((sv > tol * sv[0]).sum())
    return r, (float(sv[r - 1] / sv[r]) if r < len(sv) and sv[r] > 0 else float("inf"))


def sizes(c, m, K, modular=True):
    g = G.graded_nilpotent_realize(c, m=m, order=K, tol=1e-12)
    gr = dict(A=g["A"], lam=g["lam"], mu=g["mu"], dim_total=g["dim_total"])
    st = staircase_realize(c, m, K, 1e-12)
    H, _ = G.staircase_hankel_matrix(c, list(range(m + 1)), K)
    r, gap = rank_gap(H, 1e-12)
    out = dict(K=K, graded_dims=list(map(int, g["dims"])), graded=int(g["dim_total"]), staircase=int(st["dim_total"]),
               rank_float=r, sv_gap=gap, graded_err=max_coef_error(gr, c, m, K), staircase_err=max_coef_error(st, c, m, K),
               graded_nilp=nilpotency_residual(gr, m, K) if (m + 1) ** (K + 1) <= 70000 else None,
               staircase_nilp=nilpotency_residual(st, m, K) if (m + 1) ** (K + 1) <= 70000 else None)
    if modular:
        out["rank_modular"] = int(exact_rank_modular(c, list(range(m + 1)), K)[0])
    return out, gr, st


if __name__ == "__main__":
    t0 = time.time(); R = {}
    # ---------------------------------------------------------------- 1. driftless Duffing
    R["dduffing"] = []
    c10 = dduffing_coeffs(10)
    for K in range(2, 9):
        c = {w: v for w, v in c10.items() if len(w) <= K}
        d, _, _ = sizes(c, 1, K)
        HL, rows, _ = lie_hankel_matrix(c, 1, K); d["lie_truncated"], _ = rank_gap(HL)
        R["dduffing"].append(d)
        print("dduffing", d, f"{time.time() - t0:.0f}s", flush=True)
    R["dduffing_nonzero_by_length"] = {L: sum(1 for w, v in c10.items() if len(w) == L and abs(v) > 0) for L in range(1, 11)}
    c2_ = {w: v for w, v in c10.items() if len(w) <= 2}
    g2 = G.graded_nilpotent_realize(c2_, m=1, order=2, tol=1e-12)
    R["dduffing_K2"] = dict(A0=np.asarray(g2["A"][0]).tolist(), A1=np.asarray(g2["A"][1]).tolist(), lam=np.asarray(g2["lam"]).tolist(),
                            mu=np.asarray(g2["mu"]).tolist(), dims=list(map(int, g2["dims"])),
                            coef={"".join(str(a + 1) for a in w) or "e": v for w, v in c2_.items()})
    H6, _ = G.staircase_hankel_matrix({w: v for w, v in c10.items() if len(w) <= 6}, [0, 1], 6)
    sv6 = np.linalg.svd(np.asarray(H6, float), compute_uv=False)
    R["dduffing_sv6"] = dict(sv=sv6[:10].tolist(), rank_tau_1e2=int((sv6 > 1e-2 * sv6[0]).sum()), rank_tau_1e6=int((sv6 > 1e-6 * sv6[0]).sum()))
    R["dduffing_lie_genuine"] = []
    for D in range(1, 6):
        HL, _, _ = lie_hankel_matrix(c10, 1, 2 * D, rows_max=D, cols_max=D)
        r, gap = rank_gap(HL); R["dduffing_lie_genuine"].append(dict(depth=D, rank=r, gap=gap))
    print("lie genuine", R["dduffing_lie_genuine"], flush=True)
    # ---------------------------------------------------------------- 2. Heisenberg
    ch = heisenberg_coeffs(6)
    nz_len = sorted({len(w) for w, v in ch.items() if abs(v) > 0})
    c2 = {w: v for w, v in ch.items() if len(w) <= 2}
    d, gr, st = sizes(c2, 1, 2)
    hres = dict(nonzero_lengths=nz_len, sizes=d, runs=[])
    for T, amp, kind in ((1.0, 1.0, "sines"), (50.0, 5.0, "sines"), (10.0, 0.2, "circle"), (100.0, 0.2, "circle")):
        t = np.linspace(0, T, 4001)
        if kind == "sines":
            u = lambda s, a=amp: np.array([a * np.sin(2 * np.pi * 3 * s), a * np.cos(2 * np.pi * 3.9 * s + 0.4)])
        else:                                   # the path runs round a circle of radius amp once per second
            u = lambda s, a=amp: np.array([-a * 2 * np.pi * np.sin(2 * np.pi * s), a * 2 * np.pi * np.cos(2 * np.pi * s)])
        yt = heisenberg_true(t, u)
        yg = G.simulate_realisation(gr["A"], gr["lam"], gr["mu"], t, u, drift=False)
        ys = G.simulate_realisation(st["A"], st["lam"], st["mu"], t, u, drift=False)
        hres["runs"].append(dict(T=T, amp=amp, kind=kind, graded_err=float(np.max(np.abs(yg - yt))), staircase_err=float(np.max(np.abs(ys - yt))),
                                 max_abs_y=float(np.max(np.abs(yt)))))
    HL, _, _ = lie_hankel_matrix(ch, 1, 6, rows_max=3, cols_max=3); hres["lie_rank"], _ = rank_gap(HL)
    R["heisenberg"] = hres
    print("heisenberg", hres, flush=True)
    # ---------------------------------------------------------------- 3. four letters
    R["four"] = []
    c4 = four_coeffs(7)
    R["four_nonzero_by_length"] = {L: sum(1 for w, v in c4.items() if len(w) == L and abs(v) > 1e-14) for L in range(1, 8)}
    for K in range(2, 7):
        c = {w: v for w, v in c4.items() if len(w) <= K}
        d, _, _ = sizes(c, 3, K, modular=(K <= 4))
        d["word_tree"] = sum(4 ** L for L in range(K + 1))
        R["four"].append(d)
        print("four", d, f"{time.time() - t0:.0f}s", flush=True)
    g7 = G.graded_nilpotent_realize(c4, m=3, order=7, tol=1e-12)
    g7r = dict(A=g7["A"], lam=g7["lam"], mu=g7["mu"], dim_total=g7["dim_total"])
    R["four_K7_graded"] = dict(graded=int(g7["dim_total"]), dims=list(map(int, g7["dims"])), err=max_coef_error(g7r, c4, 3, 7),
                               word_tree=sum(4 ** L for L in range(8)))
    print("four K7", R["four_K7_graded"], flush=True)
    # ---------------------------------------------------------------- 4. forward runs, driftless Duffing K = 6
    c6 = {w: v for w, v in c10.items() if len(w) <= 6}
    _, gr6, st6 = sizes(c6, 1, 6, modular=False)
    R["forward"] = []
    cases = [(0.15, 1.0, 0.0, "sines"), (10.0, 1.0, 0.0, "sines"), (10.0, 10.0, 0.0, "sines"), (10.0, 30.0, 0.0, "sines"),
             (10.0, 100.0, 0.0, "sines"), (10.0, 1.0, 0.1, "sines"), (10.0, 1.0, 0.3, "sines"),
             (10.0, 0.2, 0.0, "circle"), (100.0, 0.2, 0.0, "circle")]
    for T, amp, bias, kind in cases:
        t = np.linspace(0, T, 4001 if T <= 10 else 20001)
        if kind == "sines":
            u = lambda s, a=amp, b=bias: np.array([0.3 * a * np.sin(2 * np.pi * 3 * s) + b, 0.1 * a * np.sin(2 * np.pi * 7 * s)])
        else:
            u = lambda s, a=amp: np.array([-a * 2 * np.pi * np.sin(2 * np.pi * s), a * 2 * np.pi * np.cos(2 * np.pi * s)])
        yt = dduffing_true(t, u)
        E, W = G.iterated_integrals(t, u, 1, 6, drift=False)
        cv = np.array([c6.get(w, 0.0) for w in W]); yser = cv @ E
        yg = G.simulate_realisation(gr6["A"], gr6["lam"], gr6["mu"], t, u, drift=False)
        ys = G.simulate_realisation(st6["A"], st6["lam"], st6["mu"], t, u, drift=False)
        iw = {w: i for i, w in enumerate(W)}
        area = 0.5 * (E[iw[(0, 1)]] - E[iw[(1, 0)]])
        R["forward"].append(dict(T=T, amp=amp, bias=bias, kind=kind, path_max=float(max(np.max(np.abs(E[iw[(0,)]])), np.max(np.abs(E[iw[(1,)]])))),
                                 area_max=float(np.max(np.abs(area))), series_vs_true=float(np.max(np.abs(yser - yt))), graded_vs_series=float(np.max(np.abs(yg - yser))),
                                 staircase_vs_series=float(np.max(np.abs(ys - yser))), max_abs_y=float(np.max(np.abs(yt))),
                                 max_abs_series=float(np.max(np.abs(yser)))))
    print("forward", R["forward"], flush=True)
    R["time"] = time.time() - t0
    json.dump(R, open(os.path.join(ROOT, "results", "driftless.json"), "w"), indent=1, default=float)
