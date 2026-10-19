"""
run_all.py
==========
Every number of the paper.  Writes results/n1.json.

  sanity      a random strictly triangular bilinear system of dimension 4 (order 3), recovered from its coefficients
  example     the Duffing coefficients at K = 2, the staircase Hankel matrix, and every step of the graded construction
  dims        K = 2..8: graded block sizes; staircase Hankel rank (float SVD and modular certificate); the sum over
              lengths of the ranks of the row blocks; true minimal dimension; reconstruction errors; nilpotency
  ode         the K = 2 realisation written out, and its solution against the true Duffing response
  forward     exact coefficients, K = 6 and 8: realisations against the true Duffing response over the 0.15 s window,
              for the training input and a held-out input; graded against staircase realisation
  breakdown   K = 6 graded realisation against the true response, for longer horizons and larger inputs
  divergence  the same realisation over 10 s, against the true response
  kernels     K = 8: number of words per kernel order; each order's contribution computed from the kernel (Gauss
              quadrature on the simplex) and from the iterated integrals (DOP853); for K = 4, 6, 8 the sum of all
              kernels of all orders against the true response of the Duffing system


@author: gary green
last edit 19-10-2026
"""
import os, sys, json, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp

import cfs_tutorial as G
from duffing_symbolic import duffing_coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA, ALPHA, BETA, X0 = 0.3, -1.0, 1.0, (0.5, 0.3)
DT = 1e-3
U_TRAIN = lambda s: np.array([0.3 * np.sin(2 * np.pi * 3.0 * s) + 0.1 * np.sin(2 * np.pi * 7.0 * s)])
U_TEST = lambda s: np.array([0.25 * np.sin(2 * np.pi * 4.5 * s + 0.7) - 0.15 * np.cos(2 * np.pi * 2.0 * s)])
U_KERNEL = lambda s: 2.0 * np.sin(6 * np.pi * np.asarray(s)) + 0.5 * np.sin(14 * np.pi * np.asarray(s))


def coeffs(K):
    c = dict(duffing_coeffs(order=K, delta=DELTA, alpha=ALPHA, beta=BETA, q1_0=X0[0], q2_0=X0[1]))
    c[()] = 0.0
    return c


def duffing(t, u):
    rhs = lambda s, q: [q[1], -DELTA * q[1] - ALPHA * q[0] - BETA * q[0] ** 3 + u(s)[0]]
    sol = solve_ivp(rhs, (t[0], t[-1]), list(X0), t_eval=t, method="DOP853", rtol=1e-13, atol=1e-15)
    return sol.y[0] - X0[0]


def r2(p, y):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def as_real(g):
    return dict(A=g["A"], lam=g["lam"], mu=g["mu"], dim_total=g["dim_total"])


def words(m, L):
    return [w for l in range(L + 1) for w in itertools.product(range(m + 1), repeat=l)]


def row_block_ranks(coef, K, tol=1e-10):
    H, W = G.staircase_hankel_matrix(coef, [0, 1], K)
    H = np.array(H, float)
    out = []
    for l in range(K + 1):
        rows = [i for i, w in enumerate(W) if len(w) == l]
        B = H[rows]
        s = np.linalg.svd(B, compute_uv=False)
        out.append(int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0)
    s = np.linalg.svd(H, compute_uv=False)
    return out, int((s > tol * s[0]).sum()), s.tolist()


if __name__ == "__main__":
    t0 = time.time()
    R = {}
    # ------------------------------------------------------------------ sanity
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sanity_check import make_ground_truth, true_coeffs, ORDER, N_TRUE
    A0, A1, lam, mu = make_ground_truth()
    cs = true_coeffs(A0, A1, lam, mu, ORDER)
    g = G.graded_nilpotent_realize(cs, m=1, order=ORDER, tol=1e-8)
    st = G.staircase_realize(cs, 1, ORDER, 1e-8)
    R["sanity"] = dict(n_true=N_TRUE, order=ORDER, word_tree=2 ** (ORDER + 1) - 1, graded_dim=g["dim_total"],
                       graded_dims=g["dims"], graded_err=G.verify_exact(g, cs, 1, ORDER), graded_nilp=G.verify_nilpotent(g, ORDER, 1),
                       staircase_dim=st["dim_total"], staircase_err=G.max_coef_error(st, cs, 1, ORDER),
                       staircase_nilp=G.nilpotency_residual(st, 1, ORDER), row_block_ranks=row_block_ranks(cs, ORDER)[0])
    print("sanity", R["sanity"], flush=True)
    # ------------------------------------------------------------------ worked example, K = 2
    c2 = coeffs(2)
    H2, W2 = G.staircase_hankel_matrix(c2, [0, 1], 2)
    H2 = np.array(H2, float)
    s2 = np.linalg.svd(H2, compute_uv=False)
    g2 = G.graded_nilpotent_realize(c2, m=1, order=2, tol=1e-10)
    R["example"] = dict(coef={"".join(map(str, w)) or "e": v for w, v in c2.items()}, hankel=H2.tolist(),
                        hankel_words=["".join(map(str, w)) or "e" for w in W2], hankel_sv=s2.tolist(),
                        hankel_rank=int((s2 > 1e-10 * s2[0]).sum()), dims=g2["dims"],
                        steps={str(l): dict(Q=g2["steps"][l]["Q"].tolist(), sv=g2["steps"][l]["sv"].tolist(),
                                            words=["".join(map(str, w)) or "e" for w in g2["steps"][l]["words"]],
                                            R=g2["reduced_basis"][l].tolist()) for l in range(3)},
                        A0=g2["A"][0].tolist(), A1=g2["A"][1].tolist(), lam=g2["lam"].tolist(), mu=g2["mu"].tolist())
    # ------------------------------------------------------------------ dimensions K = 2..8
    R["dims"] = []
    for K in range(2, 9):
        c = coeffs(K)
        g = G.graded_nilpotent_realize(c, m=1, order=K, tol=1e-10)
        rb, rank_float, sv = row_block_ranks(c, K)
        rank_mod = G.exact_rank_modular(c, [0, 1], K)[0] if K <= 8 else None
        st = G.staircase_realize(c, 1, K, 1e-10)
        rec = dict(K=K, graded_dims=g["dims"], graded_dim=g["dim_total"], sum_row_block_ranks=int(sum(rb)), row_block_ranks=rb,
                   staircase_rank_float=rank_float, staircase_rank_modular=rank_mod, word_tree=2 ** (K + 1) - 1,
                   graded_err=G.verify_exact(g, c, 1, K), graded_nilp=G.verify_nilpotent(g, K, 1),
                   staircase_dim=st["dim_total"], staircase_err=G.max_coef_error(st, c, 1, K),
                   staircase_nilp=G.nilpotency_residual(st, 1, K), sv_gap=float(sv[rank_float - 1] / sv[rank_float]) if rank_float < len(sv) and sv[rank_float] > 0 else None)
        if K in (4, 6):
            tm = G.true_minimal_realize(c, [0, 1], K, method="exact")
            v = G.verify_true_minimal(tm, c, K, [0, 1])
            rec.update(true_min_dim=tm["dim"], true_min_method="exact", true_min_err=v["max_reconstruction_error"],
                       true_min_products=G.products_vanish(tm, [0, 1], K))
        if K == 8:
            tm = G.true_minimal_realize(c, [0, 1], K, method="numeric")
            v = G.verify_true_minimal(tm, c, K, [0, 1])
            rec.update(true_min_dim=tm["dim"], true_min_method="numeric", true_min_err=v["max_reconstruction_error"],
                       true_min_products=G.products_vanish(tm, [0, 1], K),
                       rank_mod_second_prime=G.exact_rank_modular(c, [0, 1], K, prime=2 ** 61 - 1)[0])
        R["dims"].append(rec)
        print("dims", {k: v for k, v in rec.items() if k not in ("row_block_ranks",)}, f"{time.time() - t0:.0f}s", flush=True)
    # ------------------------------------------------------------------ explicit ODE, K = 2
    t = np.arange(0, 0.15 + DT / 2, DT)
    y_true = duffing(t, U_TRAIN)
    y2 = G.simulate_bilinear(g2, t, U_TRAIN)
    R["ode"] = dict(max_abs_err=float(np.max(np.abs(y2 - y_true))), max_abs_y=float(np.max(np.abs(y_true))), r2=r2(y2, y_true))
    # ------------------------------------------------------------------ forward validation with exact coefficients
    R["forward"] = {}
    for K in (6, 8):
        c = coeffs(K)
        g = as_real(G.graded_nilpotent_realize(c, m=1, order=K, tol=1e-10))
        st = G.staircase_realize(c, 1, K, 1e-10)
        rec = {}
        for name, u in (("train", U_TRAIN), ("test", U_TEST)):
            yt = duffing(t, u)
            E, ws = G.signature_matrix(t, u, K)
            ys = np.array([c.get(w, 0.0) for w in ws]) @ E
            yg = G.simulate_bilinear(g, t, u); yst = G.simulate_bilinear(st, t, u)
            rec[name] = dict(graded_vs_true=float(np.max(np.abs(yg - yt))), staircase_vs_true=float(np.max(np.abs(yst - yt))),
                             series_vs_true=float(np.max(np.abs(ys - yt))), graded_vs_series=float(np.max(np.abs(yg - ys))),
                             staircase_vs_series=float(np.max(np.abs(yst - ys))), graded_vs_staircase=float(np.max(np.abs(yg - yst))),
                             max_abs_y=float(np.max(np.abs(yt))), r2_graded=r2(yg, yt))
        rec.update(graded_dim=g["dim_total"], staircase_dim=st["dim_total"])
        R["forward"][str(K)] = rec
        print("forward", K, rec, flush=True)
    # ------------------------------------------------------------------ breakdown and divergence, K = 6
    c6 = coeffs(6); g6 = as_real(G.graded_nilpotent_realize(c6, m=1, order=6, tol=1e-10))
    R["breakdown"] = []
    for T_h in (0.15, 0.3, 0.5, 1.0, 2.0, 3.0):
        for amp in (0.3, 1.0, 3.0):
            u = lambda s, amp=amp: np.array([amp * np.sin(2 * np.pi * 3.0 * s) + 0.3 * amp * np.sin(2 * np.pi * 7.0 * s)])
            th = np.arange(0, T_h + DT / 2, DT)
            yt = duffing(th, u); yg = G.simulate_bilinear(g6, th, u)
            R["breakdown"].append(dict(T=T_h, amp=amp, r2=r2(yg, yt), max_abs_err=float(np.max(np.abs(yg - yt))),
                                       max_abs_y=float(np.max(np.abs(yt)))))
    th = np.arange(0, 10 + DT / 2, DT)
    yt = duffing(th, U_TRAIN); yg = G.simulate_bilinear(g6, th, U_TRAIN)
    R["divergence"] = dict(T=10.0, max_abs_true=float(np.max(np.abs(yt + X0[0]))), max_abs_realised=float(np.max(np.abs(yg + X0[0]))),
                           max_abs_true_increment=float(np.max(np.abs(yt))), max_abs_realised_increment=float(np.max(np.abs(yg))))
    print("divergence", R["divergence"], flush=True)
    # ------------------------------------------------------------------ kernels
    c8 = coeffs(8)
    T_K = 0.1
    R["kernels"] = dict(T=T_K, counts={}, contributions={})
    E, ws = G.signature_matrix(np.array([0.0, T_K]), lambda s: np.array([U_KERNEL(s)]), 8)
    for n in range(0, 5):
        terms = G.extract_kernel_terms(c8, n, 8)
        R["kernels"]["counts"][str(n)] = len(terms)
        from_ii = float(sum(c8.get(w, 0.0) * E[i, -1] for i, w in enumerate(ws) if sum(1 for a in w if a == 1) == n))
        if n == 0:
            from_ii = float(sum(c8.get(w, 0.0) * E[i, -1] for i, w in enumerate(ws) if all(a == 0 for a in w)))
        from_k = G.kernel_contribution(c8, n, 8, T_K, U_KERNEL, n_nodes=30) if n <= 4 else None
        R["kernels"]["contributions"][str(n)] = dict(from_kernel=from_k, from_iterated_integrals=from_ii,
                                                     diff=abs(from_k - from_ii) if from_k is not None else None)
    yk = duffing(np.array([0.0, T_K]), lambda s: np.array([U_KERNEL(s)]))
    R["kernels"]["true_increment"] = float(yk[-1])
    R["kernels"]["series_total"] = float(sum(c8.get(w, 0.0) * E[i, -1] for i, w in enumerate(ws)))
    # symmetric kernels on the whole cube [0, t]^n give the same contributions as the ordered kernels on the simplex
    R["kernels"]["cube"] = {}
    for n in (1, 2, 3):
        v_cube = G.cube_contribution(c8, n, 8, T_K, U_KERNEL, n_nodes=30)
        v_simp = G.kernel_contribution(c8, n, 8, T_K, U_KERNEL, n_nodes=30)
        R["kernels"]["cube"][str(n)] = dict(cube=v_cube, simplex=v_simp, rel_diff=abs(v_cube - v_simp) / abs(v_simp))
    # the starting value placed in the empty-word coefficient gives realisations of the same size
    R["offset"] = []
    for K in range(2, 9):
        c_ = coeffs(K); c1 = dict(c_); c1[()] = X0[0]
        R["offset"].append(dict(K=K, staircase=[G.staircase_realize(cc, 1, K, 1e-12)["dim_total"] for cc in (c_, c1)],
                                graded=[G.graded_nilpotent_realize(cc, m=1, order=K, tol=1e-12)["dim_total"] for cc in (c_, c1)]))
    # all kernels together: the sum over every order of the kernel contributions, against the true response
    R["kernels"]["full"] = {}
    tg = np.linspace(0.01, 0.15, 15)
    ytrue = duffing(np.r_[0.0, tg], lambda s: np.array([U_KERNEL(s)]))[1:]
    for K in (4, 6, 8):
        cK = coeffs(K)
        ysum = np.array([sum(G.kernel_contribution(cK, n, K, tt, U_KERNEL, n_nodes=30) for n in range(0, K + 1)) for tt in tg])
        R["kernels"]["full"][str(K)] = dict(t=tg.tolist(), from_kernels=ysum.tolist(), true=ytrue.tolist(),
                                            max_abs_err=float(np.max(np.abs(ysum - ytrue))), max_abs_y=float(np.max(np.abs(ytrue))))
    print("kernels", {k: v for k, v in R["kernels"].items() if k != "full"}, {K: v["max_abs_err"] for K, v in R["kernels"]["full"].items()}, flush=True)
    # ---------------------------------------------------------------- Lyndon basis: Lie-Hankel ranks
    from cfs_tutorial.lyndon import lie_hankel_matrix, lyndon_words
    from fractions import Fraction
    def rank_gap(H):
        sv = np.linalg.svd(H, compute_uv=False); r = int((sv > 1e-9 * sv[0]).sum())
        return r, float(sv[r - 1] / sv[r]) if r < len(sv) and sv[r] > 0 else float("inf")
    R["lyndon"] = dict(truncated=[], genuine=[])
    for K in range(2, 9):
        H, rows_, cols_ = lie_hankel_matrix(coeffs(K), 1, K)
        r, gap = rank_gap(H)
        st = next(d["staircase_dim"] for d in R["dims"] if d["K"] == K)
        R["lyndon"]["truncated"].append(dict(K=K, n_rows=len(rows_), n_cols=len(cols_), lie_rank=r, gap=gap, staircase_rank=st))
    c10 = coeffs(10)
    for D in range(1, 6):
        H, rows_, cols_ = lie_hankel_matrix(c10, 1, 2 * D, rows_max=D, cols_max=D)
        r, gap = rank_gap(H)
        R["lyndon"]["genuine"].append(dict(depth=D, n_rows=len(rows_), n_cols=len(cols_), lie_rank=r, gap=gap))
    R["lyndon"]["n_lyndon"] = {L: sum(1 for l in lyndon_words(1, 8) if len(l) == L) for L in range(1, 9)}
    print("lyndon", R["lyndon"], flush=True)
    # ---------------------------------------------------------------- accuracy of the iterated integrals
    from math import factorial
    tA = np.arange(0, 0.15 + DT / 2, DT)
    EA, WA = G.iterated_integrals(tA, U_TRAIN, 1, 8); WA = list(WA)
    U1 = 0.3 * (1 - np.cos(6 * np.pi * tA)) / (6 * np.pi) + 0.1 * (1 - np.cos(14 * np.pi * tA)) / (14 * np.pi)
    EB, _ = G.iterated_integrals(tA, U_TRAIN, 1, 8, rtol=3e-14, atol=1e-17)
    acc = {}
    for L in range(1, 9):
        e1 = EA[WA.index((1,) * L)]; x1 = U1 ** L / factorial(L)
        e0 = EA[WA.index((0,) * L)]; x0 = tA ** L / factorial(L)
        rows_ = [i for i, w in enumerate(WA) if len(w) == L]
        mixed = max(np.max(np.abs(EA[i] - EB[i])) / np.max(np.abs(EB[i])) for i in rows_)
        acc[str(L)] = dict(input_only=float(np.max(np.abs(e1 - x1)) / np.max(np.abs(x1))),
                           drift_only=float(np.max(np.abs(e0 - x0)) / np.max(np.abs(x0))), all_vs_tighter=float(mixed))
    R["integral_accuracy"] = acc
    print("integral accuracy", acc, flush=True)
    # ---------------------------------------------------------------- largest possible sizes (random coefficients)
    rng_g = np.random.default_rng(7)
    R["generic"] = []
    for K in range(2, 9):
        cr = {w: float(rng_g.standard_normal()) for w in words(1, K)}; cr[()] = 0.0
        Hs, _ = G.staircase_hankel_matrix(cr, [0, 1], K); Hs = np.array(Hs, float)
        sv = np.linalg.svd(Hs, compute_uv=False); st_r = int((sv > 1e-10 * sv[0]).sum())
        g_r = G.graded_nilpotent_realize(cr, m=1, order=K, tol=1e-12)["dim_total"]
        HL, _, _ = lie_hankel_matrix(cr, 1, K); svl = np.linalg.svd(HL, compute_uv=False)
        R["generic"].append(dict(K=K, staircase=st_r, graded=g_r, lie=int((svl > 1e-10 * svl[0]).sum())))
    print("generic", R["generic"], flush=True)
    # ---------------------------------------------------------------- iisignature on sampled inputs against the exact method

    R["iis_check"] = {}
    for dts in (1e-3, 1e-4, 1e-5):
        ts = np.arange(0, 0.15 + dts / 2, dts)
        Ex, Wx = G.iterated_integrals(ts, U_TRAIN, 1, 6); Wx = list(Wx)
        Es, Ws = G.sampled_iterated_integrals(ts, np.array([U_TRAIN(x)[0] for x in ts]), 6)
        assert list(Ws) == Wx
        R["iis_check"][f"{dts:g}"] = {str(L): float(max(abs(Es[i, -1] - Ex[i, -1]) for i, w in enumerate(Wx) if len(w) == L) /
                                                 max(abs(Ex[i, -1]) for i, w in enumerate(Wx) if len(w) == L)) for L in range(1, 7)}
    print("iisignature check", R["iis_check"], flush=True)
    R["time"] = time.time() - t0
    json.dump(R, open(os.path.join(ROOT, "results", "n1.json"), "w"), indent=1, default=float)
