"""
fill.py -- fill every <<key>> of paper.tex from results/n1.json, compare.json, check_beyond_K.json and identifiability.json,
after checking each statement the text makes.  Writes paper_filled.tex; fails if a statement is unsupported or a key
is left unfilled.


@author: gary green
last edit 19-10-2026
"""
import os, re, json, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
R, C, B, I = J("n1.json"), J("compare.json"), J("check_beyond_K.json"), J("identifiability.json")
V = {}


def sci(x, d=1):
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) == 10:
        m, e = 1.0, e + 1
    return f"${m:.{d}f}\\times10^{{{e}}}$"


def num(x, d=3):
    return f"${x:.{d}f}$" if x < 0 else f"{x:.{d}f}"


# ---------------------------------------------------------------- example K = 2
ex = R["example"]
nz = [(k, v) for k, v in ex["coef"].items() if abs(v) > 1e-15]
V["ex_coefs"] = ", ".join(f"$c_{{({','.join(k)})}}={v:.4g}$" for k, v in nz)
lab = lambda w: "()" if w == "e" else "(" + ",".join(w) + ")"
H = np.array(ex["hankel"]); W = ex["hankel_words"]
rows = [" & " + " & ".join(f"\\scriptstyle {lab(w)}" for w in W) + r"\\ \hline"]
for i, w in enumerate(W):
    rows.append(f"\\scriptstyle {lab(w)} & " + " & ".join(("0" if abs(x) < 1e-15 else f"{x:.4g}") for x in H[i]) + r"\\")
V["ex_hankel"] = r"\begin{array}{c|" + "c" * len(W) + "}" + "\n".join(rows) + r"\end{array}"
V["ex_rank"] = ex["hankel_rank"]
assert ex["hankel_rank"] == 3 and ex["dims"] == [1, 2, 1]
F2 = np.array(ex["steps"]["2"]["Q"])[:, 0]; V["ex_F2"] = "(" + ",\\ ".join(f"{x:.4g}" for x in F2) + ")"
F1 = np.array(ex["steps"]["1"]["Q"])
V["ex_F1"] = r"\begin{pmatrix}" + r"\\".join(" & ".join(f"{x:.4g}" for x in r) for r in F1) + r"\end{pmatrix}"
V["ex_r0"], V["ex_r1"], V["ex_r2"] = ex["dims"]; V["ex_n"] = sum(ex["dims"])
assert sum(ex["dims"]) == ex["hankel_rank"] + 1                       # "one more than the minimal realisation"
fmt = lambda M: r"\begin{pmatrix}" + r"\\".join(" & ".join(("0" if abs(x) < 1e-12 else f"{x:.4f}") for x in r) for r in M) + r"\end{pmatrix}"
V["ode_mats"] = (f"A_0={fmt(ex['A0'])},\\qquad A_1={fmt(ex['A1'])},\\\\ \\lambda={fmt([[x] for x in ex['lam']])},"
                 f"\\quad \\mu={fmt([[x] for x in ex['mu']])}")
V["ode_err"] = sci(R["ode"]["max_abs_err"]); V["ode_y"] = f"{R['ode']['max_abs_y']:.3f}"
# ---------------------------------------------------------------- sanity
s = R["sanity"]
V["san_n"], V["san_K"], V["san_st"], V["san_g"], V["san_wt"] = s["n_true"], s["order"], s["staircase_dim"], s["graded_dim"], s["word_tree"]
V["san_dims"] = "(" + ",".join(map(str, s["graded_dims"])) + ")"; V["san_rb"] = sum(s["row_block_ranks"])
V["san_err"] = sci(max(s["graded_err"], s["staircase_err"]))
assert s["staircase_dim"] == s["n_true"] and s["graded_dim"] == sum(s["row_block_ranks"]) and max(s["graded_err"], s["staircase_err"]) < 1e-14
# ---------------------------------------------------------------- dimensions
rows = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rrrrrrrr}\toprule",
        r"$K$ & word tree & staircase rank & minimal & graded blocks & graded & error & nilpotency (staircase)\\\midrule"]
ratios = []
for d in R["dims"]:
    assert d["graded_dim"] == d["sum_row_block_ranks"]
    assert d["staircase_rank_float"] == d["staircase_rank_modular"] == d["staircase_dim"]
    if "true_min_dim" in d:
        assert d["true_min_dim"] == d["staircase_dim"] and d["true_min_products"] < 1e-14
    if d["K"] == 8:
        assert d["rank_mod_second_prime"] == d["staircase_rank_modular"]
    assert d["graded_nilp"] == 0.0 and d["graded_err"] < 1e-12 and d["staircase_err"] < 1e-12 and d["staircase_nilp"] < 1e-14
    assert d["sv_gap"] > 1e10
    ratios.append(d["graded_dim"] / d["staircase_dim"])
    rows.append(f"{d['K']} & {d['word_tree']} & {d['staircase_rank_float']} & {d.get('true_min_dim', '')} & ("
                + ",".join(map(str, d["graded_dims"])) + f") & {d['graded_dim']} & {sci(max(d['graded_err'], d['staircase_err']))} & {sci(d['staircase_nilp'])}\\\\")
rows.append(r"\bottomrule\end{tabular}}")
V["tab_dims"] = "\n".join(rows)
V["dim_ratio_lo"], V["dim_ratio_hi"] = f"{min(ratios):.2f}", f"{max(ratios):.2f}"
# ---------------------------------------------------------------- forward
rows = [r"\begin{tabular}{llcccc}\toprule",
        r"$K$ & input & graded $-$ true & staircase $-$ true & graded $-$ series & staircase $-$ series\\\midrule"]
for K in ("6", "8"):
    for nm, lab_ in (("train", "first"), ("test", "second")):
        f = R["forward"][K][nm]
        assert f["graded_vs_series"] < 1e-13 and f["staircase_vs_series"] < 1e-13
        rows.append(f"{K} & {lab_} & {sci(f['graded_vs_true'])} & {sci(f['staircase_vs_true'])} & {sci(f['graded_vs_series'])} & {sci(f['staircase_vs_series'])}\\\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_forward"] = "\n".join(rows)
V["fw6"] = sci(max(R["forward"]["6"][k]["graded_vs_true"] for k in ("train", "test")))
V["fw8"] = sci(max(R["forward"]["8"][k]["graded_vs_true"] for k in ("train", "test")))
# ---------------------------------------------------------------- breakdown
bk = R["breakdown"]; Ts = sorted({b["T"] for b in bk}); As = sorted({b["amp"] for b in bk})
rows = [r"\begin{tabular}{l" + "c" * len(As) + r"}\toprule", "horizon & " + " & ".join(f"$A={a:g}$" for a in As) + r"\\\midrule"]
for T in Ts:
    rows.append(f"{T:g}\\,s & " + " & ".join(num(next(b["r2"] for b in bk if b["T"] == T and b["amp"] == a), 4) for a in As) + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_breakdown"] = "\n".join(rows)
r2 = {(b["T"], b["amp"]): b["r2"] for b in bk}
assert r2[(3.0, 0.3)] < r2[(0.15, 3.0)]                      # time matters more than amplitude
V["div_true"] = f"{R['divergence']['max_abs_true']:.2f}"; V["div_real"] = f"{R['divergence']['max_abs_realised']:.0f}"
assert R["divergence"]["max_abs_realised"] > 1000 > 10 * R["divergence"]["max_abs_true"]
# ---------------------------------------------------------------- regression
S = C["settings"]
V["n_trials"], V["n_train"], V["n_val"] = S["N_TRIALS"], S["TRAIN"], S["VAL"]; V["n_test"] = S["N_TRIALS"] - S["TRAIN"] - S["VAL"]
V["sd_u"], V["n_seeds"] = S["SD_U"], S["SEEDS"]
runs = [r for r in C["runs"] if r["source"] == "fit"]
sel = lambda **k: [r for r in runs if all(r[a] == b for a, b in k.items())]
V["r2_series_6_1"] = f"{np.median([r['series_r2'] for r in sel(K=6, T=0.15, method='graded', tau=1e-6, noise=0.01)]):.4f}"
V["r2_series_6_10"] = f"{np.median([r['series_r2'] for r in sel(K=6, T=0.15, method='graded', tau=1e-6, noise=0.1)]):.3f}"
meta = C["runs_meta_6_0.15"]
# ---------------------------------------------------------------- identifiability
run = lambda K, T, sd, n=12: next(r for r in I["runs"] if r["K"] == K and r["T"] == T and r["sd_u"] == sd and r["n_train"] == n)
i4, i6, i8 = run(4, 0.15, 0.3), run(6, 0.15, 0.3), run(8, 0.15, 0.3)
assert i4["rank"] == i4["n_words"] and i6["rank"] == i6["n_words"] and i8["rank"] < i8["n_words"]
V["ncol_4"], V["ncol_6"], V["ncol_8"], V["rank_8"] = i4["n_words"], i6["n_words"], i8["n_words"], i8["rank"]
V["self_4"] = sci(max(i4["self_err"].values())); V["self_6"] = sci(max(i6["self_err"].values()))
i8l = run(8, 0.6, 0.3)
assert i8l["rank"] == i8l["n_words"] and i8["cond_scaled"] > 1e16
i848 = run(8, 0.15, 0.3, 48); i648 = run(6, 0.15, 0.3, 48)
assert i848["rank"] == i848["n_words"] and i8["rank_exact_features"] < i8["n_words"]
V["cond_8_long"] = sci(i8l["cond_scaled"]); assert i8l["rank"] == i8l["n_words"]
V["cond_8_48"] = sci(i848["cond_scaled"]); V["cond_6_48"] = sci(i648["cond_scaled"])
assert i648["cond_scaled"] < i6["cond_scaled"] / 10
assert max(i8l["self_err"].values()) > 1
assert max(i4["self_err"].values()) < 1e-6 and max(i6["self_err"].values()) < 1e-2 * i6["exact_max"]["6"]
V["cond_4"] = sci(i4["cond_scaled"]); V["cond_6"] = sci(i6["cond_scaled"]); V["cond_8"] = sci(i8["cond_scaled"])
assert abs(i6["cond_scaled"] / np.median([r["cond_scaled"] for r in sel(K=6, T=0.15, method="graded", tau=1e-6)]) - 1) < 1e-6
n6 = i6["noise_0.01"]
rows = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rcccccc}\toprule",
        r"length & largest exact & series data & no noise, exact & no noise, sampled & predicted s.e. & observed s.d.\\\midrule"]
for L in range(1, 7):
    L_ = str(L)
    rows.append(f"{L} & {i6['exact_max'][L_]:.3g} & {sci(i6['self_err'][L_])} & {sci(i6['noiseless_err_exact_features'][L_])} & {sci(i6['noiseless_err'][L_])} & {sci(n6['se_pred'][L_])} & {sci(n6['se_obs'][L_])}" + r"\\")
    assert 0.6 < n6["se_obs"][L_] / n6["se_pred"][L_] < 1.4                    # predicted and observed agree
rows.append(r"\bottomrule\end{tabular}}")
V["tab_coefs"] = "\n".join(rows)
gr = [n6["se_pred"][str(L + 1)] / n6["se_pred"][str(L)] for L in range(1, 6)]
assert all(100 < g < 1000 for g in gr)
sr = n6["se_ratio"]; V["ser_lo"], V["ser_hi"], V["ser_med"] = f"{sr['lo']:.2f}", f"{sr['hi']:.2f}", f"{sr['median']:.2f}"
assert 0.6 < sr["lo"] and sr["hi"] < 1.5 and 0.9 < sr["median"] < 1.1
V["noiseless_6x"] = f"{i6['noiseless_err_exact_features']['6']:.1f}"
assert i6["noiseless_err"]["6"] > 1000 * i6["noiseless_err_exact_features"]["6"]
ce_ = i6["column_sampling_err"]; V["colerr_1"], V["colerr_2"], V["colerr_6"] = sci(ce_["1"]), sci(ce_["2"]), sci(ce_["6"])
V["colerr_8"] = sci(i8["column_sampling_err"]["8"]); V["rank_8x"] = i8["rank_exact_features"]; V["cond_8x"] = sci(i8["cond_exact_features"])
V["noiseless_6"] = sci(i6['noiseless_err']['6']); V["exact_6"] = f"{i6['exact_max']['6']:.1f}"
assert i6["noiseless_err"]["1"] < 1e-4 and i6["noiseless_err"]["6"] > i6["exact_max"]["6"]
rows = [r"\begin{tabular}{ccccccc}\toprule",
        r"window & input s.d. & training trials & condition number & s.e.\ length 2 & s.e.\ length 4 & s.e.\ length 6\\\midrule"]
for T, sd, n_ in ((0.15, 0.3, 12), (0.3, 0.3, 12), (0.6, 0.3, 12), (0.15, 1.0, 12), (0.15, 3.0, 12), (0.15, 0.3, 48)):
    r = run(6, T, sd, n_); se = r["noise_0.01"]["se_pred"]
    rows.append(f"{T:g}" + r"\,s" + f" & {sd:g} & {n_} & {sci(r['cond_scaled'])} & {sci(se['2'])} & {sci(se['4'])} & {sci(se['6'])}" + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_excite"] = "\n".join(rows)
V["se6_base"] = sci(n6["se_pred"]["6"]); V["se6_long"] = sci(run(6, 0.6, 0.3)["noise_0.01"]["se_pred"]["6"])
V["se6_amp"] = sci(run(6, 0.15, 3.0)["noise_0.01"]["se_pred"]["6"]); V["se6_48"] = sci(i648["noise_0.01"]["se_pred"]["6"])
assert run(6, 0.6, 0.3)["noise_0.01"]["se_pred"]["6"] < n6["se_pred"]["6"] / 100 and run(6, 0.15, 3.0)["noise_0.01"]["se_pred"]["6"] < n6["se_pred"]["6"] / 100
assert all(run(6, T, sd)["noise_0.01"]["se_pred"]["6"] > 1e6 * i6["exact_max"]["6"] for T, sd in ((0.6, 0.3), (0.15, 3.0)))
V["ols_r2_6"] = num(n6["test_r2_median"], 1); V["ols_r2_4"] = num(i4["noise_0.01"]["test_r2_median"], 4)
V["ridge_r2_6"] = num(n6["ridge"]["test_r2_median"], 4)
V["off_r2_4"] = num(i4["noise_0.01"]["ridge"]["test_r2_median"], 4); V["nooff_r2_4"] = num(i4["noise_0.01"]["no_offset_ridge_test_r2_median"], 4)
V["off_r2_6"] = num(n6["ridge"]["test_r2_median"], 4); V["nooff_r2_6"] = num(n6["no_offset_ridge_test_r2_median"], 4)
assert all(r["noise_0.01"]["no_offset_ridge_test_r2_median"] < r["noise_0.01"]["ridge"]["test_r2_median"] for r in I["runs"])
assert n6["test_r2_median"] < 0 and n6["ridge"]["test_r2_median"] > 0.99
V["h1e_4"] = sci(i4["noise_0.01"]["ridge"]["h1_err_median"]); V["h1e_6"] = sci(n6["ridge"]["h1_err_median"]); V["h1max"] = f"{n6['h1_max']:.3f}"
V["h1pc_4"] = f"{100 * i4['noise_0.01']['ridge']['h1_err_median'] / n6['h1_max']:.0f}"; V["h1pc_6"] = f"{100 * n6['ridge']['h1_err_median'] / n6['h1_max']:.0f}"
V["h2e_6"] = f"{n6['ridge']['h2_err_median']:.2f}"; V["h2max"] = sci(n6["h2_max"])
V["h2e_4"] = f"{i4['noise_0.01']['ridge']['h2_err_median']:.2f}"; assert i4["noise_0.01"]["h2_max"] == 0.0 and i4["noise_0.01"]["ridge"]["h2_err_median"] > 0.01
V["n_kdraws"] = n6["n_kernel_draws"]
V["o2part"] = sci(i6["order2_part_max"]); assert i6["order2_part_max"] < 1e-6
ak = i6["allkernels_fit_0.01"]; V["allk_6"] = sci(ak["max_abs_err"]); V["allk_y"] = f"{ak['max_abs_y']:.3f}"
assert abs(ak["max_abs_err"] - ak["series_err"]) < 1e-3 * ak["series_err"]
assert n6["ridge"]["h1_err_median"] < 0.2 * n6["h1_max"] and n6["ridge"]["h2_err_median"] > 1000 * n6["h2_max"]
# Lyndon bases in the regression
V["rad_cond_6"] = sci(i6["radford"]["cond_scaled"]); V["rad_diff"] = sci(max(r["radford"]["max_pred_diff_vs_words"] for r in (i4, i6)))
assert i6["radford"]["cond_scaled"] > i6["cond_scaled"] and max(r["radford"]["max_pred_diff_vs_words"] for r in (i4, i6)) < 1e-10
V["lyn_n_6"] = i6["lyndon_only"]["n"]; V["lyn_r2_6"] = num(i6["lyndon_only"]["test_r2"], 4); V["full_r2_6"] = num(i6["noiseless_test_r2"], 4)
ls_ = i6["logsig_linear"]
assert i6["lyndon_only"]["test_r2"] < i6["noiseless_test_r2"] and n6["lyndon_only_ridge_test_r2_median"] < n6["ridge"]["test_r2_median"]
assert ls_["test_r2"] < i6["noiseless_test_r2"] and ls_["cond_scaled"] < i6["cond_scaled"] and i6["lyndon_only"]["cond_scaled"] < i6["cond_scaled"]
rows = [r"\begin{tabular}{lrccc}\toprule", r"basis & coefficients & condition number & $R^2$, no noise & $R^2$, 1\,\% noise\\\midrule"]
rn = i6["ridge_noiseless"]
rows.append(f"all words & {i6['n_words']} & {sci(i6['cond_scaled'])} & {num(rn['full'], 4)} & {num(n6['ridge']['test_r2_median'], 4)}" + r"\\")
rows.append(f"Lyndon products $R_w$ & {i6['n_words']} & {sci(i6['radford']['cond_scaled'])} & same model & same model" + r"\\")
rows.append(f"Lyndon words only & {i6['lyndon_only']['n']} & {sci(i6['lyndon_only']['cond_scaled'])} & {num(rn['lyndon_only'], 4)} & {num(n6['lyndon_only_ridge_test_r2_median'], 4)}" + r"\\")
rows.append(f"log-signature, linear & {ls_['n']} & {sci(ls_['cond_scaled'])} & {num(rn['logsig'], 4)} & {num(n6['logsig_ridge_test_r2_median'], 4)}" + r"\\")
assert n6["logsig_ridge_test_r2_median"] < n6["ridge"]["test_r2_median"] and max(rn["lyndon_only"], rn["logsig"]) < rn["full"] - 0.001
rows.append(r"\bottomrule\end{tabular}")
V["tab_lynreg"] = "\n".join(rows)
V["lie_fit_4"] = i4["lie_fit_0"]["rank"]; V["lie_fit_6"] = i6["lie_fit_0"]["rank"]
assert all(r[k]["rank"] == r[k.replace("0.01", "0")]["rank"] for r in (i4, i6) for k in ("lie_fit_0.01",))
gen = {d["K"]: d for d in R["generic"]}
assert i4["lie_fit_0"]["rank"] == gen[4]["lie"] and i6["lie_fit_0"]["rank"] == gen[6]["lie"] and min(i4["lie_fit_0"]["gap"], i6["lie_fit_0"]["gap"]) > 1e10
# Lie-Hankel
LY = R["lyndon"]
rows = [r"\begin{tabular}{rrrrr}\toprule", r"$K$ & Lyndon rows & columns & Lie--Hankel rank & staircase rank\\\midrule"]
for d in LY["truncated"]:
    assert d["gap"] > 1e10 and (d["lie_rank"] < d["staircase_rank"])
    rows.append(f"{d['K']} & {d['n_rows']} & {d['n_cols']} & {d['lie_rank']} & {d['staircase_rank']}" + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_lie"] = "\n".join(rows)
assert all(d["lie_rank"] == 2 and d["gap"] > 1e10 for d in LY["genuine"])
V["lie_maxlen"] = 2 * max(d["depth"] for d in LY["genuine"])
rows = [r"\begin{tabular}{rcccccc}\toprule", r" & \multicolumn{3}{c}{largest possible} & \multicolumn{3}{c}{exact Duffing}\\", r"$K$ & staircase & graded & Lie--Hankel & staircase & graded & Lie--Hankel\\\midrule"]
for d in R["generic"]:
    dd = next(x for x in R["dims"] if x["K"] == d["K"]); ll = next(x for x in LY["truncated"] if x["K"] == d["K"])
    assert dd["staircase_dim"] <= d["staircase"] and dd["graded_dim"] <= d["graded"] and ll["lie_rank"] <= d["lie"]
    rows.append(f"{d['K']} & {d['staircase']} & {d['graded']} & {d['lie']} & {dd['staircase_dim']} & {dd['graded_dim']} & {ll['lie_rank']}" + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_generic"] = "\n".join(rows)
ic = R["iis_check"]
V["iis_1ms"] = sci(max(ic["0.001"].values())); V["iis_01ms"] = sci(max(ic["0.0001"].values()))
assert all(70 < ic["0.001"][L] / ic["0.0001"][L] < 130 for L in ("2", "3", "4", "5", "6"))
acc = R["integral_accuracy"]
V["acc_closed"] = sci(max(max(a["input_only"], a["drift_only"]) for a in acc.values())); V["acc_all"] = sci(max(a["all_vs_tighter"] for a in acc.values()))
assert max(max(a["input_only"], a["drift_only"]) for a in acc.values()) < 1e-11 and max(a["all_vs_tighter"] for a in acc.values()) < 1e-11
V["r2_other_6"] = num(meta["r2_other_x0"], 3)
V["r2_same_6"] = num(sel(K=6, T=0.15, method="graded", tau=1e-6, noise=0.0)[0]["series_r2"], 4)
assert meta["r2_other_x0"] < 0.9
md = max(abs(r["test_r2"] - r["series_r2"]) for r in runs if r["tau"] == 1e-6 and np.isfinite(r["test_r2"]))
V["maxdiff_small"] = sci(md); assert md < 1e-4
mdg = max(abs(r["test_r2"] - r["series_r2"]) for r in runs if r["tau"] == 1e-6 and r["method"] == "graded" and np.isfinite(r["test_r2"]))
V["maxdiff_g"] = sci(mdg); assert mdg < 1e-8 < md
gd6 = np.median([r["dim"] for r in sel(K=6, T=0.15, method="graded", tau=1e-6, noise=0.1)])
sd6 = np.median([r["dim"] for r in sel(K=6, T=0.15, method="staircase", tau=1e-6, noise=0.1)])
V["g_dim_6"], V["st_dim_6"] = int(gd6), int(sd6); assert sd6 < gd6
gd4 = np.median([r["dim"] for r in sel(K=4, T=0.15, method="graded", tau=1e-6, noise=0.1)])
sd4 = np.median([r["dim"] for r in sel(K=4, T=0.15, method="staircase", tau=1e-6, noise=0.1)])
V["g_dim_4"], V["st_dim_4"] = int(gd4), int(sd4)
gen = {d["K"]: d for d in R["generic"]}
fs = [r for r in runs if r["tau"] == 1e-6 and r["T"] == 0.15]
nb = sum(r["dim"] != gen[r["K"]][r["method"]] for r in fs); V["n_below"], V["n_fitreal"] = nb, len(fs)
assert nb == 1 and all(r["dim"] <= gen[r["K"]][r["method"]] for r in fs)      # fitted sizes = largest possible
# reduction at tau = 1e-2: staircase smaller at both K, graded smaller at K = 6 only
for K_ in (4, 6):
    assert np.median([r["dim"] for r in sel(K=K_, T=0.15, method="staircase", tau=1e-2, noise=0.1)]) < gen[K_]["staircase"]
assert np.median([r["dim"] for r in sel(K=6, T=0.15, method="graded", tau=1e-2, noise=0.1)]) < 26
assert all(np.median([r["dim"] for r in sel(K=4, T=0.15, method="graded", tau=1e-2, noise=nzv)]) == 11 for nzv in (0.01, 0.03, 0.1))
for nzv in (0.01, 0.03, 0.1):                                        # staircase at least as accurate when reduced, K = 6
    a_ = np.nanmedian([r["test_r2"] for r in sel(K=6, T=0.15, method="staircase", tau=1e-2, noise=nzv)])
    b_ = np.nanmedian([r["test_r2"] for r in sel(K=6, T=0.15, method="graded", tau=1e-2, noise=nzv)])
    assert a_ >= b_, (nzv, a_, b_)
rows = [r"\begin{tabular}{rcrcccc}\toprule", r"$K$ & $\tau$ & noise & graded $R^2$ & graded states & staircase $R^2$ & staircase states\\\midrule"]
for K in (4, 6):
    for tau in (1e-6, 1e-2):
        for nzv in (0.01, 0.03, 0.1):
            cell = []
            for m in ("graded", "staircase"):
                rr = sel(K=K, T=0.15, method=m, tau=tau, noise=nzv)
                v = np.array([r["test_r2"] for r in rr])
                cell.append(f"{np.nanmedian(v):.4f} ({np.nanpercentile(v,5):.3f}-{np.nanpercentile(v,95):.4f}) & {np.nanmedian([r['dim'] for r in rr]):.0f}")
            rows.append(f"{K} & ${'10^{-6}' if tau == 1e-6 else '10^{-2}'}$ & {nzv*100:g}\\,\\% & " + " & ".join(cell) + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_regression"] = "\n".join(rows)
# beyond-K and leakage
for r in B:
    if r["method"] == "graded":
        assert all(v == 0.0 for v in r["beyond"].values()) and r["nilp"] == 0.0
        assert all(r["solution"][k]["rel_diff"] < 1e-9 for k in r["solution"])
    if r["method"] == "staircase" and r["tau"] == 1e-6:
        assert r["nilp"] < 1e-13
srb = max(max(r["beyond"].values()) / r["coef_max"] for r in B if r["method"] == "staircase" and r["tau"] == 1e-6)
V["st_rel_beyond"] = sci(srb); assert srb < 1e-13
st = [r for r in B if r["method"] == "staircase" and r["K"] == 6 and r["tau"] == 1e-2 and r["source"] == "fit 0.1"][0]
sys.path.insert(0, os.path.join(ROOT, "examples"))
from duffing_symbolic import duffing_coeffs as _dc
from math import factorial as _fa
_c6 = _dc(order=6, q1_0=0.5, q2_0=0.3)
_ref = max(abs(v) * 0.15 ** len(w) / _fa(len(w)) for w, v in _c6.items() if len(w) >= 1)
V["st_beyond7_rel"] = sci(st["beyond"]["7"] * 0.15 ** 7 / _fa(7) / _ref); assert st["beyond"]["7"] * 0.15 ** 7 / _fa(7) / _ref < 1e-3
V["st_beyond7"] = f"{st['beyond']['7']:.0f}"; V["st_leak1"] = sci(st["solution"]["1.0"]["max_abs_diff"]); V["st_leak3"] = f"{st['solution']['3.0']['max_abs_diff']:.0f}"
V["st_leak015"] = sci(st["solution"]["0.15"]["max_abs_diff"]); V["ser_1"] = f"{st['solution']['1.0']['max_abs_y']:.0f}"; V["ser_3"] = sci(st["solution"]["3.0"]["max_abs_y"])
assert st["solution"]["0.15"]["max_abs_diff"] < 1e-5 and st["solution"]["1.0"]["max_abs_y"] > 100 * 1.27
assert st["nilp"] > 1e-8
# ---------------------------------------------------------------- kernels
k = R["kernels"]
V["k_n0"], V["k_n1"], V["k_n2"], V["k_n3"] = (k["counts"][str(n)] for n in range(4)); assert k["counts"]["4"] == 0
V["k_diff"] = sci(max(v["diff"] for v in k["contributions"].values()))
V["k_c2"] = sci(k["contributions"]["2"]["from_iterated_integrals"]); V["k_c3"] = sci(k["contributions"]["3"]["from_iterated_integrals"])
assert max(v["rel_diff"] for v in k["cube"].values()) < 1e-12
assert all(d["staircase"][0] == d["staircase"][1] and d["graded"][0] == d["graded"][1] for d in R["offset"])
F = k["full"]
for K_ in ("4", "6", "8"):
    V[f"kfull_{K_}"] = sci(F[K_]["max_abs_err"])
V["kfull_y"] = f"{F['8']['max_abs_y']:.3f}"
assert F["8"]["max_abs_err"] < F["6"]["max_abs_err"] < F["4"]["max_abs_err"]
assert all(F[K_]["max_abs_err"] < 1e-3 * F[K_]["max_abs_y"] for K_ in F)

# ---------------------------------------------------------------- tutorial: driftless systems and the fitting study
DL, FS = J("driftless.json"), J("fit_study.json")
dd = DL["dduffing"]
for d in dd:
    assert d["rank_float"] == d["rank_modular"] == d["staircase"] and d["graded_err"] < 1e-12 and d["staircase_err"] < 1e-12
    assert d["graded_nilp"] == 0.0 and d["staircase_nilp"] < 1e-13
V["dd_st_2"], V["dd_st_8"] = dd[0]["staircase"], dd[-1]["staircase"]; V["dd_g_2"], V["dd_g_8"] = dd[0]["graded"], dd[-1]["graded"]
assert all(dd[i + 1]["staircase"] == dd[i]["staircase"] + 1 for i in range(len(dd) - 1))
rows = [r"\resizebox{\textwidth}{!}{\begin{tabular}{rrrrrrr}\toprule", r"$K$ & word tree & staircase rank & graded blocks & graded & error & nilpotency (staircase)\\\midrule"]
for d in dd:
    rows.append(f"{d['K']} & {2 ** (d['K'] + 1) - 1} & {d['staircase']} & (" + ",".join(map(str, d["graded_dims"])) + f") & {d['graded']} & {sci(max(d['graded_err'], d['staircase_err']))} & {sci(d['staircase_nilp'])}" + r"\\")
rows.append(r"\bottomrule\end{tabular}}")
V["tab_ddims"] = "\n".join(rows)
rows = [r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccccc}\toprule", r"input & duration & excursion & area & largest output & series $-$ true & graded $-$ series & staircase $-$ series\\\midrule"]
for d in DL["forward"]:
    if d["T"] < 10.0:
        continue
    assert d["graded_vs_series"] < 1e-8 and d["staircase_vs_series"] < 1e-8
    lab = (f"sines, $A={d['amp']:g}$" + (f", $b={d['bias']:g}$" if d["bias"] else "")) if d["kind"] == "sines" else "circle"
    rows.append(f"{lab} & {d['T']:g}\\,s & {d['path_max']:.3f} & {sci(d['area_max'])} & {d['max_abs_y']:.3g} & {sci(d['series_vs_true'])} & {sci(d['graded_vs_series'])} & {sci(d['staircase_vs_series'])}" + r"\\")
rows.append(r"\bottomrule\end{tabular}}")
V["tab_dforward"] = "\n".join(rows)
fw = {(d["amp"], d["bias"]): d for d in DL["forward"] if d["T"] == 10.0 and d["kind"] == "sines"}
circ = {d["T"]: d for d in DL["forward"] if d["kind"] == "circle"}
assert circ[100.0]["path_max"] < 0.41 and circ[100.0]["area_max"] > 9 * circ[10.0]["area_max"] and circ[100.0]["series_vs_true"] > circ[10.0]["series_vs_true"]
assert fw[(1.0, 0.0)]["series_vs_true"] < 1e-12 < fw[(100.0, 0.0)]["series_vs_true"] and fw[(1.0, 0.3)]["series_vs_true"] > 1e-3
hb = DL["heisenberg"]; assert hb["nonzero_lengths"] == [1, 2]
V["hb_st"], V["hb_g"] = hb["sizes"]["staircase"], hb["sizes"]["graded"]
V["hb_err"] = sci(max(max(r["graded_err"], r["staircase_err"]) for r in hb["runs"])); V["hb_y"] = f"{max(r['max_abs_y'] for r in hb['runs']):.1f}"
assert all(max(r["graded_err"], r["staircase_err"]) < 1e-10 * max(1.0, r["max_abs_y"]) for r in hb["runs"])
V["hb_lie"] = hb["lie_rank"]; assert hb["lie_rank"] == 3
K2 = DL["dduffing_K2"]
lab2 = lambda k: "()" if k == "e" else "(" + ",".join(k) + ")"
V["dd2_coefs"] = ", ".join(f"$c_{{{lab2(k)}}}={v:.4g}$" for k, v in K2["coef"].items() if abs(v) > 1e-15)
fmt4 = lambda M: r"\begin{pmatrix}" + r"\\".join(" & ".join(("0" if abs(x) < 1e-12 else f"{x:.4f}") for x in r) for r in M) + r"\end{pmatrix}"
V["dd2_mats"] = (f"A_1={fmt4(K2['A0'])},\\qquad A_2={fmt4(K2['A1'])},\\\\ \\lambda={fmt4([[x] for x in K2['lam']])},\\quad \\mu={fmt4([[x] for x in K2['mu']])}")
A_ = [np.array(K2["A0"]), np.array(K2["A1"])]; lam0 = K2["lam"][0]; n_ = len(K2["lam"])
lines = []
for i in range(1, n_):
    terms = []
    for a in range(2):
        for j in range(n_):
            v = A_[a][i, j]
            if abs(v) < 1e-12:
                continue
            coefv = v * lam0 if j == 0 else v
            terms.append(f"{coefv:+.4f}\\," + ("" if j == 0 else f"z_{j}\\,") + f"u_{a + 1}(t)")
    lines.append(f"\\dot z_{i}=" + " ".join(terms).lstrip("+"))
mu_ = K2["mu"]
lines.append("y(t)-y(0)=" + " ".join(f"{mu_[i]:+.4f}\\,z_{i}" for i in range(1, n_) if abs(mu_[i]) > 1e-12).lstrip("+"))
V["dd2_odes"] = r"\\".join(lines)
sv6 = DL["dduffing_sv6"]; V["dd_sv"] = ", ".join(f"{x:.3g}" for x in sv6["sv"][:7])
V["dd_r_1e2"], V["dd_r_1e6"] = sv6["rank_tau_1e2"], sv6["rank_tau_1e6"]; assert sv6["rank_tau_1e2"] < sv6["rank_tau_1e6"] == 7
assert all(v == 2 for v in DL["dduffing_nonzero_by_length"].values())
V["dd_lie"] = DL["dduffing_lie_genuine"][-1]["rank"]; assert all(d["rank"] == 2 for d in DL["dduffing_lie_genuine"])
V["dd_lie_trunc"] = ", ".join(str(d["lie_truncated"]) for d in dd); V["dd_st_list"] = ", ".join(str(d["staircase"]) for d in dd)
f4 = DL["four"]
rows = [r"\begin{tabular}{rrrrr}\toprule", r"$K$ & word tree & staircase rank & graded blocks & graded\\\midrule"]
for d in f4:
    assert d["graded_err"] < 1e-12 and d["staircase_err"] < 1e-12 and d["graded_nilp"] == 0.0 and d["staircase_nilp"] < 1e-13
    if "rank_modular" in d:
        assert d["rank_modular"] == d["staircase"]
    rows.append(f"{d['K']} & {d['word_tree']} & {d['staircase']} & (" + ",".join(map(str, d["graded_dims"])) + f") & {d['graded']}" + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_four"] = "\n".join(rows)
V["f4_wt_6"], V["f4_st_6"], V["f4_g_6"] = f4[-1]["word_tree"], f4[-1]["staircase"], f4[-1]["graded"]; assert f4[-1]["K"] == 6
k7 = DL["four_K7_graded"]; V["f4_g_7"], V["f4_wt_7"] = k7["graded"], k7["word_tree"]; assert k7["err"] < 1e-11
V["f4_nz6"] = DL["four_nonzero_by_length"]["6"]
fsr = {(r["system"], r["K"]): r for r in FS["runs"]}
noises = ["0", "0.001", "0.01", "0.03", "0.1"]
rows = [r"\begin{tabular}{llccccc}\toprule", r"system & $K$ & no noise & 0.1\,\% & 1\,\% & 3\,\% & 10\,\%\\\midrule"]
for (nm, lab) in (("duffing", "driven"), ("dduffing", "driftless")):
    for K_ in (4, 6):
        r = fsr[(nm, K_)]
        rows.append(f"{lab} & {K_} & " + " & ".join(num(r["noise"][n]["r2_median"], 4) for n in noises) + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_fitr2"] = "\n".join(rows)
V["fs_seeds"] = 20
V["fo_d4"] = num(fsr[("duffing", 4)]["noise"]["0.01"]["r2_median"], 4); V["fno_d4"] = num(fsr[("duffing", 4)]["noise"]["0.01"]["r2_no_offset_median"], 4)
V["fo_d6"] = num(fsr[("duffing", 6)]["noise"]["0.01"]["r2_median"], 4); V["fno_d6"] = num(fsr[("duffing", 6)]["noise"]["0.01"]["r2_no_offset_median"], 4)
V["fo_dd6"] = num(fsr[("dduffing", 6)]["noise"]["0.01"]["r2_median"], 4); V["fno_dd6"] = num(fsr[("dduffing", 6)]["noise"]["0.01"]["r2_no_offset_median"], 4)
assert all(fsr[k]["noise"]["0.01"]["r2_no_offset_median"] < fsr[k]["noise"]["0.01"]["r2_median"] for k in fsr)
V["fols_d6"] = num(fsr[("duffing", 6)]["noise"]["0.01"]["r2_ols_median"], 0); V["fols_dd6"] = num(fsr[("dduffing", 6)]["noise"]["0.01"]["r2_ols_median"], 1)
assert fsr[("duffing", 6)]["noise"]["0.01"]["r2_ols_median"] < 0 and fsr[("dduffing", 6)]["noise"]["0.01"]["r2_ols_median"] < 0
V["fs_nreal"] = sum(r["noise"][n]["sizes"]["n"] for r in fsr.values() for n in noises)
assert all(fsr[k]["noise"]["0"]["r2_median"] > 0.99995 for k in fsr)
V["fr_d6_3"] = num(fsr[("duffing", 6)]["noise"]["0.03"]["r2_median"], 3)
V["fr_dd6_1"] = num(fsr[("dduffing", 6)]["noise"]["0.01"]["r2_median"], 3); V["fr_dd6_10"] = num(fsr[("dduffing", 6)]["noise"]["0.1"]["r2_median"], 2)
assert fsr[("dduffing", 6)]["noise"]["0.01"]["r2_median"] < fsr[("duffing", 6)]["noise"]["0.01"]["r2_median"]
rows = [r"\begin{tabular}{rcccc}\toprule", r" & \multicolumn{2}{c}{driven} & \multicolumn{2}{c}{driftless}\\", r"length & no noise & 1\,\% & no noise & 1\,\%\\\midrule"]
for L in range(1, 7):
    L_ = str(L)
    rows.append(f"{L} & " + " & ".join(sci(fsr[(nm, 6)]["noise"][n]["coef_rel_err"][L_]) for nm in ("duffing", "dduffing") for n in ("0", "0.01")) + r"\\")
rows.append(r"\bottomrule\end{tabular}")
V["tab_coefrel"] = "\n".join(rows)
for nm in ("duffing", "dduffing"):
    ce1 = fsr[(nm, 6)]["noise"]["0.01"]["coef_rel_err"]
    assert all(ce1[str(L)] > 0.1 for L in range(3, 7)) and fsr[(nm, 6)]["noise"]["0"]["coef_rel_err"]["1"] < 1e-3
for k, r in fsr.items():
    for n in noises:
        assert r["noise"][n]["sizes"]["staircase"] == [gen[k[1]]["staircase"]] and r["noise"][n]["sizes"]["graded"] == [gen[k[1]]["graded"]], (k, n)

src = open(os.path.join(ROOT, "latex", sys.argv[1] if len(sys.argv) > 1 else "tutorial.tex")).read()
keys = set(re.findall(r"<<(\w+)>>", src)); missing = keys - set(V); assert not missing, missing
out = re.sub(r"<<(\w+)>>", lambda m: str(V[m.group(1)]), src)
open(os.path.join(ROOT, "latex", (sys.argv[1] if len(sys.argv) > 1 else "tutorial.tex").replace(".tex", "_filled.tex")), "w").write(out)
print("filled", len(keys), "keys")
