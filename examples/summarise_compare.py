"""summarise_compare.py -- tables from results/compare.json (medians and 5-95 % ranges over noise seeds).


@author: gary green
last edit 19-10-2026
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results", "compare.json")))
runs = d["runs"]
q = lambda v, p: float(np.nanpercentile(v, p))
for K, T in d["settings"]["CONFIGS"]:
    print(f"\n===== K={K}  T={T} s  (truncation floor R2 {d['runs_meta_' + f'{K}_{T}']['r2_trunc']:.6f})")
    fits = [r for r in runs if r["K"] == K and r["T"] == T and r["source"] == "fit"]
    for nz in d["settings"]["NOISES"]:
        rr = [r for r in fits if r["noise"] == nz and r["method"] == "graded" and r["tau"] == 1e-6]
        print(f"  noise {nz:g}: ridge chosen {sorted(set(r['ridge'] for r in rr))}  cond(raw) {q([r['cond_raw'] for r in rr],50):.1e}  "
              f"cond(scaled) {q([r['cond_scaled'] for r in rr],50):.1e}  |c-c_exact|max med {q([r['coef_err_vs_exact'] for r in rr],50):.2e}  "
              f"series R2 med {q([r['series_r2'] for r in rr],50):.5f} [{q([r['series_r2'] for r in rr],5):.5f}, {q([r['series_r2'] for r in rr],95):.5f}]")
    print(f"  {'noise':>6} {'tau':>7} {'method':>7} {'dim med':>8} {'R2 med':>10} {'R2 5%':>10} {'R2 95%':>10} {'R2-series max|d|':>17} {'nilp med':>9} {'beyond med':>10} {'recon med':>9}")
    for nz in d["settings"]["NOISES"]:
        for tau in d["settings"]["TAUS"]:
            for m in ("graded", "staircase"):
                rr = [r for r in fits if r["noise"] == nz and r["tau"] == tau and r["method"] == m]
                if not rr:
                    continue
                dd = [abs(r["test_r2"] - r["series_r2"]) for r in rr]
                print(f"  {nz:6g} {tau:7.0e} {m:>7} {q([r['dim'] for r in rr],50):8.0f} {q([r['test_r2'] for r in rr],50):10.5f} "
                      f"{q([r['test_r2'] for r in rr],5):10.4f} {q([r['test_r2'] for r in rr],95):10.5f} {np.nanmax(dd):17.2e} "
                      f"{q([r['nilp'] for r in rr],50):9.1e} {q([r['beyond'] for r in rr],50):10.1e} {q([r['recon_err'] for r in rr],50):9.1e}")
    ex = [r for r in runs if r["K"] == K and r["T"] == T and r["source"] == "exact"]
    print("  exact coefficients:", "; ".join(f"{r['method']} tau {r['tau']:.0e} dim {r['dim']} R2 {r['test_r2']:.6f} nilp {r['nilp']:.1e}" for r in ex))
