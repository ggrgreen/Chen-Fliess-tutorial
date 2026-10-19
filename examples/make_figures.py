"""
make_figures.py -- the figures of the paper, from results/n1.json and results/compare.json, into latex/.
  fig_h1.pdf          first-order kernel h_1(0.1; s) at K = 8; differences of K = 4, 6 from it; error of the sum of
                      all kernels of all orders against the true Duffing response
  fig_h2.pdf          symmetric second-order kernel h_2(t; s_1, s_2) at t = 0.1 s (K = 8), 2-D map on the whole square [0, t]^2
  fig_regression.pdf  R^2 on the test trials and number of states against noise, graded and staircase realisations

@author: gary green
last edit 19-10-2026
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import cfs_tutorial as G
from run_all import coeffs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "latex")
BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#9a9890"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

R = json.load(open(os.path.join(ROOT, "results", "n1.json")))
T_K = R["kernels"]["T"]

# ---------------------------------------------------------------- first-order kernel and the full-series check
sf = np.linspace(0, T_K, 400)
h1 = {K: G.symmetric_kernel(G.extract_kernel_terms(coeffs(K), 1, K), T_K, sf[:, None]) for K in (4, 6, 8)}
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.4))
axs[0].plot(sf, h1[8], color=BLUE, lw=1.2)
axs[0].set_xlabel("input time $s$ (s)"); axs[0].set_ylabel("$h_1(0.1; s)$, $K = 8$")
for K, col in ((4, ORANGE), (6, AQUA)):
    axs[1].semilogy(sf, np.maximum(np.abs(h1[K] - h1[8]), 1e-18), color=col, lw=1.2, label=f"K = {K}")
axs[1].set_xlabel("input time $s$ (s)"); axs[1].set_ylabel("$|h_1^{(K)} - h_1^{(8)}|$"); axs[1].legend(frameon=False, fontsize=7)
for K, col in ((4, ORANGE), (6, AQUA), (8, BLUE)):
    F = R["kernels"]["full"][str(K)]
    err = np.abs(np.array(F["from_kernels"]) - np.array(F["true"]))
    axs[2].semilogy(F["t"], np.maximum(err, 1e-17), "o-", color=col, ms=3, lw=1.2, label=f"K = {K}")
axs[2].set_xlabel("time $t$ (s)"); axs[2].set_ylabel("|all kernels $-$ true|"); axs[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_h1.pdf")); plt.close(fig)

# ---------------------------------------------------------------- second-order kernel, symmetric, whole square (2-D map)
terms2 = G.extract_kernel_terms(coeffs(8), 2, 8)
g = np.linspace(0, T_K, 201); S1, S2 = np.meshgrid(g, g, indexing="ij")
Z = G.symmetric_kernel(terms2, T_K, np.stack([S1, S2], axis=-1))
fig, ax = plt.subplots(figsize=(3.9, 3.3))
vm = np.max(np.abs(Z))
im = ax.pcolormesh(S1, S2, Z, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
ax.contour(S1, S2, Z, levels=8, colors=INK, linewidths=0.4)
ax.plot([0, T_K], [0, T_K], "--", color=INK, lw=0.8)
ax.set_aspect("equal"); ax.set_xlabel("$s_1$ (s)"); ax.set_ylabel("$s_2$ (s)")
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("$h_2(0.1; s_1, s_2)$")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_h2.pdf")); plt.close(fig)

# ---------------------------------------------------------------- regression comparison
C = json.load(open(os.path.join(ROOT, "results", "compare.json")))
runs = [r for r in C["runs"] if r["source"] == "fit"]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
noises = [n for n in C["settings"]["NOISES"] if n > 0]
for j, K in enumerate((4, 6)):
    for tau, ls in ((1e-6, "-"), (1e-2, "--")):
        for m, col, mk in (("graded", BLUE, "o"), ("staircase", ORANGE, "s")):
            rr = lambda nz: [r for r in runs if r["K"] == K and r["T"] == 0.15 and r["noise"] == nz and r["tau"] == tau and r["method"] == m]
            med = [np.nanmedian([r["test_r2"] for r in rr(nz)]) for nz in noises]
            dim = [np.nanmedian([r["dim"] for r in rr(nz)]) for nz in noises]
            lab = f"{m}, $\\tau$ = {tau:g}"
            axs[0, j].semilogx(np.array(noises) * 100, med, ls, color=col, marker=mk, ms=3, lw=1, label=lab)
            axs[1, j].semilogx(np.array(noises) * 100, dim, ls, color=col, marker=mk, ms=3, lw=1, label=lab)
    axs[0, j].set_title(f"K = {K}, window 0.15 s", fontsize=9)
    axs[1, j].set_xlabel("noise (% of response SD)")
axs[0, 0].set_ylabel("$R^2$ on test trials (median)"); axs[1, 0].set_ylabel("number of states (median)")
axs[0, 1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_regression.pdf")); plt.close(fig)
#print("figures written")
