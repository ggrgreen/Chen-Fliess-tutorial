# cfs-tutorial

Code for the tutorial *Bilinear Realisations of Truncated Chen–Fliess Series and Path Signatures*
by Gary Green, University of York. The tutorial itself is [`latex/tutorial_filled.pdf`](latex/tutorial_filled.pdf).

The tutorial covers series with a letter for time (a drift) and series without one (path signatures). It shows how to:

- compute iterated integrals from formulas and from samples;
- set up the staircase Hankel matrix of a truncated series;
- build minimal (staircase) and graded realisations;
- fit a series to sampled data with a free offset for each window;
- derive Volterra kernels from the coefficients;
- build the Lie–Hankel matrix from Lyndon words.

The test systems are the driven Duffing oscillator, a driftless Duffing system, the Heisenberg integrator and a
four-channel system.

## Install

    git clone <this repository>
    cd cfs-tutorial
    python3 -m pip install -e .        # numpy, scipy, sympy, matplotlib, iisignature

## Using the code

Letters are numbered from 0:

- **With a drift:** letter 0 is time, and letters 1..m are the m inputs.
- **Without a drift (n channels):** the channels written 1..n in the tutorial are letters 0..n−1. Pass
  `drift=False` to the integral and simulation functions, and use m = n − 1 (the highest letter) wherever a function
  asks for m.

A minimal session for a series with one input and a letter for time:

```python
import numpy as np
import cfs_tutorial as G

t = np.arange(0, 0.15 + 5e-4, 1e-3)               # 1 ms samples
u = 0.3 * np.sin(6 * np.pi * t)                     # sampled input
E, words = G.sampled_iterated_integrals(t, u, 4)    # iterated integrals by iisignature
# ... fit coefficients c (one per word) to y(t) - y(0), e.g. by ridge regression ...
c = np.zeros(len(words))
coef = {(): 0.0, **dict(zip(words, c))}
R = G.staircase_realize(coef, 1, 4, 1e-6)           # minimal (staircase) realisation
yhat = G.simulate_realisation_sampled(R["A"], R["lam"], R["mu"], t, u)
```

Main functions:

| function | purpose |
|---|---|
| `iterated_integrals` | exact iterated integrals for an input given as a formula (DOP853, tolerance scaled per word) |
| `sampled_iterated_integrals` | iterated integrals of sampled inputs by iisignature (cubic-spline integral of the input) |
| `staircase_hankel_matrix`, `staircase_realize` | staircase Hankel matrix of a truncated series and its minimal realisation |
| `graded_nilpotent_realize` | graded construction, nilpotent for any tolerance |
| `simulate_realisation`, `simulate_realisation_sampled` | run a bilinear realisation with a formula input, or with the sampled path |
| `extract_kernel_terms`, `symmetric_kernel`, `kernel_contribution` | Volterra kernels from the coefficients (one input with a letter for time) |
| `lyndon_words`, `bracket_polynomial`, `lie_hankel_matrix` | Lyndon words, standard bracketing and the Lie–Hankel matrix |
| `exact_rank_modular` | rank certificate for the staircase matrix |

## Rule for iterated integrals

- **Input known as a formula:** use the exact word-tree ODE (`iterated_integrals`).
- **Input known only as samples**  use iisignature on the sampled path  (`sampled_iterated_integrals`).
- **Always report** which method was used, and give iisignature's sampling error against the exact method
  (`results/n1.json`, key `iis_check`).

## Package layout (`src/cfs_tutorial`)

    integrate.py        iterated integrals on the word tree (DOP853); simulation of bilinear systems
    signature.py        iterated integrals from a formula or from samples (iisignature)
    realization.py      graded nilpotent construction
    hankel.py           staircase realisation, coefficient checks, nilpotency residual, coefficients beyond K
    true_minimal.py     staircase Hankel matrix, exact or modular rank certificate, true minimal size
    word_tree.py        unreduced word-tree realisation
    lyndon.py           Lyndon words, bracketing, Lie-Hankel matrix, Lyndon factorisation, Radford basis
    volterra_kernel.py  kernels from coefficients; Gauss-Legendre (Duffy) quadrature on the simplex

No trapezoidal integration is used anywhere.

## Reproducing the tutorial

    python3 examples/run_all.py               # results/n1.json
    python3 examples/compare_constructions.py # results/compare.json 
    python3 examples/check_beyond_K.py        # results/check_beyond_K.json
    python3 examples/identifiability.py       # results/identifiability.json
    python3 examples/driftless.py             # results/driftless.json 
    python3 examples/fit_study.py             # results/fit_study.json
    python3 examples/summarise_compare.py
    python3 examples/make_figures.py                            # latex/fig_*.pdf
    cd latex && python3 fill.py tutorial.tex && pdflatex tutorial_filled.tex && pdflatex tutorial_filled.tex

`fill.py` writes every number in the tutorial from the result files, and checks each statement the text makes.

## Example scripts (`examples/`)

    duffing_symbolic.py       exact coefficients of the driven Duffing oscillator (symbolic Lie derivatives)
    systems.py                driftless Duffing, Heisenberg and four-channel systems: coefficients and true responses
    sanity_check.py           synthetic system with a known nilpotent realisation
    run_all.py                sizes, the K = 2 system, runs within and beyond the window, kernels, Lie-Hankel ranks
    driftless.py              realisations without a letter for time, with exact coefficients
    fit_study.py              fitting both Duffing systems from sampled data
    compare_constructions.py  graded against staircase realisations of fitted series (50 noise draws)
    check_beyond_K.py         whether either realisation produces terms beyond K
    identifiability.py        trials needed, offsets, kernels from fits, Lyndon bases in the regression
    make_figures.py           figures of the tutorial

## Tests

    python3 -m pytest tests

## Licence

MIT licence; see [`LICENSE`](LICENSE). Copyright (c) 2026 Gary Green.
