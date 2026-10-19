#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cfs_tutorial
============
Bilinear realisations of truncated Chen-Fliess series and path signatures: iterated integrals 
(exact and by iisignature), 
the staircase and graded realisations, Volterra kernels, Lyndon words and the Lie-Hankel matrix.

@author: gary green
last edit 19-10-2026
"""
from .signature import signature_matrix, sampled_iterated_integrals
from .realization import graded_nilpotent_realize, verify_exact, verify_nilpotent, simulate_bilinear
from .word_tree import build_word_tree_system, verify_nilpotent as verify_nilpotent_word_tree
from .volterra_kernel import extract_kernel_terms, evaluate_kernel, kernel_contribution, simplex_quadrature, symmetric_kernel, cube_contribution
from .true_minimal import (full_hankel_matrix, staircase_hankel_matrix, exact_rank_modular, true_minimal_realize,
                            verify_true_minimal, products_vanish)
from .hankel import staircase_realize, coefficient, max_coef_error, nilpotency_residual, beyond_K
from .integrate import iterated_integrals, simulate_realisation, simulate_realisation_sampled
from .lyndon import is_lyndon, lyndon_words, bracket_polynomial, lie_hankel_matrix, lyndon_factorisation, radford_features, radford_matrix

__all__ = [
    "signature_matrix",
    "graded_nilpotent_realize", "verify_exact", "verify_nilpotent", "simulate_bilinear",
    "build_word_tree_system", "verify_nilpotent_word_tree",
    "extract_kernel_terms", "evaluate_kernel", "kernel_contribution", "simplex_quadrature",
    "full_hankel_matrix", "staircase_hankel_matrix", "exact_rank_modular", "true_minimal_realize",
    "verify_true_minimal", "products_vanish", "staircase_realize", "coefficient", "max_coef_error",
    "nilpotency_residual", "beyond_K", "iterated_integrals", "simulate_realisation",
]
