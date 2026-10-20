#!/usr/bin/env python3
"""
Standalone statistics primitives: Pearson r with Fisher-z CI, Spearman rho
with bootstrap CI, Holm-Bonferroni and Benjamini-Hochberg correction.

Independent reimplementation (no imports from the original repo). Matches
the formulas used throughout the manuscript's correlation/robustness
analyses. All bootstrap resampling uses an explicit, fixed seed (passed in
by the caller; run_all.py fixes it to 42 everywhere) so results are
bit-identical across machines/runs for the same numpy version.
"""
import math

import numpy as np
from scipy import stats as scipy_stats


def pearson_with_ci(x, y, alpha=0.05, n_boot=10000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = scipy_stats.pearsonr(x, y)
    n = len(x)
    if n > 3 and abs(r) < 1:
        z = np.arctanh(r)
        se = 1 / math.sqrt(n - 3)
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)
    else:
        lo, hi = float("nan"), float("nan")
    return {"r": float(r), "p": float(p), "n": n, "ci95_fisherz": (float(lo), float(hi))}


def spearman_with_ci(x, y, alpha=0.05, n_boot=10000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho, p = scipy_stats.spearmanr(x, y)
    n = len(x)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            boots[b] = np.nan
            continue
        boots[b], _ = scipy_stats.spearmanr(bx, by)
    valid = boots[~np.isnan(boots)]
    lo, hi = (np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if len(valid) else (float("nan"), float("nan")))
    return {"rho": float(rho), "p": float(p), "n": n, "ci95_bootstrap": (float(lo), float(hi))}


def holm_bonferroni(pvals, alpha=0.05):
    """Step-down Holm correction. Returns (adjusted_p, reject) in ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist(), (adj < alpha).tolist()


def benjamini_hochberg(pvals, alpha=0.05):
    """Step-up BH-FDR correction. Returns (adjusted_p, reject) in ORIGINAL order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)[::-1]
    adj = np.empty(m)
    running_min = 1.0
    for rank, idx in enumerate(order):
        k = m - rank
        val = min(pvals[idx] * m / k, 1.0)
        running_min = min(running_min, val)
        adj[idx] = running_min
    return adj.tolist(), (adj < alpha).tolist()
