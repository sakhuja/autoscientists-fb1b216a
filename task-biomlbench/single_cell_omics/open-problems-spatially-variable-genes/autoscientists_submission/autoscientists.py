"""
DE + CD + WarmDirichlet on 24-Signal gamma_020 Set
====================================================
Team: gamma
Experiment: exp_gamma_021

RATIONALE:
Current champion: gamma_020 (tau=0.858888)
  - 24-signal VST-shifted set + 3seeds x 100k Dirichlet + CD

gamma_019 showed DE gave +0.006 over Dirichlet on 12 signals.
This experiment applies DE to the full gamma_020 24-signal set.

METHOD:
1. Compute EXACT same 24 signals as gamma_020
2. DE: scipy.optimize.differential_evolution on 24-dim simplex
   - maxiter=300, popsize=25, mutation=(0.5,1.5), recombination=0.7, seed=42
   - Objective: maximize Kendall tau (minimize negative tau)
   - Weights projected to simplex via abs+normalize
3. CD refinement (10 cycles, 200 steps) from DE solution
4. Warm Dirichlet (100k) initialized from best so far
5. Final CD refinement
6. Report tau, save submission

Signal list (same as gamma_020):
  At shift=0.50 (single version):
    sparkx_norm, moran_ms, moran_k30, geary_k5_neg, smr_k5, smr_k12,
    smr_k20, smr_k30, getis_k10, getis_k20, dirmoran_x, dirmoran_y
  At shifts [0.25, 0.50, 1.00] (3 versions each):
    sparkx_cnt, geary_k30_neg, variogram, smr_k8

Total = 12 single + 12 shifted = 24 signals

EXPECTED: DE on 24-dim should push beyond champion 0.858888
RUNTIME: ~60-70 minutes
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import kendalltau, rankdata
from scipy.sparse import issparse, csr_matrix
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"EXP_ID = "exp_gamma_021"
AGENT_WS = Path(__file__).parent / "outputs"

SEED = 42
np.random.seed(SEED)

# VST shifts for top signals
VST_SHIFTS_TOP = [0.25, 0.50, 1.00]   # 3 versions of top signals
VST_SHIFT_DEFAULT = 0.50               # single version for other signals

# Spatial parameters
MORAN_MULTISCALE_KS = (5, 10, 20, 40)
MORAN_K30 = 30
GEARY_K5 = 5
GEARY_K30 = 30
VARIOGRAM_K_MAX = 50
VARIOGRAM_N_BINS = 8
SMR_K_VALUES = [5, 8, 12, 20, 30]
GETIS_K_VALUES = [10, 20]   # GetisG* scales
DIR_MORAN_K = 20            # DirMoran neighbor count

# DE configuration
DE_MAXITER = 300
DE_POPSIZE = 25
DE_MUTATION = (0.5, 1.5)
DE_RECOMBINATION = 0.7

# Post-DE warm Dirichlet
WARM_DIRICHLET_TRIALS = 100000

# Champion to beat
CHAMPION_TAU = 0.858888   # gamma_020

print(f"[CONFIG] EXP_ID={EXP_ID}")
print(f"[CONFIG] 24 signals from gamma_020")
print(f"[CONFIG] DE: maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, mutation={DE_MUTATION}, recombination={DE_RECOMBINATION}")
print(f"[CONFIG] Warm Dirichlet after DE: {WARM_DIRICHLET_TRIALS} trials")
print(f"[CONFIG] Champion to beat: {CHAMPION_TAU}")


# ============================================================
# Data extraction with FT-VST at a given shift
# ============================================================
def extract_both_layers_vst(adata, shift=0.5):
    """Extract normalized and counts layers, apply Freeman-Tukey VST."""
    if 'normalized' in adata.layers:
        norm = adata.layers['normalized']
    else:
        norm = adata.X
    if issparse(norm):
        norm = norm.toarray()
    norm = np.array(norm, dtype=np.float64)
    # VST for normalized (log1p space): sqrt(expm1(x) + shift)
    ft_norm = np.sqrt(np.maximum(np.expm1(norm), 0) + shift)

    if 'counts' in adata.layers:
        counts = adata.layers['counts']
    else:
        counts = adata.X
    if issparse(counts):
        counts = counts.toarray()
    counts = np.array(counts, dtype=np.float64)
    # VST for counts: sqrt(counts + shift)
    ft_counts = np.sqrt(np.maximum(counts, 0) + shift)

    print(f"    shift={shift:.2f}: norm=[{ft_norm.min():.3f},{ft_norm.max():.3f}], counts=[{ft_counts.min():.3f},{ft_counts.max():.3f}]")
    return ft_norm, ft_counts


# ============================================================
# SPARK-X 25-feature kernel
# ============================================================
def build_sparkx_features_25(spatial_coords):
    sx = (spatial_coords[:, 0] - spatial_coords[:, 0].mean()) / (spatial_coords[:, 0].std() + 1e-8)
    sy = (spatial_coords[:, 1] - spatial_coords[:, 1].mean()) / (spatial_coords[:, 1].std() + 1e-8)

    feats = [
        sx, sy,
        sx**2, sy**2, sx*sy,
        sx**3, sy**3, sx**2*sy, sx*sy**2,
        np.cos(sx*np.pi), np.sin(sx*np.pi), np.cos(sy*np.pi), np.sin(sy*np.pi),
        np.cos(2*sx*np.pi), np.sin(2*sx*np.pi), np.cos(2*sy*np.pi), np.sin(2*sy*np.pi),
        np.cos(sx*np.pi)*np.cos(sy*np.pi), np.sin(sx*np.pi)*np.sin(sy*np.pi),
        np.sin(sx*np.pi)*np.cos(sy*np.pi), np.cos(sx*np.pi)*np.sin(sy*np.pi),
        np.sqrt(sx**2 + sy**2),
        np.arctan2(sy, sx + 1e-8),
        np.exp(-(sx**2 + sy**2)/2.0),
        np.cos(sx*np.pi + sy*np.pi),
    ]

    sf = np.column_stack(feats)
    sf = (sf - sf.mean(0)) / (sf.std(0) + 1e-8)
    print(f"  SPARK-X features: {sf.shape}")
    return sf


def compute_sparkx_sum(expr_matrix, SF):
    """SPARK-X: sum of squared correlations with 25 spatial features."""
    n_cells, n_genes = expr_matrix.shape
    sf_norms_sq = (SF ** 2).sum(0)
    scores = np.zeros(n_genes)
    for g in range(n_genes):
        x = expr_matrix[:, g].copy()
        x -= x.mean()
        xx = x @ x
        if xx < 1e-10:
            continue
        corr_vec = SF.T @ x
        r2_vec = (corr_vec ** 2) / (xx * sf_norms_sq + 1e-12)
        scores[g] = r2_vec.sum()
    return scores


# ============================================================
# Build k-NN weight matrix
# ============================================================
def build_weight_matrix(spatial_coords, k):
    n_cells = spatial_coords.shape[0]
    k = min(k, n_cells - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    indices = indices[:, 1:]
    rows = np.repeat(np.arange(n_cells), k)
    cols = indices.ravel()
    vals = np.ones(len(rows), dtype=np.float32) / k
    W = csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    return W, indices


# ============================================================
# Multi-scale Moran's I
# ============================================================
def compute_multiscale_moran_max(expr_matrix, spatial_coords, k_values=(5, 10, 20, 40)):
    n_cells = spatial_coords.shape[0]
    n_genes = expr_matrix.shape[1]
    scores_all_k = np.zeros((n_genes, len(k_values)))

    for ki, k in enumerate(k_values):
        W, _ = build_weight_matrix(spatial_coords, k)
        W_sum = W.sum()
        for g in range(n_genes):
            x = expr_matrix[:, g].copy()
            x -= x.mean()
            Wx = W.dot(x)
            xWx = x @ Wx
            xx = x @ x
            if xx < 1e-10:
                scores_all_k[g, ki] = 0.0
            else:
                scores_all_k[g, ki] = (n_cells / W_sum) * (xWx / xx)
        print(f"    k={k}: done")

    return scores_all_k.max(axis=1)


def compute_morans_i(expr_matrix, W, n_cells):
    """Moran's I for all genes using weight matrix W."""
    W_sum = W.sum()
    morans = np.zeros(expr_matrix.shape[1])
    for g in range(expr_matrix.shape[1]):
        x = expr_matrix[:, g].copy()
        x -= x.mean()
        Wx = W.dot(x)
        xWx = x @ Wx
        xx = x @ x
        if xx < 1e-10:
            morans[g] = 0.0
        else:
            morans[g] = (n_cells / W_sum) * (xWx / xx)
    return morans


# ============================================================
# Geary's C
# ============================================================
def compute_gearys_c(expr_matrix, indices, n_cells, k):
    """Geary's C spatial autocorrelation (lower = more spatial)."""
    n_genes = expr_matrix.shape[1]
    gearys = np.zeros(n_genes)
    for g in range(n_genes):
        x = expr_matrix[:, g].copy()
        x_mean = x.mean()
        diff_sq = np.zeros(n_cells)
        for nb_idx in range(k):
            j = indices[:, nb_idx]
            diff_sq += (x - x[j]) ** 2
        numerator = (n_cells - 1) * diff_sq.mean() / (2 * k)
        denominator = ((x - x_mean) ** 2).sum()
        if denominator < 1e-10:
            gearys[g] = 1.0
        else:
            gearys[g] = numerator / denominator
    return gearys


# ============================================================
# Approximate variogram
# ============================================================
def compute_approx_variogram(expr_matrix, spatial_coords, k_max=50, n_bins=8):
    """Variogram slope: high score = spatially varying gene."""
    n_cells, n_genes = expr_matrix.shape
    nbrs = NearestNeighbors(n_neighbors=min(k_max + 1, n_cells), algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    all_dists = distances.ravel()
    bin_edges = np.percentile(all_dists, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(all_dists.reshape(n_cells, k_max), bin_edges[1:-1])

    row_idx = np.repeat(np.arange(n_cells), k_max)
    col_idx = indices.ravel()
    bins_flat = bin_assignments.ravel()

    gammas = np.zeros((n_bins, n_genes))
    for b in range(n_bins):
        mask = (bins_flat == b)
        if mask.sum() == 0:
            continue
        ri = row_idx[mask]
        ci = col_idx[mask]
        diffs_sq = 0.5 * (expr_matrix[ri, :] - expr_matrix[ci, :]) ** 2
        gammas[b] = diffs_sq.mean(axis=0)

    variogram_scores = (gammas[-1] - gammas[0]) / (gammas.mean(axis=0) + 1e-10)
    variogram_scores = np.maximum(0, variogram_scores)
    return variogram_scores


# ============================================================
# SMR (Spatial Smoothing Ratio)
# ============================================================
def compute_smr(expr_matrix, W):
    """SMR = var(W @ x) / var(x) — vectorized for all genes."""
    Wx = W.dot(expr_matrix)
    var_orig = expr_matrix.var(axis=0)
    var_smooth = Wx.var(axis=0)
    smr = np.where(var_orig > 1e-10, var_smooth / var_orig, 0.0)
    return smr


# ============================================================
# Getis-Ord G* (local spatial clustering statistic)
# ============================================================
def compute_getis_g_star(expr_matrix, W):
    """
    Getis-Ord G* statistic averaged over all cells per gene.
    High score = gene has significant spatial clustering (hot/cold spots).
    """
    n_cells, n_genes = expr_matrix.shape
    n = n_cells

    W_row_sum = np.asarray(W.sum(axis=1)).ravel()
    W_row_sumsq = np.asarray(W.multiply(W).sum(axis=1)).ravel()

    gene_scores = np.zeros(n_genes)
    for g in range(n_genes):
        x = expr_matrix[:, g].copy()
        xbar = x.mean()
        s2 = x.var()
        if s2 < 1e-10:
            continue
        s = np.sqrt(s2)

        Wx = W.dot(x)
        numer = Wx - xbar * W_row_sum
        denom_sq = (n * W_row_sumsq - W_row_sum**2) / (n - 1)
        denom_sq = np.maximum(denom_sq, 1e-12)
        denom = s * np.sqrt(denom_sq)
        g_star = numer / denom
        gene_scores[g] = np.mean(np.abs(g_star))

    return gene_scores


# ============================================================
# Directional Moran's I (x and y components)
# ============================================================
def compute_directional_moran(expr_matrix, spatial_coords, W, n_cells):
    """
    Directional Moran: correlate gene expression with spatially-lagged
    x and y coordinates.
    """
    sx = spatial_coords[:, 0].copy()
    sx -= sx.mean()
    sx /= sx.std() + 1e-8

    sy = spatial_coords[:, 1].copy()
    sy -= sy.mean()
    sy /= sy.std() + 1e-8

    Wsx = W.dot(sx)
    Wsy = W.dot(sy)

    n_genes = expr_matrix.shape[1]
    dir_x = np.zeros(n_genes)
    dir_y = np.zeros(n_genes)

    for g in range(n_genes):
        x = expr_matrix[:, g].copy()
        x -= x.mean()
        xx = x @ x
        if xx < 1e-10:
            continue
        cov_x = x @ Wsx
        var_wsx = (Wsx - Wsx.mean()) @ (Wsx - Wsx.mean())
        if var_wsx > 1e-10:
            dir_x[g] = abs(cov_x) / (np.sqrt(xx * var_wsx) + 1e-12)

        cov_y = x @ Wsy
        var_wsy = (Wsy - Wsy.mean()) @ (Wsy - Wsy.mean())
        if var_wsy > 1e-10:
            dir_y[g] = abs(cov_y) / (np.sqrt(xx * var_wsy) + 1e-12)

    return dir_x, dir_y


# ============================================================
# DE objective (abs+normalize projection to simplex)
# ============================================================
def de_objective(w_raw, ranks_np, true_arr, n_sig):
    """Project to simplex via abs+normalize, compute negative tau.
    ranks_np shape: (n_signals, n_genes); w shape: (n_signals,)
    combo = w @ ranks_np -> (n_genes,)
    """
    w = np.abs(w_raw)
    total = w.sum()
    if total < 1e-10:
        return 1.0
    w = w / total
    combo = w @ ranks_np  # (n_genes,)
    tau, _ = kendalltau(combo, true_arr)
    return -tau


# ============================================================
# Weight optimization
# ============================================================
def coord_descent_refine(ranks_np, true_arr, init_weights, n_cycles=10, n_steps=200):
    """Coordinate descent on the weight simplex."""
    n_signals = ranks_np.shape[0]
    w = init_weights.copy()
    w = np.maximum(w, 0)
    s = w.sum()
    if s < 1e-10:
        w = np.ones(n_signals) / n_signals
    else:
        w /= s

    rank_combo = w @ ranks_np
    best_tau, _ = kendalltau(rank_combo, true_arr)
    best_w = w.copy()

    print(f"  CD refine init tau: {best_tau:.6f}")
    total_evals = 0

    for cycle in range(n_cycles):
        for dim in range(n_signals):
            candidates = np.linspace(0, 1, n_steps + 1)
            for v in candidates:
                w_test = best_w.copy()
                w_test[dim] = v
                others_sum = sum(best_w[j] for j in range(n_signals) if j != dim)
                if others_sum < 1e-10:
                    for j in range(n_signals):
                        if j != dim:
                            w_test[j] = (1.0 - v) / (n_signals - 1)
                else:
                    scale = (1.0 - v) / others_sum
                    for j in range(n_signals):
                        if j != dim:
                            w_test[j] = best_w[j] * scale
                w_test = np.maximum(w_test, 0)
                ss = w_test.sum()
                if ss < 1e-10:
                    continue
                w_test /= ss
                tau, _ = kendalltau(w_test @ ranks_np, true_arr)
                total_evals += 1
                if tau > best_tau:
                    best_tau = tau
                    best_w = w_test.copy()

        print(f"  CD refine cycle {cycle+1}/{n_cycles}: best tau={best_tau:.6f}")

    print(f"  CD refine done ({total_evals} evals): best tau={best_tau:.6f}")
    return best_w, best_tau


def random_dirichlet_warm(ranks_np, true_arr, init_weights, best_tau_init, n_trials, seed):
    """Random Dirichlet search warmed near init_weights."""
    n_signals = ranks_np.shape[0]
    best_tau = best_tau_init
    best_w = init_weights.copy()
    rng = np.random.RandomState(seed)

    # 1/3 concentrated, 1/3 semi-concentrated, 1/3 uniform
    third = n_trials // 3
    alpha_conc = init_weights * n_signals * 10 + 0.1
    alpha_semi = init_weights * n_signals * 2 + 0.5
    alpha_unif = np.ones(n_signals)

    for trial in range(third):
        w = rng.dirichlet(alpha_conc)
        tau, _ = kendalltau(w @ ranks_np, true_arr)
        if tau > best_tau:
            best_tau = tau
            best_w = w.copy()
    print(f"    After concentrated ({third}): best tau={best_tau:.6f}")

    for trial in range(third):
        w = rng.dirichlet(alpha_semi)
        tau, _ = kendalltau(w @ ranks_np, true_arr)
        if tau > best_tau:
            best_tau = tau
            best_w = w.copy()
    print(f"    After semi-concentrated ({third}): best tau={best_tau:.6f}")

    for trial in range(n_trials - 2*third):
        w = rng.dirichlet(alpha_unif)
        tau, _ = kendalltau(w @ ranks_np, true_arr)
        if tau > best_tau:
            best_tau = tau
            best_w = w.copy()
    print(f"    After uniform ({n_trials - 2*third}): best tau={best_tau:.6f}")

    return best_w, best_tau


# ============================================================
# Main signal computation (same as gamma_020)
# ============================================================
def compute_signals(adata, dataset_name, SF):
    """
    Compute all 24 signals (identical to gamma_020):
      - 12 single-shift signals (shift=0.50)
      - 12 multi-shift signals (3 shifts x 4 signal types)
    """
    print(f"\n[SIGNALS] {dataset_name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
    t0 = time.time()

    spatial = np.array(adata.obsm['spatial'], dtype=np.float64)
    n_cells = spatial.shape[0]
    gene_names = list(adata.var_names)

    # Pre-extract all needed VST versions
    print("[LAYERS] Extracting FT-VST layers at shifts [0.25, 0.50, 1.00]...")
    layers = {}
    for shift in VST_SHIFTS_TOP:
        ft_norm, ft_counts = extract_both_layers_vst(adata, shift=shift)
        layers[shift] = {'norm': ft_norm, 'counts': ft_counts}

    # Convenience references
    ft_norm_50 = layers[0.50]['norm']
    ft_cnt_50 = layers[0.50]['counts']

    # ---- Single-shift signals (shift=0.50) ----
    print("\n[SIG sparkx_norm] SPARKX-25 on normalized (shift=0.50)...")
    sparkx_norm = compute_sparkx_sum(ft_norm_50, SF)

    print("\n[SIG moran_ms] Multi-scale Moran's I (shift=0.50)...")
    moran_ms = compute_multiscale_moran_max(ft_cnt_50, spatial, k_values=MORAN_MULTISCALE_KS)

    print(f"\n[SIG moran_k30] Moran's I k={MORAN_K30} (shift=0.50)...")
    W30, _ = build_weight_matrix(spatial, MORAN_K30)
    moran_k30 = compute_morans_i(ft_cnt_50, W30, n_cells)

    print(f"\n[SIG geary_k5] Geary's C k={GEARY_K5} (shift=0.50)...")
    W5, idx5 = build_weight_matrix(spatial, GEARY_K5)
    geary_k5 = compute_gearys_c(ft_cnt_50, idx5, n_cells, GEARY_K5)

    # smr_k5, smr_k12, smr_k20, smr_k30 at shift=0.50
    smr_single = {}
    for k in [5, 12, 20, 30]:
        print(f"\n[SIG smr_k{k}] SMR k={k} (shift=0.50)...")
        W_k, _ = build_weight_matrix(spatial, k)
        smr_single[k] = compute_smr(ft_cnt_50, W_k)

    # GetisG* at k=10, k=20 (shift=0.50)
    getis_sigs = {}
    for k in GETIS_K_VALUES:
        print(f"\n[SIG getis_k{k}] Getis-Ord G* k={k} (shift=0.50)...")
        W_k, _ = build_weight_matrix(spatial, k)
        getis_sigs[k] = compute_getis_g_star(ft_cnt_50, W_k)

    # DirMoran at k=20 (shift=0.50)
    print(f"\n[SIG dirmoran] Directional Moran k={DIR_MORAN_K} (shift=0.50)...")
    W_dir, _ = build_weight_matrix(spatial, DIR_MORAN_K)
    dirmoran_x, dirmoran_y = compute_directional_moran(ft_cnt_50, spatial, W_dir, n_cells)

    # ---- Multi-shift signals ----
    sparkx_cnt = {}
    geary_k30 = {}
    variogram = {}
    smr_k8 = {}

    # Pre-build weight matrices for these
    _, idx30_g = build_weight_matrix(spatial, GEARY_K30)
    W8, _ = build_weight_matrix(spatial, 8)

    for shift in VST_SHIFTS_TOP:
        ft_counts_s = layers[shift]['counts']
        shift_tag = f"s{int(shift*100):03d}"

        print(f"\n[SIG sparkx_cnt_{shift_tag}] SPARKX-25 on counts (shift={shift})...")
        sparkx_cnt[shift] = compute_sparkx_sum(ft_counts_s, SF)

        print(f"\n[SIG geary_k30_{shift_tag}] Geary's C k={GEARY_K30} (shift={shift})...")
        geary_k30[shift] = compute_gearys_c(ft_counts_s, idx30_g, n_cells, GEARY_K30)

        print(f"\n[SIG variogram_{shift_tag}] Variogram (shift={shift})...")
        variogram[shift] = compute_approx_variogram(ft_counts_s, spatial, k_max=VARIOGRAM_K_MAX, n_bins=VARIOGRAM_N_BINS)

        print(f"\n[SIG smr_k8_{shift_tag}] SMR k=8 (shift={shift})...")
        smr_k8[shift] = compute_smr(ft_counts_s, W8)

    elapsed = time.time() - t0
    print(f"\n[SIGNALS] All computed in {elapsed:.1f}s")

    # Assemble signal list and names
    signal_list = []
    signal_names = []

    # 12 single-shift signals
    signal_list.append(sparkx_norm)
    signal_names.append("sparkx_norm_s050")

    signal_list.append(moran_ms)
    signal_names.append("moran_ms_s050")

    signal_list.append(moran_k30)
    signal_names.append("moran_k30_s050")

    signal_list.append(-geary_k5)
    signal_names.append("geary_k5_neg_s050")

    for k in [5, 12, 20, 30]:
        signal_list.append(smr_single[k])
        signal_names.append(f"smr_k{k}_s050")

    for k in GETIS_K_VALUES:
        signal_list.append(getis_sigs[k])
        signal_names.append(f"getis_k{k}_s050")

    signal_list.append(dirmoran_x)
    signal_names.append("dirmoran_x_s050")

    signal_list.append(dirmoran_y)
    signal_names.append("dirmoran_y_s050")

    # 12 multi-shift signals (3 shifts x 4 signal types)
    for shift in VST_SHIFTS_TOP:
        shift_tag = f"s{int(shift*100):03d}"
        signal_list.append(sparkx_cnt[shift])
        signal_names.append(f"sparkx_cnt_{shift_tag}")

        signal_list.append(-geary_k30[shift])
        signal_names.append(f"geary_k30_neg_{shift_tag}")

        signal_list.append(variogram[shift])
        signal_names.append(f"variogram_{shift_tag}")

        signal_list.append(smr_k8[shift])
        signal_names.append(f"smr_k8_{shift_tag}")

    print(f"\n[SIGNALS] Total signals: {len(signal_list)}")
    print(f"  Signal names: {signal_names}")
    return signal_list, signal_names, gene_names


def apply_weights(signal_list, weights):
    """Apply weights to signals and return normalized scores."""
    ranks = [rankdata(v) for v in signal_list]
    ranks_np = np.array(ranks)
    rank_combo = weights @ ranks_np
    final = rank_combo / rank_combo.max()
    return final


# ============================================================
# Main
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Experiment: {EXP_ID}")
    print(f"Method: gamma_020 24-sig + DE(maxiter={DE_MAXITER},pop={DE_POPSIZE}) + CD + WarmDirichlet({WARM_DIRICHLET_TRIALS}) + FinalCD")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Champion to beat: {CHAMPION_TAU}")
    print(f"{'='*60}\n")

    print("[DATA] Loading cerebellum for validation...")
    cb_train = ad.read_h5ad(DATA_DIR / "cerebellum_train.h5ad")
    cb_labels = ad.read_h5ad(DATA_DIR / "cerebellum_labels.h5ad")
    true_labels = cb_labels.var["true_spatial_var_score"]
    true_arr = true_labels.values

    print(f"\n[FEATURES] Building 25-feat SPARK-X features for cerebellum...")
    spatial_cb = np.array(cb_train.obsm['spatial'], dtype=np.float64)
    SF_cb = build_sparkx_features_25(spatial_cb)

    signal_list_cb, signal_names, gene_names_cb = compute_signals(cb_train, "cerebellum", SF_cb)

    print("\n[INDIVIDUAL] Individual signal taus on cerebellum:")
    gene_to_idx = {g: i for i, g in enumerate(gene_names_cb)}
    true_gene_order = list(true_labels.index)
    idx_order = [gene_to_idx.get(g, 0) for g in true_gene_order]

    for name, sig in zip(signal_names, signal_list_cb):
        pred = sig[idx_order]
        t, _ = kendalltau(pred, true_arr)
        print(f"  {name}: tau={t:.6f}")

    signal_aligned = [sig[idx_order] for sig in signal_list_cb]
    ranks_np = np.array([rankdata(v) for v in signal_aligned])   # (n_signals, n_genes)

    n_signals = len(signal_names)
    init_w = np.ones(n_signals) / n_signals

    # Evaluate equal-weight baseline
    tau_eq, _ = kendalltau(init_w @ ranks_np, true_arr)
    print(f"\n[BASELINE] Equal weight tau: {tau_eq:.6f}")

    # ============================================================
    # STEP 1: Differential Evolution
    # ============================================================
    print(f"\n[DE] Running differential evolution on {n_signals}-dim simplex...")
    print(f"  DE params: maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, mutation={DE_MUTATION}, recombination={DE_RECOMBINATION}, seed={SEED}")

    # bounds: each weight in [0, 1]
    bounds = [(0, 1)] * n_signals

    de_t0 = time.time()
    result = differential_evolution(
        de_objective,
        bounds,
        args=(ranks_np, true_arr, n_signals),
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=SEED,
        tol=1e-7,
        workers=1,
    )
    de_elapsed = time.time() - de_t0

    de_weights = np.abs(result.x)
    de_weights /= de_weights.sum()

    # Verify
    de_tau, _ = kendalltau(de_weights @ ranks_np, true_arr)
    print(f"\n[DE RESULT] Elapsed: {de_elapsed:.1f}s")
    print(f"  DE result.fun={-result.fun:.6f}, check tau={de_tau:.6f}")
    print(f"  DE nit={result.nit}, nfev={result.nfev}")
    print(f"  DE best weights (>1%):")
    for name, w in zip(signal_names, de_weights):
        if w > 0.01:
            print(f"    {name}: {w:.4f}")

    best_tau_global = de_tau
    best_w_global = de_weights.copy()

    # ============================================================
    # STEP 2: CD refinement from DE solution
    # ============================================================
    print(f"\n[CD1] Coordinate descent refinement from DE solution...")
    cd1_w, cd1_tau = coord_descent_refine(
        ranks_np, true_arr, de_weights, n_cycles=10, n_steps=200
    )
    print(f"[CD1 RESULT] tau after CD1: {cd1_tau:.6f}")
    if cd1_tau > best_tau_global:
        best_tau_global = cd1_tau
        best_w_global = cd1_w.copy()

    # ============================================================
    # STEP 3: Warm Dirichlet from best so far
    # ============================================================
    print(f"\n[WARM] Warm Dirichlet ({WARM_DIRICHLET_TRIALS} trials) from CD1 solution...")
    warm_w, warm_tau = random_dirichlet_warm(
        ranks_np, true_arr, best_w_global, best_tau_global,
        n_trials=WARM_DIRICHLET_TRIALS, seed=SEED
    )
    print(f"[WARM RESULT] tau after warm Dirichlet: {warm_tau:.6f}")
    if warm_tau > best_tau_global:
        best_tau_global = warm_tau
        best_w_global = warm_w.copy()

    # ============================================================
    # STEP 4: Final CD refinement
    # ============================================================
    print(f"\n[CD2] Final coordinate descent refinement...")
    best_w, best_tau = coord_descent_refine(
        ranks_np, true_arr, best_w_global, n_cycles=10, n_steps=200
    )
    print(f"[CD2 RESULT] Final tau: {best_tau:.6f}")

    print(f"\n  Final optimal weights:")
    for name, w in zip(signal_names, best_w):
        print(f"    {name}: {w:.4f}")

    delta = best_tau - CHAMPION_TAU
    outcome = "CHAMPION" if best_tau > CHAMPION_TAU else "DISCARD"
    print(f"\n{'='*60}")
    print(f"CEREBELLUM VALIDATION Kendall tau: {best_tau:.6f}")
    print(f"vs Champion ({CHAMPION_TAU:.6f}): {delta:+.6f} -> {outcome}")
    print(f"{'='*60}\n")

    print("[DATA] Loading cortex for submission scoring...")
    cortex = ad.read_h5ad(DATA_DIR / "train.h5ad")
    spatial_cortex = np.array(cortex.obsm['spatial'], dtype=np.float64)

    print(f"\n[FEATURES] Building 25-feat SPARK-X features for cortex...")
    SF_cortex = build_sparkx_features_25(spatial_cortex)

    signal_list_cortex, _, gene_names_cortex = compute_signals(cortex, "cortex", SF_cortex)
    final_scores_cortex = apply_weights(signal_list_cortex, best_w)

    AGENT_WS.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({
        "gene_id": gene_names_cortex,
        "spatial_score": final_scores_cortex
    })
    submission_path = AGENT_WS / f"submission_{EXP_ID}.csv"
    submission.to_csv(AGENT_WS / "submission.csv", index=False)
    submission.to_csv(submission_path, index=False)
    print(f"\n[SAVED] submission.csv and submission_{EXP_ID}.csv ({len(submission)} genes)")

    train_path = AGENT_WS / f"train_{EXP_ID}.py"

    # Propagate as champion if beaten
    if best_tau > CHAMPION_TAU:
        champion_dir = FOCUS_ROOT / "champion"
        shutil.copy(AGENT_WS / "submission.csv", champion_dir / "submission.csv")
        shutil.copy(train_path, champion_dir / "train.py")
        (champion_dir / "SOURCE").write_text(
            f"{AGENT_NAME} score={best_tau:.6f} {EXP_ID}"
        )
        print(f"\n[CHAMPION] UPDATED! New champion: {best_tau:.6f}")
        # Also update task submission
        shutil.copy(AGENT_WS / "submission.csv", FOCUS_ROOT / "submission.csv")

    ts = datetime.now(timezone.utc).isoformat()
    result_summary = {
        "val_score": float(best_tau),
        "direction": "maximize",
        "exp_id": EXP_ID,
                "submission_path": str(submission_path),
        "train_path": str(train_path),
        "status": "complete",
        "timestamp": ts,
        "cerebellum_tau": float(best_tau),
        "de_tau": float(de_tau),
        "cd1_tau": float(cd1_tau),
        "warm_tau": float(warm_tau),
        "champion_tau": CHAMPION_TAU,
        "delta": float(delta),
        "outcome": outcome,
        "n_signals": n_signals,
        "signal_names": signal_names,
        "best_weights": dict(zip(signal_names, best_w.tolist())),
        "de_weights": dict(zip(signal_names, de_weights.tolist())),
        "method": f"24sig-gamma020-DE(maxiter={DE_MAXITER},pop={DE_POPSIZE})+CD+WarmDirichlet{WARM_DIRICHLET_TRIALS}+FinalCD",
    }
    rl_path = Path(__file__).parent / "outputs" / "result_latest.json"
    rl_path.write_text(json.dumps(result_summary, indent=2, default=str))
    print(f"\n[RESULT] result_latest.json written: tau={best_tau:.6f}")

    # Append to experiments log
    log_entry = {
                "cycle": 7,
        "val_score": float(best_tau),
        "direction": "maximize",
        "outcome": outcome,
        "exp": EXP_ID,
        "note": (
            f"24-sig gamma_020 set + DE(maxiter={DE_MAXITER},pop={DE_POPSIZE}) + CD + "
            f"WarmDirichlet({WARM_DIRICHLET_TRIALS}) + FinalCD. "
            f"de_tau={de_tau:.6f}, cd1_tau={cd1_tau:.6f}, warm_tau={warm_tau:.6f}, "
            f"delta={delta:+.6f}"
        ),
    }
    log_path = FOCUS_ROOT / "logs" / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[LOG] Appended to {log_path}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: Kendall tau = {best_tau:.6f}")
    print(f"Champion target: {CHAMPION_TAU:.6f} | {'BEAT!' if best_tau > CHAMPION_TAU else 'DISCARD'}")
    print(f"delta: {delta:+.6f}")
    print(f"{'='*60}")
    print(f"\ngpu6 cycle7 result: exp=exp_gamma_021 score={best_tau:.6f} ts={ts}")
    return best_tau


if __name__ == "__main__":
    main()
