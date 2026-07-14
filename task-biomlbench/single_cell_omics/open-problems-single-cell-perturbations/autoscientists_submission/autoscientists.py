"""
Experiment: exp_gamma_013 — Extended HP search on top of champion (exp_gamma_010)
Agent: biomlbp_scpert_2_gpu5 / Team: gamma

Key changes from champion (exp_gamma_010):
- Extended per-gene alpha grid: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
  vs champion's [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
- Finer ensemble weight grid: [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
  vs champion's [0.0, 0.1, 0.2, ..., 1.0] (coarser but wider)
- Also search global Ridge alpha: [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]
- Same 4D per-gene features: [T_g, NK_g, Treg_g, tanimoto_max]
- CT-specific: separate alpha + weight tuning for B cells vs Myeloid cells
- No KRR

Rationale:
- Champion used coarse alpha grid; finer search may find better per-gene alpha
- Weight grid in champion was [0,0.1,...,1.0]; the optimal values for B cells were 0.5/0.5
  and for Myeloid were 0.7/0.3 — adding fine grid around these regions may improve
- Global Ridge alpha was fixed at 5000; searching it may help
"""

import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WORKSPACE = Path(__file__).parent / "outputs"
AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
EXP_ID = "exp_gamma_013"

print("Loading data...")
de_train = pd.read_parquet(DATA_DIR / "de_train.parquet")
id_map = pd.read_csv(DATA_DIR / "id_map.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

gene_cols = [c for c in sample_sub.columns if c != "id"]
N_GENES = len(gene_cols)
print(f"Genes: {N_GENES}")

val_df = de_train[de_train["split"] == "public_test"].copy()
all_bm_df = de_train[de_train["cell_type"].isin(["B cells", "Myeloid cells"])].copy()
print(f"Val rows: {len(val_df)}, All B/M rows: {len(all_bm_df)}")

FEATURE_CELL_TYPES = ["T cells", "NK cells", "Tregs"]
TARGET_CELL_TYPES = ["B cells", "Myeloid cells"]

# --- RDKit fingerprints ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not available — Tanimoto will be 0 for all compounds")

FP_BITS = 2048
smiles_map = {}
for _, row in de_train.iterrows():
    sm = row["sm_name"]
    if sm not in smiles_map and "SMILES" in row and isinstance(row.get("SMILES"), str):
        smiles_map[sm] = row["SMILES"]

all_compounds = list(de_train["sm_name"].unique())
fp_dict = {}
fp_rdkit = {}
if RDKIT_OK:
    for sm in all_compounds:
        smi = smiles_map.get(sm)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
                fp_rdkit[sm] = rdkit_fp
                fp_dict[sm] = np.array(rdkit_fp, dtype=np.float32)
            else:
                fp_dict[sm] = np.zeros(FP_BITS, dtype=np.float32)
        else:
            fp_dict[sm] = np.zeros(FP_BITS, dtype=np.float32)
    print(f"Valid Morgan FPs: {len(fp_rdkit)}/{len(all_compounds)}")


def max_tanimoto_to_set(query_sm, train_sms):
    """Compute max Tanimoto similarity of query compound to a set of training compounds."""
    if not RDKIT_OK:
        return 0.0
    fpa = fp_rdkit.get(query_sm)
    if fpa is None:
        return 0.0
    best = 0.0
    for sm in train_sms:
        fpb = fp_rdkit.get(sm)
        if fpb is not None:
            s = DataStructs.TanimotoSimilarity(fpa, fpb)
            if s > best:
                best = s
    return best


FP_WEIGHT = 1.0


def build_per_gene_profiles(df, feature_cell_types, gene_cols):
    """Build per-gene profiles: for each compound, T/NK/Treg gene expression (3 x N_GENES)."""
    profiles = {}
    for sm in df["sm_name"].unique():
        sub = df[df["sm_name"] == sm]
        rows_per_ct = []
        for ct in feature_cell_types:
            ct_rows = sub[sub["cell_type"] == ct][gene_cols].values
            rows_per_ct.append(ct_rows.mean(axis=0) if len(ct_rows) > 0 else np.zeros(len(gene_cols)))
        profiles[sm] = np.stack(rows_per_ct, axis=0)  # (3, N_GENES)
    return profiles


def solve_per_gene_4d(X_pg_4d, y, alpha):
    """
    Per-gene Ridge with 4D features per gene: [T_g, NK_g, Treg_g, tanimoto_max].
    X_pg_4d: (n_compounds, 4, N_GENES)
    y: (n_compounds, N_GENES)
    Returns: pg_coefs of shape (N_GENES, 4)
    """
    n_feat = X_pg_4d.shape[1]  # 4
    pg_coefs = np.zeros((N_GENES, n_feat))
    for g in range(N_GENES):
        X_g = X_pg_4d[:, :, g]  # (n_compounds, 4)
        y_g = y[:, g]
        XtX = X_g.T @ X_g + alpha * np.eye(n_feat)
        Xty = X_g.T @ y_g
        pg_coefs[g] = np.linalg.solve(XtX, Xty)
    return pg_coefs


def predict_per_gene_4d(X_val_pg_4d, pg_coefs):
    """
    X_val_pg_4d: (n_val, 4, N_GENES)
    pg_coefs: (N_GENES, 4)
    Returns: (n_val, N_GENES)
    """
    X_T = X_val_pg_4d.transpose(2, 0, 1)  # (N_GENES, n_val, 4)
    pred_T = np.einsum('gni,gi->gn', X_T, pg_coefs)
    return pred_T.T  # (n_val, N_GENES)


def mrrmse(y_true, y_pred):
    return float(np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))))


# Build base per-gene profiles (T/NK/Treg only)
print("Building per-gene profiles from all data...")
full_per_gene_profiles = build_per_gene_profiles(de_train, FEATURE_CELL_TYPES, gene_cols)

# --- Extended HP search: also tune global Ridge alpha ---
# Search global Ridge alpha via LOOCV over val compounds
print("\nSearching global Ridge alpha via LOOCV on val compounds...")
GLOBAL_ALPHA_CANDIDATES = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]
PG_ALPHA_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# First pass: find best global alpha and per-gene alpha jointly via LOOCV
# To save time, use LOOCV on ALL B/M compounds for per-gene alpha selection
# Then test different global alphas during the val evaluation loop

print("\nTuning per-gene alpha (4D features) via LOOCV over ALL B/M compounds...")

BEST_PG_ALPHAS = {}
for ct in TARGET_CELL_TYPES:
    all_ct_rows = all_bm_df[all_bm_df["cell_type"] == ct]
    all_sms_ct = list(all_ct_rows["sm_name"].unique())
    all_sms_ct = [sm for sm in all_sms_ct if sm in full_per_gene_profiles]
    n = len(all_sms_ct)

    y_all = np.array([all_ct_rows[all_ct_rows["sm_name"] == sm][gene_cols].values.mean(0) for sm in all_sms_ct])

    # Precompute tanimoto for each compound to all others in B/M set
    tanimoto_vec = np.array([max_tanimoto_to_set(sm, [s for s in all_sms_ct if s != sm]) for sm in all_sms_ct])

    # Build 4D array: (n, 4, N_GENES)
    X_pg_base = np.array([full_per_gene_profiles[sm] for sm in all_sms_ct])  # (n, 3, N_GENES)
    tanimoto_broadcast = np.tile(tanimoto_vec[:, None, None], (1, 1, N_GENES))  # (n, 1, N_GENES)
    X_pg_4d_all = np.concatenate([X_pg_base, tanimoto_broadcast], axis=1)  # (n, 4, N_GENES)

    best_alpha = PG_ALPHA_CANDIDATES[0]
    best_loocv = float("inf")

    for alpha in PG_ALPHA_CANDIDATES:
        loocv_preds = np.zeros_like(y_all)
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            X_tr = X_pg_4d_all[idx]
            y_tr = y_all[idx]
            pg_coefs = solve_per_gene_4d(X_tr, y_tr, alpha)
            loocv_preds[i] = predict_per_gene_4d(X_pg_4d_all[i:i+1], pg_coefs)[0]
        score = mrrmse(y_all, loocv_preds)
        print(f"  {ct} pg_alpha={alpha}: LOOCV MRRMSE={score:.4f}")
        if score < best_loocv:
            best_loocv = score
            best_alpha = alpha

    BEST_PG_ALPHAS[ct] = best_alpha
    print(f"  => {ct} best pg4d alpha={best_alpha} (LOOCV={best_loocv:.4f})")

print(f"\nBest per-gene (4D) alphas: {BEST_PG_ALPHAS}")

# Second pass: evaluate different global Ridge alphas via val LOOCV
# Then use finer weight grid
print("\nEvaluating ensemble via LOOCV on val compounds (searching global Ridge alpha + finer weights)...")
val_df_r = val_df.reset_index(drop=True)
y_true_all_val = val_df_r[gene_cols].values

# Store per-compound predictions for each global alpha
val_preds_by_global_alpha = {ga: np.zeros((len(val_df_r), N_GENES)) for ga in GLOBAL_ALPHA_CANDIDATES}
val_preds_pergene4d = np.zeros((len(val_df_r), N_GENES))

# We only need to run per-gene once (alpha is fixed above), but global changes per ga
# For efficiency: do one LOOCV loop, store both per-gene preds + global preds per alpha

for ct in TARGET_CELL_TYPES:
    all_ct_rows = all_bm_df[all_bm_df["cell_type"] == ct]
    all_sms_ct = list(all_ct_rows["sm_name"].unique())
    all_sms_ct_feat = [sm for sm in all_sms_ct if sm in full_per_gene_profiles]

    val_ct = val_df_r[val_df_r["cell_type"] == ct]
    val_sms = list(val_ct["sm_name"])
    mask = val_df_r["cell_type"] == ct

    pg_alpha = BEST_PG_ALPHAS[ct]

    preds_pg4d = np.zeros((len(val_sms), N_GENES))
    preds_by_ga = {ga: np.zeros((len(val_sms), N_GENES)) for ga in GLOBAL_ALPHA_CANDIDATES}

    print(f"\n{ct}: predicting {len(val_sms)} val compounds via LOOCV...")
    for vi, val_sm in enumerate(val_sms):
        loocv_sms = [sm for sm in all_sms_ct_feat if sm != val_sm]

        # Build loocv features (global with tanimoto)
        loocv_feats = {}
        for sm in loocv_sms:
            sub = de_train[de_train["sm_name"] == sm]
            parts = []
            for fct in FEATURE_CELL_TYPES:
                fct_rows = sub[sub["cell_type"] == fct][gene_cols].values
                parts.append(fct_rows.mean(axis=0) if len(fct_rows) > 0 else np.zeros(N_GENES))
            fp = fp_dict.get(sm, np.zeros(FP_BITS, dtype=np.float32)) * FP_WEIGHT
            parts.append(fp)
            other_sms = [s for s in loocv_sms if s != sm]
            max_sim = max_tanimoto_to_set(sm, other_sms) if other_sms else 0.0
            parts.append(np.array([max_sim], dtype=np.float32))
            loocv_feats[sm] = np.concatenate(parts).astype(np.float32)

        # Val global features
        sub_val = de_train[de_train["sm_name"] == val_sm]
        val_parts = []
        for fct in FEATURE_CELL_TYPES:
            fct_rows = sub_val[sub_val["cell_type"] == fct][gene_cols].values
            val_parts.append(fct_rows.mean(axis=0) if len(fct_rows) > 0 else np.zeros(N_GENES))
        fp_val = fp_dict.get(val_sm, np.zeros(FP_BITS, dtype=np.float32)) * FP_WEIGHT
        val_parts.append(fp_val)
        max_sim_val = max_tanimoto_to_set(val_sm, loocv_sms)
        val_parts.append(np.array([max_sim_val], dtype=np.float32))
        val_feat_g = np.concatenate(val_parts).astype(np.float32)

        # Build 4D per-gene features for training
        n_tr = len(loocv_sms)
        X_tr_pg_base = np.array([full_per_gene_profiles.get(sm, np.zeros((3, N_GENES))) for sm in loocv_sms])
        tanimoto_tr_vec = np.array([loocv_feats[sm][-1] for sm in loocv_sms])
        tanimoto_tr_broadcast = np.tile(tanimoto_tr_vec[:, None, None], (1, 1, N_GENES))
        X_tr_pg_4d = np.concatenate([X_tr_pg_base, tanimoto_tr_broadcast], axis=1)

        X_val_pg_base = full_per_gene_profiles.get(val_sm, np.zeros((3, N_GENES))).reshape(1, 3, N_GENES)
        tanimoto_val_broadcast = np.tile(np.array([[[max_sim_val]]]), (1, 1, N_GENES))
        X_val_pg_4d = np.concatenate([X_val_pg_base, tanimoto_val_broadcast], axis=1)

        y_tr = np.array([all_ct_rows[all_ct_rows["sm_name"] == sm][gene_cols].values.mean(0) for sm in loocv_sms])

        X_tr_g = np.array([loocv_feats[sm] for sm in loocv_sms])

        # Global Ridge for each alpha candidate
        for ga in GLOBAL_ALPHA_CANDIDATES:
            ridge_g = Ridge(alpha=ga)
            ridge_g.fit(X_tr_g, y_tr)
            preds_by_ga[ga][vi] = ridge_g.predict(val_feat_g.reshape(1, -1))[0]

        # Per-gene Ridge (fixed alpha)
        pg_coefs = solve_per_gene_4d(X_tr_pg_4d, y_tr, pg_alpha)
        preds_pg4d[vi] = predict_per_gene_4d(X_val_pg_4d, pg_coefs)[0]

        if (vi + 1) % 5 == 0:
            print(f"  Val {vi+1}/{len(val_sms)} done")

    val_preds_pergene4d[mask.values] = preds_pg4d
    for ga in GLOBAL_ALPHA_CANDIDATES:
        val_preds_by_global_alpha[ga][mask.values] = preds_by_ga[ga]

    y_true_ct = val_ct[gene_cols].values
    print(f"  {ct}: per-gene4d={mrrmse(y_true_ct, preds_pg4d):.4f}")
    for ga in GLOBAL_ALPHA_CANDIDATES:
        print(f"    global alpha={ga}: {mrrmse(y_true_ct, preds_by_ga[ga]):.4f}")

# Finer weight grid search: for each global alpha, find best per-CT weights
# Then pick the best overall (global_alpha, w_B, w_M) combination
print("\nSearching best global alpha + per-CT weights with finer grid...")
WEIGHT_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

best_overall_score = float("inf")
best_global_alpha = 5000.0
best_weights_per_ct = {}

for ga in GLOBAL_ALPHA_CANDIDATES:
    val_preds_global = val_preds_by_global_alpha[ga]

    weights_ct = {}
    ct_scores = {}
    for ct in TARGET_CELL_TYPES:
        mask = val_df_r["cell_type"] == ct
        y_true_ct = val_df_r[mask][gene_cols].values
        g_ct = val_preds_global[mask.values]
        pg_ct = val_preds_pergene4d[mask.values]

        best_score_ct = float("inf")
        best_w_ct = (0.5, 0.5)
        for w_g in WEIGHT_GRID:
            w_pg = round(1.0 - w_g, 8)
            ens = w_g * g_ct + w_pg * pg_ct
            score = mrrmse(y_true_ct, ens)
            if score < best_score_ct:
                best_score_ct = score
                best_w_ct = (w_g, w_pg)
        weights_ct[ct] = best_w_ct
        ct_scores[ct] = best_score_ct

    # Compute overall val MRRMSE with this global alpha and per-CT weights
    val_preds_final = np.zeros((len(val_df_r), N_GENES))
    for ct in TARGET_CELL_TYPES:
        mask = val_df_r["cell_type"] == ct
        w_g, w_pg = weights_ct[ct]
        val_preds_final[mask.values] = w_g * val_preds_global[mask.values] + w_pg * val_preds_pergene4d[mask.values]

    overall_score = mrrmse(y_true_all_val, val_preds_final)
    print(f"  global_alpha={ga}: score={overall_score:.4f} | B={ct_scores['B cells']:.4f}({weights_ct['B cells']}) | M={ct_scores['Myeloid cells']:.4f}({weights_ct['Myeloid cells']})")

    if overall_score < best_overall_score:
        best_overall_score = overall_score
        best_global_alpha = ga
        best_weights_per_ct = weights_ct

print(f"\nBest configuration:")
print(f"  Global Ridge alpha: {best_global_alpha}")
print(f"  Per-gene alphas: {BEST_PG_ALPHAS}")
print(f"  Per-CT weights: {best_weights_per_ct}")
print(f"  Final Val MRRMSE (exp_gamma_013): {best_overall_score:.4f}")
print(f"  Champion (exp_gamma_010) MRRMSE: 0.8504")

RIDGE_ALPHA = best_global_alpha

final_weights = {ct: {"w_global": best_weights_per_ct[ct][0], "w_pg4d": best_weights_per_ct[ct][1]}
                 for ct in TARGET_CELL_TYPES}

print("\n" + "=" * 50)
print(json.dumps({
    "model": "ExtendedHPSearch-4D-PerGeneRidge-PerCTWeights",
    "ridge_alpha": RIDGE_ALPHA,
    "best_pg_alphas": {ct: BEST_PG_ALPHAS[ct] for ct in TARGET_CELL_TYPES},
    "final_weights": final_weights,
    "val_mrrmse": best_overall_score,
}))
print("=" * 50 + "\n")

# Retrain on full data for submission
print("Retraining on full data for submission...")
full_all_bm = de_train[de_train["cell_type"].isin(TARGET_CELL_TYPES)]

all_sms_full = list(de_train["sm_name"].unique())
full_feats_tanimoto = {}
for sm in all_sms_full:
    sub = de_train[de_train["sm_name"] == sm]
    parts = []
    for fct in FEATURE_CELL_TYPES:
        fct_rows = sub[sub["cell_type"] == fct][gene_cols].values
        parts.append(fct_rows.mean(axis=0) if len(fct_rows) > 0 else np.zeros(N_GENES))
    fp = fp_dict.get(sm, np.zeros(FP_BITS, dtype=np.float32)) * FP_WEIGHT
    parts.append(fp)
    other = [s for s in all_sms_full if s != sm]
    max_sim = max_tanimoto_to_set(sm, other) if other else 0.0
    parts.append(np.array([max_sim], dtype=np.float32))
    full_feats_tanimoto[sm] = np.concatenate(parts).astype(np.float32)

FEAT_DIM_FULL = len(next(iter(full_feats_tanimoto.values())))

test_preds = []
for ct in TARGET_CELL_TYPES:
    full_ct_df = full_all_bm[full_all_bm["cell_type"] == ct]
    all_sms_ct = [sm for sm in full_ct_df["sm_name"].unique() if sm in full_feats_tanimoto]
    n = len(all_sms_ct)

    X_all_g = np.array([full_feats_tanimoto[sm] for sm in all_sms_ct])
    y_all = np.array([full_ct_df[full_ct_df["sm_name"] == sm][gene_cols].values.mean(0) for sm in all_sms_ct])

    X_all_pg_base = np.array([full_per_gene_profiles.get(sm, np.zeros((3, N_GENES))) for sm in all_sms_ct])
    tanimoto_all_vec = np.array([full_feats_tanimoto[sm][-1] for sm in all_sms_ct])
    tanimoto_all_broadcast = np.tile(tanimoto_all_vec[:, None, None], (1, 1, N_GENES))
    X_all_pg_4d = np.concatenate([X_all_pg_base, tanimoto_all_broadcast], axis=1)

    ridge_g = Ridge(alpha=RIDGE_ALPHA)
    ridge_g.fit(X_all_g, y_all)

    pg_alpha = BEST_PG_ALPHAS[ct]
    pg_coefs = solve_per_gene_4d(X_all_pg_4d, y_all, pg_alpha)

    test_ct = id_map[id_map["cell_type"] == ct]
    test_sms_ct = list(test_ct["sm_name"])

    test_feats_ct = {}
    for sm in test_sms_ct:
        sub = de_train[de_train["sm_name"] == sm]
        parts = []
        for fct in FEATURE_CELL_TYPES:
            fct_rows = sub[sub["cell_type"] == fct][gene_cols].values
            parts.append(fct_rows.mean(axis=0) if len(fct_rows) > 0 else np.zeros(N_GENES))
        fp = fp_dict.get(sm, np.zeros(FP_BITS, dtype=np.float32)) * FP_WEIGHT
        parts.append(fp)
        max_sim = max_tanimoto_to_set(sm, all_sms_ct)
        parts.append(np.array([max_sim], dtype=np.float32))
        test_feats_ct[sm] = np.concatenate(parts).astype(np.float32)

    X_test_g = np.array([test_feats_ct.get(sm, np.zeros(FEAT_DIM_FULL)) for sm in test_sms_ct])

    X_test_pg_base = np.array([full_per_gene_profiles.get(sm, np.zeros((3, N_GENES))) for sm in test_sms_ct])
    tanimoto_test_vec = np.array([test_feats_ct.get(sm, np.zeros(FEAT_DIM_FULL))[-1] for sm in test_sms_ct])
    tanimoto_test_broadcast = np.tile(tanimoto_test_vec[:, None, None], (1, 1, N_GENES))
    X_test_pg_4d = np.concatenate([X_test_pg_base, tanimoto_test_broadcast], axis=1)

    pred_g = ridge_g.predict(X_test_g)
    pred_pg = predict_per_gene_4d(X_test_pg_4d, pg_coefs)

    w_g, w_pg = best_weights_per_ct[ct]
    pred_final = w_g * pred_g + w_pg * pred_pg

    for i, (_, row) in enumerate(test_ct.iterrows()):
        test_preds.append((row["id"], pred_final[i]))

    print(f"  {ct}: {len(test_sms_ct)} test predictions done")

test_preds_sorted = sorted(test_preds, key=lambda x: x[0])
ids = [p[0] for p in test_preds_sorted]
preds_matrix = np.array([p[1] for p in test_preds_sorted])

submission = pd.DataFrame(preds_matrix, columns=gene_cols)
submission.insert(0, "id", ids)

out_path = AGENT_WORKSPACE / "submission.csv"
submission.to_csv(out_path, index=False)
print(f"Saved submission, shape={submission.shape}")

import shutil as _sh
_sh.copy(str(out_path), str(AGENT_WORKSPACE / f"submission_{EXP_ID}.csv"))
_sh.copy(str(Path(__file__).resolve()), str(AGENT_WORKSPACE / f"train_{EXP_ID}.py"))
print(f"\nFinal Val MRRMSE: {best_overall_score:.4f}")
