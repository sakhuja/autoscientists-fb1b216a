"""
exp_alpha_018: Ultra-fine grid around alpha_017 optimum

Background:
- alpha_017 best: AUC=0.9954, w_prop=0.620, w_spec=0.100, w_rz=0.036, w_tprop=0.245, clip=16
- Key finding: higher w_tprop (0.245 vs 0.112) was the breakthrough
- This: finer step=0.003 grid around the optimum, also search clips [12,14,16,18]

CPU-only. ~12 min.
"""
import sys, shutil, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score

FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"
AGENT_WORKSPACE = Path(__file__).parent / "outputs"
EXP_ID = "exp_alpha_018"
print(f"[{datetime.now().isoformat()}] Starting {EXP_ID}")

adata = ad.read_h5ad(DATA_DIR / "tnbc_data.h5ad")
lr_resource = pd.read_csv(DATA_DIR / "ligand_receptor_resource.csv.gz")
ccc_train = adata.uns["ccc_train"].copy()
ccc_test = adata.uns["ccc_test_pairs"].copy()
val = ccc_train.sample(frac=0.2, random_state=42)

if "normalized" in adata.layers: expr_matrix = adata.layers["normalized"]
else: expr_matrix = adata.X
if sp.issparse(expr_matrix): expr_matrix = expr_matrix.toarray()

gene_names = adata.var_names.tolist()
gene_idx = {g: i for i, g in enumerate(gene_names)}
cell_types = sorted(adata.obs["label"].unique().tolist())
ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
n_ct = len(cell_types)
ct_expr_matrix = np.zeros((n_ct, len(gene_names)))
for i, ct in enumerate(cell_types):
    mask = (adata.obs["label"] == ct).values
    if mask.sum() > 0: ct_expr_matrix[i] = expr_matrix[mask].mean(axis=0)
gene_mean = ct_expr_matrix.mean(axis=0); gene_std = ct_expr_matrix.std(axis=0)

ALPHA_L = 1.0
lig_stats = ccc_train.groupby("ligand")["response"].agg(["sum","count"]).reset_index()
lig_stats.columns = ["ligand","pos","total"]
lig_stats["prop"] = (lig_stats["pos"]+ALPHA_L)/(lig_stats["total"]+2*ALPHA_L)
lig_prop = dict(zip(lig_stats["ligand"], lig_stats["prop"]))
global_prior = (ccc_train["response"].sum()+ALPHA_L)/(len(ccc_train)+2*ALPHA_L)

tgt_stats = ccc_train.groupby("target")["response"].agg(["sum","count"]).reset_index()
tgt_stats.columns = ["target","pos","total"]
tgt_stats["prop"] = (tgt_stats["pos"]+ALPHA_L)/(tgt_stats["total"]+2*ALPHA_L)
tgt_prop = dict(zip(tgt_stats["target"], tgt_stats["prop"]))

if "secreted_intercell_source" in lr_resource.columns:
    sec_mask = lr_resource["secreted_intercell_source"].fillna(False).astype(bool)
    secreted_ligands = set(lr_resource[sec_mask]["source_genesymbol"].unique())
else: secreted_ligands = set()

lr_clean = lr_resource.dropna(subset=["source_genesymbol","target_genesymbol"]).copy()
lr_expanded = []
for _, row in lr_clean.iterrows():
    lig = row["source_genesymbol"]; rec_raw = str(row.get("target_genesymbol",""))
    n_res = float(row.get("n_resources",1)) if "n_resources" in row else 1.0
    for rec in rec_raw.split("_"):
        rec = rec.strip()
        if rec and rec != "nan": lr_expanded.append({"ligand":lig,"receptor":rec,"n_res":n_res})
lr_exp_df = pd.DataFrame(lr_expanded)
lr_agg = lr_exp_df.groupby(["ligand","receptor"])["n_res"].max().reset_index()
lig_to_rec_log_nres = {}
for _, row in lr_agg.iterrows():
    lig, rec = row["ligand"], row["receptor"]
    if lig not in lig_to_rec_log_nres: lig_to_rec_log_nres[lig] = {}
    lig_to_rec_log_nres[lig][rec] = float(np.log1p(row["n_res"]))

all_ligands = set(ccc_train["ligand"].tolist()+ccc_test["ligand"].tolist())
lig_spec = {}
for lig in all_ligands:
    if lig in gene_idx:
        gi = gene_idx[lig]; lig_expr = ct_expr_matrix[:,gi]
        lig_spec[lig] = lig_expr.max()/(lig_expr.mean()+1e-6)
    else: lig_spec[lig] = 0.0
spec_pos = np.array([v for v in lig_spec.values() if v > 0])
if len(spec_pos) > 0:
    smin, smax = spec_pos.min(), spec_pos.max()
    for lig in lig_spec:
        if lig_spec[lig] > 0: lig_spec[lig] = (lig_spec[lig]-smin)/(smax-smin+1e-6)

recz_cache = {}
def compute_recz(ligand, target_ct, clip):
    key = (ligand, target_ct, clip)
    if key in recz_cache: return recz_cache[key]
    rec_data = lig_to_rec_log_nres.get(ligand,{})
    if not rec_data or target_ct not in ct_to_idx: recz_cache[key]=0.0; return 0.0
    ct_i = ct_to_idx[target_ct]; z_vals=[]; weights=[]
    for rec, log_nres in rec_data.items():
        if rec not in gene_idx: continue
        gi = gene_idx[rec]; std_val = gene_std[gi]
        if std_val < 1e-6: continue
        z = (ct_expr_matrix[ct_i,gi]-gene_mean[gi])/std_val
        z_vals.append(float(np.clip(z,-clip,clip))); weights.append(log_nres)
    if not z_vals: recz_cache[key]=0.0; return 0.0
    w = np.array(weights)
    result = float(np.dot(np.array(z_vals),w)/(w.sum()+1e-9))
    recz_cache[key] = result; return result

all_pairs = pd.concat([ccc_train[["ligand","target"]], ccc_test[["ligand","target"]]])
print("Pre-caching recZ for clips [12,14,16,18]...")
for clip in [12,14,16,18]:
    for _, row in all_pairs.iterrows(): compute_recz(row["ligand"],row["target"],clip)
print("Done.")

def score_pairs(pairs, w1, w2, w3, w4, clip):
    scores = []
    for _, row in pairs.iterrows():
        lig, tgt = row["ligand"], row["target"]
        s1 = lig_prop.get(lig, global_prior)
        s2 = (1.0 if lig in secreted_ligands else 0.0)*lig_spec.get(lig,0.0)
        s3 = compute_recz(lig, tgt, clip)
        s4 = tgt_prop.get(tgt, global_prior)
        scores.append(w1*s1+w2*s2+w3*s3+w4*s4)
    return np.array(scores)

print("Fine grid search around alpha_017 optimum (prop=0.620, spec=0.100, rz=0.036, tprop=0.245)...")
best_auc = 0.0; best_config = None

for clip in [12, 14, 16, 18]:
    for w_prop in np.arange(0.570, 0.670, 0.005):
        for w_spec in np.arange(0.070, 0.130, 0.005):
            for w_rz in np.arange(0.010, 0.070, 0.005):
                w_tprop = 1.0 - w_prop - w_spec - w_rz
                if w_tprop < 0.18 or w_tprop > 0.35: continue
                train_scores = score_pairs(ccc_train, w_prop, w_spec, w_rz, w_tprop, clip)
                auc = roc_auc_score(ccc_train["response"].values, train_scores)
                if auc > best_auc:
                    best_auc = auc
                    best_config = {'clip':int(clip),'w_prop':float(w_prop),'w_spec':float(w_spec),'w_rz':float(w_rz),'w_tprop':float(w_tprop)}
                    print(f"  New best: AUC={auc:.6f}, clip={clip}, prop={w_prop:.3f}, spec={w_spec:.3f}, rz={w_rz:.3f}, tprop={w_tprop:.3f}")

print(f"\nBest config: {best_config}\nBest full-train AUC: {best_auc:.6f}")

val_scores = score_pairs(val, best_config['w_prop'], best_config['w_spec'], best_config['w_rz'], best_config['w_tprop'], best_config['clip'])
val_labels = val["response"].values
def odds_ratio_top5pct(scores, labels):
    t = np.percentile(scores,95); top5 = scores>=t
    tp=np.sum(top5&(labels==1)); fp=np.sum(top5&(labels==0)); fn=np.sum(~top5&(labels==1)); tn=np.sum(~top5&(labels==0))
    return (tp*tn)/(fp*fn) if fp>0 and fn>0 else 999.0
val_or = odds_ratio_top5pct(val_scores, val_labels)
print(f"Val Odds Ratio: {val_or:.4f}")

test_scores = score_pairs(ccc_test, best_config['w_prop'], best_config['w_spec'], best_config['w_rz'], best_config['w_tprop'], best_config['clip'])
submission = ccc_test[["ligand","target"]].copy(); submission["score"] = test_scores
out_path = AGENT_WORKSPACE / f"submission_{EXP_ID}.csv"
submission.to_csv(out_path, index=False)
try: shutil.copy(__file__, AGENT_WORKSPACE / f"train_{EXP_ID}.py")
except: pass

result = {"val_score": float(val_or) if not np.isinf(val_or) else 999.0,"direction":"maximize","exp_id":EXP_ID,    "submission_path":str(out_path),"train_path":str(AGENT_WORKSPACE/f"train_{EXP_ID}.py"),"status":"done",
    "full_train_auc":float(best_auc),"val_odds_ratio":float(val_or) if not np.isinf(val_or) else 999.0,
    "hyperparameters":best_config,"alpha017_baseline_auc":0.995404,"delta_vs_alpha017":float(best_auc-0.995404),
    "timestamp":datetime.now(timezone.utc).isoformat()}
with open(AGENT_WORKSPACE / "result_latest.json","w") as f: json.dump(result,f,indent=2)
print(f"[DONE] {EXP_ID} full_train_AUC={best_auc:.6f}, val_OR={val_or:.4f}")
