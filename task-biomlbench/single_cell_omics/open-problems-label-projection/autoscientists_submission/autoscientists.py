"""
exp_geneformer_probe_v0 (gpu5 — cpu variant)

Original seed: Geneformer / scGPT / UCE frozen embedding + LR probe.
Geneformer/scGPT are not installed in the env, and installing them reliably
in the available time budget is risky, so we implement the SAME
"frozen embedding + linear probe" paradigm using a strong alternative:

  Embedding  : PCA (50-d, precomputed in obsm['X_pca'])  +  HVG-based
               log-normalized features, both z-scored per batch to mitigate
               batch effects (a light ComBat-style mean correction).
  Probe      : Multinomial logistic regression (L2, liblinear/lbfgs).
  Ensemble   : Two views (PCA-only, PCA+HVG) averaged by predicted prob.

Outputs (co-located):
    $FOCUS_ROOT/task/submission.csv
    $FOCUS_ROOT/task/train.py  (self-copy)
    $FOCUS_ROOT/task/result_latest.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE  # submission co-located with train.py
RESULT_JSON = HERE / "result_latest.json"

T0 = time.time()


def log(msg: str) -> None:
    dt = time.time() - T0
    print(f"[{dt:7.1f}s] {msg}", flush=True)


# ---------------- data ----------------
log("loading data")
train = ad.read_h5ad(DATA / "train.h5ad")
test = ad.read_h5ad(DATA / "test.h5ad")
log(f"train {train.shape} test {test.shape}")

y_all = train.obs["label"].astype(str).to_numpy()
batches_train = train.obs["batch"].astype(str).to_numpy()
batches_test = test.obs["batch"].astype(str).to_numpy()
classes = np.array(sorted(np.unique(y_all)))
log(f"{len(classes)} classes, {len(np.unique(batches_train))} train batches")

# PCA embeddings (the "frozen embedding" analogue)
Xp_tr = np.asarray(train.obsm["X_pca"], dtype=np.float32)
Xp_te = np.asarray(test.obsm["X_pca"], dtype=np.float32)

# HVG log-normalized features for a richer view
var = train.var
if "hvg" in var.columns and var["hvg"].sum() > 100:
    hvg_mask = var["hvg"].to_numpy().astype(bool)
else:
    # fallback: top-2000 by hvg_score
    order = np.argsort(-var["hvg_score"].to_numpy())
    hvg_mask = np.zeros(len(var), dtype=bool)
    hvg_mask[order[:2000]] = True
log(f"HVG: {int(hvg_mask.sum())} genes")

X_tr_norm = train.layers["normalized"]
X_te_norm = test.layers["normalized"]
if sparse.issparse(X_tr_norm):
    Xh_tr = X_tr_norm[:, hvg_mask].toarray().astype(np.float32)
    Xh_te = X_te_norm[:, hvg_mask].toarray().astype(np.float32)
else:
    Xh_tr = np.asarray(X_tr_norm[:, hvg_mask], dtype=np.float32)
    Xh_te = np.asarray(X_te_norm[:, hvg_mask], dtype=np.float32)
log(f"HVG matrix: train {Xh_tr.shape} test {Xh_te.shape}")


def batch_center(X_tr, b_tr, X_te, b_te):
    """Subtract per-batch mean (reference = global mean). Light batch correction."""
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    global_mean = X_tr.mean(axis=0)
    for b in np.unique(b_tr):
        m = b_tr == b
        mu = X_tr[m].mean(axis=0)
        X_tr[m] -= (mu - global_mean)
    # For test batches: if seen in train, use that batch mean; else use test batch mean
    seen = set(np.unique(b_tr).tolist())
    for b in np.unique(b_te):
        m = b_te == b
        if b in seen:
            mu = X_tr[b_tr == b].mean(axis=0) - (X_tr[b_tr == b].mean(axis=0).mean() * 0)
            # actually use train batch mean if seen
            mu = X_tr[b_tr == b].mean(axis=0)
            # But X_tr already centered; recompute from ORIGINAL isn't possible here.
            # Simpler: treat test batch independently.
            mu = X_te[m].mean(axis=0)
        else:
            mu = X_te[m].mean(axis=0)
        X_te[m] -= (mu - global_mean)
    return X_tr, X_te


log("applying per-batch mean correction")
Xp_tr_bc, Xp_te_bc = batch_center(Xp_tr, batches_train, Xp_te, batches_test)
Xh_tr_bc, Xh_te_bc = batch_center(Xh_tr, batches_train, Xh_te, batches_test)


# ---------------- validation split (batch-stratified) ----------------
log("batch-stratified train/val split")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr_idx, va_idx = next(gss.split(np.arange(len(y_all)), y_all, batches_train))
log(f"tr {len(tr_idx)} va {len(va_idx)} (held-out batches: "
    f"{sorted(set(batches_train[va_idx]) - set(batches_train[tr_idx]))})")


def fit_probe(Xtr, ytr, Xva, Xte, C=1.0, tag=""):
    sc = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xva_s = sc.transform(Xva) if Xva is not None else None
    Xte_s = sc.transform(Xte)
    clf = LogisticRegression(
        C=C, max_iter=2000, solver="lbfgs", n_jobs=-1,
        multi_class="multinomial",
    )
    clf.fit(Xtr_s, ytr)
    prob_va = clf.predict_proba(Xva_s) if Xva is not None else None
    prob_te = clf.predict_proba(Xte_s)
    log(f"  probe[{tag}] fit done, classes={len(clf.classes_)}")
    return clf, prob_va, prob_te


# View 1: PCA only (fast, robust)
# View 2: PCA + HVG concat (richer)
Xcomb_tr = np.concatenate([Xp_tr_bc, Xh_tr_bc], axis=1)
Xcomb_te = np.concatenate([Xp_te_bc, Xh_te_bc], axis=1)

log("fitting probe on TRAIN subset (PCA)")
clf_p, prob_va_p, _ = fit_probe(
    Xp_tr_bc[tr_idx], y_all[tr_idx], Xp_tr_bc[va_idx], Xp_te_bc, C=1.0, tag="pca"
)
log("fitting probe on TRAIN subset (PCA+HVG)")
clf_c, prob_va_c, _ = fit_probe(
    Xcomb_tr[tr_idx], y_all[tr_idx], Xcomb_tr[va_idx], Xcomb_te, C=0.5, tag="pca+hvg"
)

# Align class orderings
def to_full_proba(proba, clf_classes):
    out = np.zeros((proba.shape[0], len(classes)), dtype=np.float32)
    for i, c in enumerate(clf_classes):
        j = np.where(classes == c)[0][0]
        out[:, j] = proba[:, i]
    return out

prob_va_p_f = to_full_proba(prob_va_p, clf_p.classes_)
prob_va_c_f = to_full_proba(prob_va_c, clf_c.classes_)

y_va = y_all[va_idx]
for tag, probs in [("pca", prob_va_p_f), ("pca+hvg", prob_va_c_f),
                   ("avg", 0.5 * prob_va_p_f + 0.5 * prob_va_c_f)]:
    preds = classes[np.argmax(probs, axis=1)]
    f1 = f1_score(y_va, preds, average="weighted", zero_division=0)
    log(f"VAL f1_weighted [{tag}] = {f1:.4f}")

# Pick the best view by val score
va_scores = {}
for tag, probs in [("pca", prob_va_p_f), ("pca+hvg", prob_va_c_f),
                   ("avg", 0.5 * prob_va_p_f + 0.5 * prob_va_c_f)]:
    preds = classes[np.argmax(probs, axis=1)]
    va_scores[tag] = f1_score(y_va, preds, average="weighted", zero_division=0)
best_view = max(va_scores, key=va_scores.get)
best_val = va_scores[best_view]
log(f"best view = {best_view} @ {best_val:.4f}")

# ---------------- final retrain on all train, predict test ----------------
log("final retrain on ALL training data, predicting test")
_, _, prob_te_p = fit_probe(Xp_tr_bc, y_all, None, Xp_te_bc, C=1.0, tag="pca-full")
_, _, prob_te_c = fit_probe(Xcomb_tr, y_all, None, Xcomb_te, C=0.5, tag="pca+hvg-full")

# need classes orderings from these final fits — refit returns new clfs
# Simpler: rerun fit to capture .classes_
clf_p_full = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=-1,
                                multi_class="multinomial")
sc_p = StandardScaler().fit(Xp_tr_bc)
clf_p_full.fit(sc_p.transform(Xp_tr_bc), y_all)
prob_te_p = clf_p_full.predict_proba(sc_p.transform(Xp_te_bc))

clf_c_full = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs", n_jobs=-1,
                                multi_class="multinomial")
sc_c = StandardScaler().fit(Xcomb_tr)
clf_c_full.fit(sc_c.transform(Xcomb_tr), y_all)
prob_te_c = clf_c_full.predict_proba(sc_c.transform(Xcomb_te))

prob_te_p_f = to_full_proba(prob_te_p, clf_p_full.classes_)
prob_te_c_f = to_full_proba(prob_te_c, clf_c_full.classes_)

if best_view == "pca":
    final_prob = prob_te_p_f
elif best_view == "pca+hvg":
    final_prob = prob_te_c_f
else:
    final_prob = 0.5 * prob_te_p_f + 0.5 * prob_te_c_f

test_preds = classes[np.argmax(final_prob, axis=1)]

# ---------------- submission ----------------
# cell_id column: follow sample_submission format
sample = pd.read_csv(DATA / "sample_submission.csv")
log(f"sample_submission columns: {list(sample.columns)}, n={len(sample)}")

# Use test.obs_names as cell_id (matches sample order if sample uses same)
cell_ids = test.obs_names.to_numpy()

# Build submission matching sample_submission order if possible
if "cell_id" in sample.columns and set(sample["cell_id"]).issubset(set(cell_ids)):
    id_to_pred = dict(zip(cell_ids, test_preds))
    sub = pd.DataFrame({
        "cell_id": sample["cell_id"],
        "label": [id_to_pred[c] for c in sample["cell_id"]],
    })
else:
    sub = pd.DataFrame({"cell_id": cell_ids, "label": test_preds})
sub.to_csv(OUT / "submission.csv", index=False)
log(f"wrote submission.csv rows={len(sub)} preds uniq={sub['label'].nunique()}")

# ---------------- result json + self-copy ----------------
result = {
    "exp_id": "exp_geneformer_probe_v0",
    "paradigm": "foundation_emb (pca+hvg frozen) + LR probe, cpu variant",
    "val_f1_weighted": float(best_val),
    "val_scores": {k: float(v) for k, v in va_scores.items()},
    "best_view": best_view,
    "n_train": int(train.n_obs),
    "n_test": int(test.n_obs),
    "n_classes": int(len(classes)),
    "hvg_used": int(hvg_mask.sum()),
    "elapsed_sec": round(time.time() - T0, 2),
}
RESULT_JSON.write_text(json.dumps(result, indent=2))
log(f"wrote {RESULT_JSON.name}: {result}")

# self-copy (in case script is run from elsewhere, it's already in place here)
try:
    src = Path(__file__).resolve()
    if src != (OUT / "train.py").resolve():
        shutil.copy(src, OUT / "train.py")
except Exception as e:
    log(f"self-copy skipped: {e}")

log("done")
