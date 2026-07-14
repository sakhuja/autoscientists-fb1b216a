"""
exp_beta_003: Residual MLP with 512 TruncatedSVD components
Team: team_beta
Agent: biomlbmodality_8_gpu4

Approach:
- TruncatedSVD 512 components (memory-efficient; avoids centering the full matrix)
  (vs 256 in exp_alpha_003 which got 0.8654)
- Residual MLP architecture: 512 -> [1024 -> 512 -> 256] with skip connections
- BatchNorm, Dropout 0.3
- No adversarial head (confirmed ineffective by exp_beta_002)
- CosineAnnealingLR, early stopping patience=20

Key differences from exp_alpha_003 (champion, 0.8654):
- 512 SVD components (2x more)
- Residual connections for training stability
- TruncatedSVD instead of PCA (memory-efficient, 40GB RAM limit)

Val: leave-site-1-out (hold out s1d1, s1d2, s1d3; train on sites 2+3)
"""

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import TruncatedSVD
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone

# ---- Config ----
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"
EXP_ID = "exp_beta_003"
N_PCA = 512
HIDDEN_DIMS = [1024, 512, 256]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 512
N_EPOCHS = 120
PATIENCE = 20
VAL_BATCHES = ["s1d1", "s1d2", "s1d3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[{EXP_ID}] Residual MLP + 512 TruncatedSVD", flush=True)
print(f"Config: n_pca={N_PCA}, hidden={HIDDEN_DIMS}, dropout={DROPOUT}, epochs={N_EPOCHS}, device={DEVICE}", flush=True)

# ---- Load data ----
print("Loading data...", flush=True)
train_rna = ad.read_h5ad(DATA_DIR / "train_mod1.h5ad")
train_protein = ad.read_h5ad(DATA_DIR / "train_mod2.h5ad")
test_rna = ad.read_h5ad(DATA_DIR / "test_mod1.h5ad")

print(f"Train RNA: {train_rna.shape}", flush=True)
print(f"Train protein: {train_protein.shape}", flush=True)
print(f"Test RNA: {test_rna.shape}", flush=True)

# ---- Split: leave-site-1-out ----
val_mask = train_rna.obs["batch"].isin(VAL_BATCHES)
tr_mask = ~val_mask

tr_rna = train_rna[tr_mask]
val_rna = train_rna[val_mask]
tr_prot = train_protein[tr_mask]
val_prot = train_protein[val_mask]

print(f"Train cells: {tr_mask.sum()}, Val cells: {val_mask.sum()}", flush=True)

# ---- Extract matrices ----
X_tr = tr_rna.layers["normalized"]
if hasattr(X_tr, "toarray"):
    X_tr = X_tr.toarray()
X_val = val_rna.layers["normalized"]
if hasattr(X_val, "toarray"):
    X_val = X_val.toarray()
X_test = test_rna.layers["normalized"]
if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()

y_tr = tr_prot.layers["normalized"]
if hasattr(y_tr, "toarray"):
    y_tr = y_tr.toarray()
y_val = val_prot.layers["normalized"]
if hasattr(y_val, "toarray"):
    y_val = y_val.toarray()

protein_names = list(train_protein.var_names)
n_proteins = len(protein_names)
print(f"n_proteins={n_proteins}, n_genes={X_tr.shape[1]}", flush=True)

# ---- TruncatedSVD on training RNA (memory-efficient; no centering of full matrix) ----
print(f"Fitting TruncatedSVD ({N_PCA} components) on training data...", flush=True)
pca = TruncatedSVD(n_components=N_PCA, n_iter=7, random_state=42)
X_tr_pca = pca.fit_transform(X_tr)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
print(f"SVD explained variance: {pca.explained_variance_ratio_.sum():.4f}", flush=True)


# ---- Residual MLP model ----
class ResidualBlock(nn.Module):
    """Residual block: Linear -> BN -> ReLU -> Dropout with skip connection."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Project input if dimensions differ
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.relu(self.bn(self.linear(x))))
        return out + self.skip(x)  # residual connection


class ResidualMLP(nn.Module):
    """Residual MLP for protein prediction from PCA embeddings."""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        # Build residual blocks
        blocks = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(in_dim, h_dim, dropout=dropout))
            in_dim = h_dim
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        out = self.blocks(x)
        return self.output(out)


# ---- Training ----
model = ResidualMLP(N_PCA, HIDDEN_DIMS, n_proteins, dropout=DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
criterion = nn.MSELoss()

X_tr_t = torch.FloatTensor(X_tr_pca).to(DEVICE)
y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
X_val_t = torch.FloatTensor(X_val_pca).to(DEVICE)
y_val_t = torch.FloatTensor(y_val).to(DEVICE)

dataset = TensorDataset(X_tr_t, y_tr_t)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

best_val_rmse = float("inf")
best_epoch = 0
patience_counter = 0
best_state = None

print(f"\nTraining for {N_EPOCHS} epochs...", flush=True)
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()

    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    train_loss = np.mean(train_losses)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_rmse={val_rmse:.4f}", flush=True)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best: {best_val_rmse:.4f} at epoch {best_epoch})", flush=True)
            break

print(f"\nBest val RMSE: {best_val_rmse:.6f} at epoch {best_epoch}", flush=True)

# Load best model for evaluation
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    val_pred_best = model(X_val_t).cpu().numpy()

val_rmse_overall = np.sqrt(np.mean((y_val - val_pred_best) ** 2))
print(f"\nSite-1 val RMSE: {val_rmse_overall:.6f}", flush=True)

val_batches_arr = val_rna.obs["batch"].values
for b in VAL_BATCHES:
    m = val_batches_arr == b
    rmse_b = np.sqrt(np.mean((y_val[m] - val_pred_best[m]) ** 2))
    print(f"  {b} RMSE: {rmse_b:.6f}  (n={m.sum()})", flush=True)

# ---- Retrain on FULL data for submission ----
print("\nRetraining on full data for submission...", flush=True)
X_full = train_rna.layers["normalized"]
if hasattr(X_full, "toarray"):
    X_full = X_full.toarray()
y_full = train_protein.layers["normalized"]
if hasattr(y_full, "toarray"):
    y_full = y_full.toarray()

pca_full = TruncatedSVD(n_components=N_PCA, n_iter=7, random_state=42)
X_full_pca = pca_full.fit_transform(X_full)
X_test_pca_full = pca_full.transform(X_test)
print(f"Full SVD explained variance: {pca_full.explained_variance_ratio_.sum():.4f}", flush=True)

model_full = ResidualMLP(N_PCA, HIDDEN_DIMS, n_proteins, dropout=DROPOUT).to(DEVICE)
optimizer_full = torch.optim.Adam(model_full.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_full = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=best_epoch)
criterion_full = nn.MSELoss()

X_full_t = torch.FloatTensor(X_full_pca).to(DEVICE)
y_full_t = torch.FloatTensor(y_full).to(DEVICE)
full_dataset = TensorDataset(X_full_t, y_full_t)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

for epoch in range(1, best_epoch + 1):
    model_full.train()
    for xb, yb in full_loader:
        optimizer_full.zero_grad()
        pred = model_full(xb)
        loss = criterion_full(pred, yb)
        loss.backward()
        optimizer_full.step()
    scheduler_full.step()
    if epoch % 10 == 0 or epoch == best_epoch:
        print(f"Full retrain epoch {epoch}/{best_epoch}", flush=True)

model_full.eval()
X_test_t = torch.FloatTensor(X_test_pca_full).to(DEVICE)
with torch.no_grad():
    test_pred = model_full(X_test_t).cpu().numpy()

# ---- Save submission ----
submission = pd.DataFrame(test_pred, index=test_rna.obs_names, columns=protein_names)
submission.reset_index(names="cell_id").to_csv("submission.csv", index=False)
print(f"Saved submission.csv with shape {submission.shape}", flush=True)

# Print hyperparameters as JSON for champion extraction
print("=" * 40)
print(json.dumps({
    "exp_id": EXP_ID,
    "n_pca": N_PCA,
    "hidden_dims": HIDDEN_DIMS,
    "dropout": DROPOUT,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "n_epochs_run": N_EPOCHS,
    "best_epoch": best_epoch,
    "patience": PATIENCE,
    "val_rmse": float(val_rmse_overall),
    "val_cells": int(val_mask.sum()),
    "train_cells": int(tr_mask.sum()),
    "n_proteins": n_proteins,
    "n_genes": int(X_tr.shape[1]),
    "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
    "approach": "residual_MLP_512SVD",
    "architecture": "residual_blocks",
}))
print("=" * 40)

print(f"\n[DONE] {EXP_ID} val RMSE = {val_rmse_overall:.6f}")
