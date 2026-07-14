---
task_type: biomlbench
name: open-problems-single-cell-perturbations
description: >
  Predict differential gene expression (clipped_sign_log10_pval) for 144 drug compounds
  across B cells and Myeloid cells. Minimize Mean Rowwise RMSE (MRRMSE).
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Target column:** `clipped_sign_log10_pval` (clipped to [-4, 4]).
- **Test is B cells + Myeloid cells only** — train on T/NK/Treg cell data primarily.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out differential expression profiles for the 151
> compound×cell-type pairs in `data/id_map.csv` (B cells + Myeloid cells). It is used **only** by
> the external grader to score your final `submission.csv`. **Never read, open, or use this file in
> `train.py` or any intermediate evaluation.** Doing so would invalidate your score.
>
> **How to evaluate during development:** Hold out 20% of B cell + Myeloid cell rows from
> `data/de_train.parquet` as a proxy validation set. Train on the remaining rows, predict DE profiles
> for the held-out rows, and compute MRRMSE against those held-out values from `de_train.parquet` —
> never from `private/answers.csv`. Note that the real test generalises across cell types; random
> within-cell-type splits underestimate this difficulty. See the **Iterative Leaderboard** section
> below for the exact code.

---

# OpenProblems: Single-Cell Perturbation Prediction

**Metric:** Mean Rowwise RMSE (MRRMSE) — lower is better  
**Difficulty:** Hard  
**Data:** PBMC perturbation data, 5317 genes, 144 compounds  
**GPU:** Recommended 8GB, ~120 min runtime

---

## The Problem

Given how 144 compounds affect gene expression in T cells, NK cells, and regulatory T cells (training), predict the differential expression response for the **same compounds in B cells and Myeloid cells** (test). This tests the ability to generalize compound effects across cell types.

**Input:** Compound identity + cell type (+ training DE profiles for other cell types)  
**Output:** `clipped_sign_log10_pval` for 5317 genes (per compound × cell type combination)  
**Evaluation:** MRRMSE — mean of per-row RMSE values

---

## Data

### Location
```
data/
├── de_train.parquet       # Training differential expression data
├── sc_train.h5ad          # Raw single-cell training data (optional, for advanced models)
├── id_map.csv             # Test set: compound × cell type combinations
└── sample_submission.csv  # Template (id + 5317 gene columns)
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

# Copy from biomlbench prepared data if available
PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/manual/open-problems-single-cell-perturbations/prepared
cp $PREP/public/de_train.parquet data/
cp $PREP/public/id_map.csv data/
cp $PREP/public/sample_submission.csv data/
cp $PREP/public/sc_train.h5ad data/ 2>/dev/null || echo "sc_train.h5ad not available (optional)"

# Or run biomlbench prepare
$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.manual.open_problems_single_cell_perturbations.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data format

```python
import pandas as pd

de_train = pd.read_parquet("data/de_train.parquet")
# Rows: (cell_type, compound) combinations
# Columns: 5317 gene names + metadata (sm_name, cell_type, etc.)
# Target column: 'clipped_sign_log10_pval' per gene

id_map = pd.read_csv("data/id_map.csv")
# Columns: id, sm_name (compound), cell_type
# 151 test rows (compounds × {B cells, Myeloid cells})

sample_sub = pd.read_csv("data/sample_submission.csv")
# Columns: id, AAK1, AAMP, ..., ZZEF1 (5317 genes)
gene_cols = [c for c in sample_sub.columns if c != 'id']
```

---

## How to Run

**`train.py` is NOT provided. You must write it from scratch.**

Your workflow:

1. **Write `train.py`** — implement a model that trains on the provided data split,
   evaluates on the val split, prints the val metric, and saves `submission.csv`.
2. **Run it** to get a baseline val score:
   ```bash
   PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
   cd $FOCUS_ROOT/task
   $PYTHON train.py
   ```
3. **Report the val score** to the ClawLab leaderboard (see Iterative Leaderboard section below).
4. **Iterate** — improve `train.py`, re-run, report each new score. Save each version as
   `train_v{N}.py` so you can compare and recover prior approaches.
5. **Select your best version** — once you have tried multiple approaches, pick the one
   with the highest val score, retrain it cleanly, and produce the final `submission.csv`.
6. **Save `submission.csv`** to `$FOCUS_ROOT/task/submission.csv` before the wall-clock
   limit expires — this is the file the grader scores.
7. **Save `train.py`** to `$FOCUS_ROOT/task/train.py` alongside `submission.csv` — this
   must be the exact script used to produce the final `submission.csv`, with all data paths
   hardcoded or resolved from the `data/` directory so the run is fully reproducible.

`train.py` must do three things on every run:
1. **Evaluate locally** — train on the data split, predict on the val split, print the val metric.
2. **Save `submission.csv`** — predicted DE values for all rows in `data/id_map.csv` (columns: `id` + one column per gene, 5317 genes total).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** MRRMSE (Mean Rowwise Root Mean Squared Error) — lower is better

### Dataset structure and the `split` column

`de_train.parquet` contains 402 rows across 4 cell types. The `split` column encodes the original Kaggle competition design and tells you exactly how to use each row:

| Cell type     | split=train | split=public_test | split=control | Total |
|---------------|-------------|-------------------|---------------|-------|
| T cells       | 138         | 0                 | 2             | 140   |
| NK cells      | 137         | 0                 | 2             | 139   |
| B cells       | 12          | 49                | 2             | 63    |
| Myeloid cells | 11          | 47                | 2             | 60    |

- **`split=train`** (298 rows): the rows released to Kaggle participants as labelled training data. All T/NK rows are here, plus 12+11 B/Myeloid rows for a small set of compounds that had partial B/Myeloid labels in the competition.
- **`split=public_test`** (96 rows): B/Myeloid rows that were the Kaggle *public leaderboard* test — ground truth was withheld during the competition but is available in `de_train.parquet` in this post-competition release. The 49+47 compounds here are **completely disjoint from `id_map.csv`**.
- **`split=control`** (8 rows): 2 compounds (Belinostat, Dabrafenib) measured across all 4 cell types. These are QC anchors in the original experiment, but they are the **only compounds with both T/NK and B/Myeloid labels** — the exact cross-cell-type mapping the model must learn. Include them in training.
- **`split=private_test`** (0 rows): the Kaggle private leaderboard compounds have no B/Myeloid rows in `de_train.parquet`. They exist only as T/NK rows under `split=train` — these are the 76 compounds in `id_map.csv`.

**`id_map.csv` = the Kaggle private test set.** The 76 test compounds have zero B/Myeloid data anywhere in `de_train.parquet` and zero overlap with `split=public_test` compounds. They appear only as T/NK rows with `split=train`. The task is: given T/NK responses for a compound, predict its B/Myeloid responses.

Compound sets are fully non-overlapping across the three groups:

| Compound group | Count | In de_train with B/Myeloid? | In id_map? |
|---|---|---|---|
| T/NK only, split=train | 76 | No | **Yes — these are the test compounds** |
| B/Myeloid partial, split=train | 12 | Yes (train labels) | No |
| B/Myeloid, split=public_test | 49 | Yes (post-competition labels) | No |
| Control | 2 | Yes (all cell types) | No |

### Use `split=public_test` as validation

`split=public_test` is the correct validation set — it has the same structure as the test: T/NK data in training, B/Myeloid to predict, with ground truth available. Use `split=train` rows as training:

```python
import pandas as pd
import numpy as np

de_train = pd.read_parquet("data/de_train.parquet")
gene_cols = [c for c in de_train.columns if c not in
             ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "split"]]

tr_df  = de_train[de_train["split"].isin(["train", "control"])]  # 306 rows: T/NK + B/Myeloid (incl. controls)
val_df = de_train[de_train["split"] == "public_test"]            # 96 rows: B/Myeloid with known labels

# ... train model on tr_df, predict B/Myeloid for val_df compounds
# (val compounds have only T/NK rows in tr_df — no B/Myeloid leakage) ...
# val_preds: array shape (len(val_df), len(gene_cols))
y_true = val_df[gene_cols].values
rmse_per_row = np.sqrt(np.mean((y_true - val_preds) ** 2, axis=1))
score = np.mean(rmse_per_row)
print(f"Val MRRMSE: {score:.4f}")
```

Report this val score to the ClawLab leaderboard after each iteration.

---

## Final Submission

After identifying your best approach, **retrain on all of `de_train.parquet`** (split=train + split=control + split=public_test) before predicting on the test pairs in `id_map.csv`. Including public_test and control rows gives the model more B/Myeloid signal for final training.

```python
# Final training — full de_train.parquet
full_train = pd.read_parquet("data/de_train.parquet")
id_map = pd.read_csv("data/id_map.csv")  # id, sm_name, cell_type
# ... retrain on full_train, predict for each row in id_map ...

import pandas as pd
submission = pd.DataFrame(test_preds, columns=gene_cols)
submission.insert(0, "id", id_map["id"])
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against held-out differential expression values using MRRMSE.

---

## Output Format

`submission.csv` with `id` + 5317 gene columns (use exact names from `sample_submission.csv`):
```csv
id,AAK1,AAMP,AASDH,...,ZZEF1
0,0.12,-0.45,0.23,...
1,-0.34,0.67,-0.12,...
```

---



---

## Background

# OpenProblems – Single-Cell Perturbation Prediction

## Overview

This task aims to predict how small molecules change gene expression in different cell types. You will predict differential expression values for cell-type and compound combinations not seen during training.

## Task Description

Given training data showing how various compounds affect gene expression in different cell types, predict the differential expression response for new compound-cell type combinations in the test set.

## Dataset

The dataset contains differential expression measurements from a single-cell perturbational experiment in human peripheral blood mononuclear cells (PBMCs).

### Experimental Setup
- **144 compounds** from the LINCS Connectivity Map dataset
- **Treatment duration**: 24 hours
- **Cell types**: B cells and Myeloid cells (test set), plus T cells, NK cells, and regulatory T cells (training set)
- **Three healthy human donors**

### Data Splits
- **Training**: All compounds in T cells, NK cells, regulatory T cells + subset of compounds in B/Myeloid cells
- **Test**: Remaining compounds in B cells and Myeloid cells

## Input Files

You will receive the following files:

### `de_train.parquet`
Training differential expression data with:
- **Rows**: Cell type and compound combinations  
- **Columns**: Gene expression values (5317 genes) + metadata
- **Values**: Differential expression scores (clipped_sign_log10_pval metric)

The training data contains multiple layers of differential expression statistics:
- `clipped_sign_log10_pval`: Primary target (`sign_log10_pval` values clipped between -4 and 4)
- `sign_log10_pval`: Unclipped differential expression values (-log10(p-value) × sign(log_fold_change))
- `logFC`: Log fold change values
- `P.Value`: Raw p-values  
- `adj.P.Value`: Adjusted p-values

### `sc_train.h5ad` (It's not required to use this data but could be helpful for certain approaches.)
Raw single-cell training data in AnnData format for advanced modeling approaches. It was from this data that the differential expression data was extracted (for training samples). adata.X is in raw counts format. adata.obs contains cell type and compount information.  adata.var_names contains the gene names that are also used in the `de_train.parquet` file.

### `id_map.csv`
Maps test set IDs to their corresponding cell types and compounds:

```
id,sm_name,cell_type
0,TIE2 Kinase Inhibitor,B cells
1,TIE2 Kinase Inhibitor,Myeloid cells
...
```

### `sample_submission.csv`
Template showing the required submission format with all gene columns initialized to 0.0.

## Target Variable

Predict the **clipped_sign_log10_pval** values for each gene, calculated as:

```
clipped_sign_log10_pval = clip(−log10(p-value) × sign(log_fold_change), -4, 4)
```

Where:
- Values are clipped to the range [-4, 4] for stability
- Positive values indicate upregulation in treatment vs control
- Negative values indicate downregulation in treatment vs control
- Magnitude reflects statistical significance (higher absolute values = more significant)

## Submission Format

Submit a CSV file with:
- **`id`** column: Test sample IDs (0 to 150)
- **Gene columns**: Predicted differential expression values for 5317 genes
- **Example**: `id,AAK1,AAMP,AASDH,...,ZZEF1`

## Evaluation

Performance is measured using **Mean Rowwise Root Mean Squared Error (MRRMSE)**:
1. Calculate RMSE for each gene across all test samples
2. Take the mean of these gene-wise RMSE values
3. Lower scores are better

Here is the code for the metric:

```python

def mean_rowwise_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    # Calculate squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Calculate mean squared error for each row (across genes)
    mse_per_row = np.mean(squared_diff, axis=1)
    
    # Calculate RMSE for each row
    rmse_per_row = np.sqrt(mse_per_row)
    
    # Calculate mean across all rows
    mrrmse = np.mean(rmse_per_row)
    
    return mrrmse

# Prepare data for evaluation
gene_cols = ['AAK1', 'AAMP', 'AASDH', ..., 'ZYX', 'ZZEF1']
y_true = answers[gene_cols].values
y_pred = submission[gene_cols].values

# Calculate MRRMSE
score = mean_rowwise_rmse(y_true, y_pred)

```

This metric treats all genes equally regardless of expression level.

## Scientific Context

This task addresses a key challenge in drug discovery: predicting how compounds affect gene expression in different cell types without expensive experiments. Success could accelerate:
- Drug discovery and development
- Understanding of drug mechanisms
- Prediction of side effects
- Personalized medicine approaches

## Getting Started

1. Load the training data: `pd.read_parquet('de_train.parquet')`
2. Explore gene expression patterns across compounds and cell types
3. Build a model to predict test set responses
4. Format predictions using the sample submission template

The task requires predicting compound effects in B cells and Myeloid cells based on training data from multiple cell types and partial data from the target cell types.

---
