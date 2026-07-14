---
task_type: biomlbench
name: open-problems-predict-modality
description: >
  Predict surface protein abundance (~134 proteins) from RNA expression (~13K genes) in BMMC CITE-seq cells.
  Minimize RMSE across all protein-cell pairs.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **H5AD format:** Use `anndata` library to read data.
- **Expression layer:** Data is in `layers['normalized']`, NOT `.X`.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out protein abundance values for the cells in
> `data/test_mod1.h5ad`. It is used **only** by the external grader to score your final `submission.csv`.
> **Never read, open, or use this file in `train.py` or any intermediate evaluation.** Doing so would
> invalidate your score.
>
> **How to evaluate during development:** Hold out 20% of cells from `data/train_mod1.h5ad` /
> `data/train_mod2.h5ad` as a validation set. Train on the remaining 80%, predict protein values for
> the held-out cells, and compute RMSE against the true protein values from `train_mod2.h5ad` —
> never from `private/answers.csv`. See the **Iterative Leaderboard** section below for the exact code.

---

# OpenProblems: Predict Modality (RNA → Protein)

**Metric:** RMSE (Root Mean Squared Error) — lower is better  
**Difficulty:** Hard  
**Data:** BMMC CITE-seq (bone marrow mononuclear cells)  
**GPU:** Recommended 16GB, ~120 min runtime

---

## The Problem

CITE-seq simultaneously measures RNA expression and surface protein abundance in the same single cells. Given paired (RNA, protein) training data, predict protein abundance from RNA alone for held-out test cells. Understanding RNA-to-protein relationships is critical for drug target discovery.

**Input:** Gene expression matrix (log-normalized CP10K) — ~13,000 genes  
**Output:** Surface protein abundance predictions — ~134 proteins  
**Evaluation:** RMSE across all protein-cell pairs — lower is better

---

## Data

### Location
```
data/
├── train_mod1.h5ad    # Training RNA expression (AnnData H5AD)
├── train_mod2.h5ad    # Training protein abundance (matched cells)
├── test_mod1.h5ad     # Test RNA expression (predict protein for these cells)
└── sample_submission.csv
```

### How to prepare data

Data is pre-downloaded. To restore or re-prepare:

```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/manual/open-problems-predict-modality/prepared
cp $PREP/public/train_mod1.h5ad data/ 2>/dev/null
cp $PREP/public/train_mod2.h5ad data/ 2>/dev/null
cp $PREP/public/test_mod1.h5ad data/ 2>/dev/null
cp $PREP/public/sample_submission.csv data/ 2>/dev/null

$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.manual.open_problems_predict_modality.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data format

All files are H5AD (AnnData) format. Expression values are in `layers['normalized']` (NOT `.X`):

```python
import anndata as ad
import numpy as np

train_rna = ad.read_h5ad("data/train_mod1.h5ad")
train_protein = ad.read_h5ad("data/train_mod2.h5ad")
test_rna = ad.read_h5ad("data/test_mod1.h5ad")

# Access expression data
X_train_rna = train_rna.layers['normalized']  # sparse matrix
X_train_protein = train_protein.layers['normalized']
X_test_rna = test_rna.layers['normalized']

# Convert sparse to dense if needed
if hasattr(X_train_rna, 'toarray'):
    X_train_rna = X_train_rna.toarray()

print(f"Train RNA: {train_rna.shape}")     # (cells, ~13K genes)
print(f"Train protein: {train_protein.shape}")  # (cells, ~134 proteins)
print(f"Test RNA: {test_rna.shape}")       # (test_cells, ~13K genes)
print(f"Proteins: {list(train_protein.var_names[:5])}...")  # e.g. CD3, CD4, CD8...
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
2. **Save `submission.csv`** — predicted protein abundances for all cells in `data/test_mod1.h5ad` (columns: `cell_id` + one column per protein name).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** RMSE — lower is better

### Leave-one-site-out CV

The batch label encodes `s{site}d{donor}`: cells were collected from 9 human donors across 4 measurement sites (labs). **Site 4 is entirely absent from training** but accounts for 688/1000 test cells (68.8%).

### Dataset structure

| Batch | Site | Donor | Train cells | Test cells |
|-------|------|-------|-------------|------------|
| s1d1  | 1    | 1     | 4,721       | 26         |
| s1d2  | 1    | 2     | 4,464       | 22         |
| s1d3  | 1    | 3     | 5,484       | 27         |
| s2d1  | 2    | 1     | 9,353       | 44         |
| s2d4  | 2    | 4     | 5,026       | 27         |
| s2d5  | 2    | 5     | 8,206       | 33         |
| s3d1  | 3    | 1     | 8,582       | 44         |
| s3d6  | 3    | 6     | 9,977       | 55         |
| s3d7  | 3    | 7     | 10,362      | 34         |
| s4d1  | 4    | 1     | **0**       | 228        |
| s4d8  | 4    | 8     | **0**       | 176        |
| s4d9  | 4    | 9     | **0**       | 284        |
| **Total** |  |      | **66,175**  | **1,000**  |

Donor 1 appears at all 4 sites (cross-site replicate). Donors 2–7 each appear at exactly one training site. Donors 8–9 appear only at site 4 (test-only, fully unseen).

### Validation: hold out site 1, report per-donor RMSE

Hold out **all of site 1** (batches `s1d1`, `s1d2`, `s1d3`) as the validation set. Train on sites 2 and 3. This validation set mirrors the test set's structure: one donor seen elsewhere in training (donor 1) plus two donors that are fully new to the model.

| Val batch | Donor | Train analogue | Test analogue |
|-----------|-------|----------------|---------------|
| s1d1 | donor 1 (appears at s2d1, s3d1 in train) | donor seen elsewhere | s4d1 (donor 1 at site 4) |
| s1d2 | donor 2 (no other site) | fully new donor | s4d8 (fully new donor) |
| s1d3 | donor 3 (no other site) | fully new donor | s4d9 (fully new donor) |

| Split | Train cells | Val cells |
|-------|-------------|-----------|
| train = sites 2 + 3 | 51,506 | — |
| val = site 1 | — | 14,669 |

```python
import anndata as ad
import numpy as np

train_rna     = ad.read_h5ad("data/train_mod1.h5ad")
train_protein = ad.read_h5ad("data/train_mod2.h5ad")

VAL_BATCHES = ["s1d1", "s1d2", "s1d3"]

val_mask = train_rna.obs["batch"].isin(VAL_BATCHES)
tr_mask  = ~val_mask

tr_rna,  val_rna  = train_rna[tr_mask],  train_rna[val_mask]
tr_prot, val_prot = train_protein[tr_mask], train_protein[val_mask]

# ... train on tr_rna -> tr_prot, predict val_rna -> val_prot_pred ...

y_val = val_prot.layers["normalized"].toarray()

# Overall site-1 RMSE
overall_rmse = np.sqrt(np.mean((y_val - val_prot_pred) ** 2))
print(f"Site-1 val RMSE: {overall_rmse:.4f}")

# Per-donor stratified RMSE
val_batches = val_rna.obs["batch"].values
for b in VAL_BATCHES:
    m = val_batches == b
    rmse_b = np.sqrt(np.mean((y_val[m] - val_prot_pred[m]) ** 2))
    print(f"  {b} RMSE: {rmse_b:.4f}  (n={m.sum()})")
```

**Report to the ClawLab leaderboard:**
1. **Overall site-1 RMSE** (primary metric, used for ranking)
2. **Per-donor stratified RMSE**: `s1d1`, `s1d2`, `s1d3` separately

---

## Final Submission

After identifying your best approach, **retrain on ALL of `data/train_mod1.h5ad` / `train_mod2.h5ad`** which includes all sites and donors before predicting on `data/test_mod1.h5ad`.

```python
# Final training — full training set
full_rna = ad.read_h5ad("data/train_mod1.h5ad")
full_prot = ad.read_h5ad("data/train_mod2.h5ad")
test_rna = ad.read_h5ad("data/test_mod1.h5ad")
# ... retrain on full data, predict on test_rna ...

import pandas as pd
protein_names = list(full_prot.var_names)
submission = pd.DataFrame(predictions, index=test_rna.obs_names, columns=protein_names)
submission.reset_index(names="cell_id").to_csv("submission.csv", index=False)
```

The final submission is graded once against held-out test protein values using RMSE.

---

## Output Format

`submission.csv` with `cell_id` + one column per protein:
```csv
cell_id,CD3,CD4,CD8,CD19,...
AAACCCAAGAAACTAG-1,0.123,0.456,0.789,...
...
```

Column names must exactly match `train_protein.var_names`.

---



---

## Background

# Predict Modality Task

## Overview
Predict Modality: Predicting the profiles of one modality (e.g. protein abundance) from another (e.g. mRNA expression).

## Task Description
Experimental techniques to measure multiple modalities within the same single cell are increasingly becoming available. 
The demand for these measurements is driven by the promise to provide a deeper insight into the state of a cell. 
Yet, the modalities are also intrinsically linked. We know that DNA must be accessible (ATAC data) to produce mRNA 
(expression data), and mRNA in turn is used as a template to produce protein (protein abundance). These processes 
are regulated often by the same molecules that they produce: for example, a protein may bind DNA to prevent the production 
of more mRNA. Understanding these regulatory processes would be transformative for synthetic biology and drug target discovery. 
Any method that can predict a modality from another must have accounted for these regulatory processes, but the demand for 
multi-modal data shows that this is not trivial.

## Dataset: BMMC CITE-seq
This task uses a Bone Marrow Mononuclear Cells (BMMC) CITE-seq dataset. CITE-seq (Cellular Indexing of Transcriptomes and Epitopes by Sequencing) is a method that simultaneously measures:
- **Modality 1 (RNA)**: Gene expression profiles (mRNA abundance)
- **Modality 2 (Protein)**: Surface protein abundance measured using oligonucleotide-labeled antibodies

The dataset contains:
- **Training data**: Paired RNA and protein measurements from the same cells
- **Test data**: RNA measurements only (participants must predict the corresponding protein measurements)

Key statistics:
- Number of genes (RNA): ~13,000
- Number of proteins: ~134
- Number of cells in training set: varies
- Number of cells in test set: varies

## Data Format
All data is provided in H5AD (AnnData) format:
- `train_mod1.h5ad`: Training RNA expression data
- `train_mod2.h5ad`: Training protein abundance data (matched cells with train_mod1)
- `test_mod1.h5ad`: Test RNA expression data

The data has been preprocessed:
- Log-transformed
- Normalized to counts per 10,000 (CP10K)
- Expression values are stored in the `layers['normalized']` attribute (not in `.X`)

## Evaluation Metric
The task uses **Root Mean Squared Error (RMSE)** calculated across all protein-cell pairs. The score is computed as:
```
RMSE = sqrt(mean((y_true - y_pred)^2))
```
where:
- `y_true`: Ground truth protein abundance values
- `y_pred`: Predicted protein abundance values

Lower RMSE indicates better performance.

## Input/Output Specification

### Input
Participants receive:
1. `train_mod1.h5ad`: RNA expression data for training cells
2. `train_mod2.h5ad`: Protein abundance data for the same training cells
3. `test_mod1.h5ad`: RNA expression data for test cells

### Output
Participants must submit a CSV file (`submission.csv`) with the following columns:
- `cell_id`: Cell identifier matching the order in test_mod1
- Remaining columns: One column per protein, containing predicted abundance values

The column names must exactly match the protein names (var_names) from train_mod2.

### Sample Submission Format
```csv
cell_id,CD3,CD4,CD8,CD19,CD20,...
AAACCCAAGAAACTAG-1,0.123,0.456,0.789,0.012,0.345,...
AAACCCAAGAAACTAT-1,0.234,0.567,0.890,0.123,0.456,...
...
```

---
