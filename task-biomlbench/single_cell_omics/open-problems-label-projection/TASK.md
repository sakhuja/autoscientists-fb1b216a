---
task_type: biomlbench
name: open-problems-label-projection
description: >
  Transfer cell type annotations from reference kidney cortex scRNA-seq to query cells.
  13 cell types (actual data), diabetic kidney disease context. Maximize F1-weighted score.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **13 cell types:** Predicted labels must match training vocabulary exactly.
- **Batch effects:** Training data may span multiple batches — batch correction can help.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out cell type labels for the cells in `data/test.h5ad`.
> It is used **only** by the external grader to score your final `submission.csv`. **Never read, open,
> or use this file in `train.py` or any intermediate evaluation.** Doing so would invalidate your score.
>
> **How to evaluate during development:** Split `data/train.h5ad` into train/val by batch (hold out
> ~20% of batches). Train on the remaining batches, predict labels for the held-out batch cells, and
> compute F1-weighted against `val.obs["label"]` from the training data — never from `private/answers.csv`.
> See the **Iterative Leaderboard** section below for the exact code.

---

# OpenProblems: Label Projection (Cell Type Annotation Transfer)

**Metric:** F1-weighted score — higher is better  
**Difficulty:** Medium  
**Data:** Diabetic kidney disease snRNA-seq, 13 cell types, 24,923 genes  
**GPU:** Recommended 16GB, ~60 min runtime

---

## The Problem

Annotating single-cell RNA-seq data with cell type labels is labor-intensive. Label projection transfers annotations from a labeled reference dataset to an unlabeled query dataset. Here, train on labeled kidney cortex cells (13 cell types) and predict cell type labels for held-out test cells.

**Input:** scRNA-seq gene expression + reference labels for training cells  
**Output:** Cell type label for each test cell (one of 29 types)  
**Evaluation:** F1-weighted score — higher is better

---

## Data

### Location
```
data/
├── train.h5ad           # Training cells with cell type labels (AnnData)
├── test.h5ad            # Test cells without labels
├── label_vocabulary.csv # List of valid cell type labels (29 types)
└── sample_submission.csv
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/manual/open-problems-label-projection/prepared
cp $PREP/public/*.h5ad data/ 2>/dev/null
cp $PREP/public/sample_submission.csv data/ 2>/dev/null

$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.manual.open_problems_label_projection.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data format

```python
import anndata as ad

train = ad.read_h5ad("data/train.h5ad")
test = ad.read_h5ad("data/test.h5ad")

# Training
X_train = train.layers['normalized']  # or train.X
y_train = train.obs['label']           # cell type labels
batches = train.obs['batch']           # batch info

# Test
X_test = test.layers['normalized']     # predict labels for these

# Available embeddings
pca_train = train.obsm['X_pca']       # PCA embeddings
pca_test = test.obsm['X_pca']

# Cell type vocabulary (29 types)
cell_types = sorted(y_train.unique())
print(f"Cell types: {cell_types}")     # e.g. Proximal_tubule, Podocyte, etc.
print(f"Genes: {train.n_vars}")        # 24923 genes
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
2. **Save `submission.csv`** — predicted cell type labels for all cells in `data/test.h5ad` (columns: `cell_id`, `label`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** F1-weighted — higher is better

Split `data/train.h5ad` into train/val by batch to simulate cross-batch generalisation (the test set comes from held-out batches). A random 80/20 cell split also works but batch-stratified is more representative.

```python
import anndata as ad
import numpy as np
from sklearn.metrics import f1_score

adata = ad.read_h5ad("data/train.h5ad")
batches = adata.obs["batch"].unique()
val_batches = np.random.choice(batches, size=max(1, len(batches)//5), replace=False)
val_mask = adata.obs["batch"].isin(val_batches)

tr, val = adata[~val_mask], adata[val_mask]
# ... train on tr, predict on val ...
score = f1_score(val.obs["label"], val_preds, average="weighted")
print(f"Val F1-weighted: {score:.4f}")
```

Report this val score to the ClawLab leaderboard after each iteration.

---

## Final Submission

After identifying your best approach, **retrain on ALL of `data/train.h5ad`** before predicting on `data/test.h5ad`.

```python
# Final training — use full train.h5ad
full_train = ad.read_h5ad("data/train.h5ad")
test = ad.read_h5ad("data/test.h5ad")
# ... retrain on full_train, predict on test ...

import pandas as pd
submission = pd.DataFrame({"cell_id": test.obs_names, "label": test_preds})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against held-out test labels using F1-weighted.

---

## Output Format

```csv
cell_id,label
AAACCCAAGAAACTAG-1,Proximal_tubule
AAACCCAAGAAACTAT-1,Podocyte
...
```

Labels must be from the training vocabulary (exactly matching strings).

---



---

## Background

# Label Projection Task

## Overview
Label projection: Automated cell type annotation from rich, labeled reference data

## Task Description
A major challenge for integrating single cell datasets is creating matching
cell type annotations for each cell. One of the most common strategies for
annotating cell types is referred to as
["cluster-then-annotate"](https://www.nature.com/articles/s41576-018-0088-9)
whereby cells are aggregated into clusters based on feature similarity and
then manually characterized based on differential gene expression or previously
identified marker genes. Recently, methods have emerged to build on this
strategy and annotate cells using
[known marker genes](https://www.nature.com/articles/s41592-019-0535-3).
However, these strategies pose a difficulty for integrating atlas-scale
datasets as the particular annotations may not match.

To ensure that the cell type labels in newly generated datasets match
existing reference datasets, some methods align cells to a previously
annotated [reference dataset](https://academic.oup.com/bioinformatics/article/35/22/4688/54802990)
and then _project_ labels from the reference to the new dataset.

Here, we compare methods for annotation based on a reference dataset.
The datasets consist of two or more samples of single cell profiles that
have been manually annotated with matching labels. These datasets are then
split into training and test batches, and the task of each method is to
train a cell type classifer on the training set and project those labels
onto the test set.

## Dataset: Diabetic Kidney Disease (DKD)
We use the **Diabetic Kidney Disease** dataset from the CellxGene Census, which provides comprehensive single-cell data from human kidney cortex samples. This dataset includes cells from donors with and without diabetic kidney disease, offering insights into disease-associated cellular changes.

### Dataset Overview:
- **Reference**: Wilson et al., 2022 - "Multimodal single cell sequencing implicates chromatin accessibility and genetic background in diabetic kidney disease progression"
- **Tissue**: Human kidney cortex
- **Condition**: Samples from donors with and without diabetic kidney disease
- **Technology**: Single nucleus RNA sequencing (snRNA-seq)
- **Cell types**: Multiple kidney cell types including:
  - Proximal tubule cells (including injured PT_VCAM1+ cells)
  - Distal tubule cells
  - Collecting duct cells
  - Glomerular cells (podocytes, endothelial, mesangial)
  - Immune cells
  - Stromal cells

## Data Format
All data is provided in H5AD (AnnData) format with the following structure:

### Training Data (`train.h5ad`)
- **obs**: 
  - `label`: Cell type annotations
  - `batch`: Batch/sample information
- **var**: Gene metadata (feature_id, feature_name, hvg, hvg_score)
- **layers**:
  - `counts`: Raw count data
  - `normalized`: Log-normalized expression (log(CP10k+1))
- **obsm**: 
  - `X_pca`: Pre-computed PCA embeddings

### Test Data (`test.h5ad`)
- Same structure as training data but without `label` column
- Methods must predict labels for these cells

## Evaluation Metric
We evaluate methods using **F1-weighted score**, which:
- Calculates F1 score for each cell type class
- Weights by the number of true instances for each class
- Provides a balanced measure that accounts for class imbalance

Formula: F1-weighted = Σ(n_i / N) × F1_i

Where:
- n_i = number of samples in class i
- N = total number of samples
- F1_i = F1 score for class i

## Input/Output Specification

### Input
Methods receive:
1. `train.h5ad`: Training data with cell type labels
2. `test.h5ad`: Test data without labels

### Output
Methods must produce a prediction file with:
- Cell identifiers matching the test set
- Predicted cell type label for each cell
- Labels must exactly match the vocabulary from the training set

## Implementation Notes
- Methods can use any layer (`counts` or `normalized`) or the pre-computed PCA
- Batch information is provided but handling batch effects is up to each method
- Gene sets are pre-aligned between train and test sets
- All cells must receive a prediction (no missing values allowed)

## Baseline Performance
The task includes control methods for comparison:
- **Random Labels**: Random assignment from training label distribution
- **Majority Vote**: Assigns the most frequent cell type to all cells
- **True Labels**: Oracle performance (upper bound)

## Important Considerations
1. **Exact label matching**: Predicted labels must exactly match training set vocabulary
2. **No missing predictions**: Every test cell must have a prediction
3. **Deterministic results**: Methods should produce consistent results across runs
4. **Resource constraints**: Methods should complete within reasonable time/memory limits

---
