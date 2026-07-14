---
task_type: biomlbench
name: open-problems-cell-cell-communication-ligand-target
description: >
  Predict ligand-target interactions in triple-negative breast cancer single-cell RNA-seq data.
  Maximize Odds Ratio of top 5% predictions vs. ground truth.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **Prior knowledge:** OmniPath/LIANA prior knowledge (ligand-receptor pairs) can help significantly.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.

---

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out response labels for the 731 ligand-target pairs in
> `adata.uns["ccc_test_pairs"]`. It is used **only** by the external grader to score your final
> `submission.csv`. **Never read, open, or use this file in `train.py` or any intermediate evaluation.**
> Doing so would invalidate your score.
>
> **How to evaluate during development:** Use a random 80/20 split of the 81 labelled pairs in
> `adata.uns["ccc_train"]` (inside `data/tnbc_data.h5ad`). Score the 20% held-out pairs with your
> method and compute the Odds Ratio against those labels — never against `private/answers.csv`.
> The training labels are intentionally small (10% of all pairs) — this task expects an
> unsupervised/statistical scoring approach. See the **Iterative Leaderboard** section below for the
> exact validation code.

---

# OpenProblems: Cell-Cell Communication — Ligand-Target Prediction

**Metric:** Odds Ratio (top 5% predictions vs. ground truth) — higher is better  
**Difficulty:** Hard  
**Data:** Triple-negative breast cancer scRNA-seq (H5AD)  
**GPU:** Recommended 16GB, ~120 min runtime

---

## The Problem

Cell-cell communication (CCC) occurs through ligand-receptor interactions — one cell secretes a ligand that binds receptors on another cell, triggering signaling cascades to downstream target genes. Predicting which ligand-target pairs are active from scRNA-seq data can reveal disease mechanisms and drug targets.

**Input:** scRNA-seq gene expression data + cell type labels + OmniPath prior knowledge  
**Output:** Continuous score for each (ligand, target gene) test pair  
**Evaluation:** Odds Ratio — comparing top 5% predictions to ground truth binary labels

---

## Data

### Location
```
data/
├── tnbc_data.h5ad              # TNBC single-cell RNA-seq data (AnnData)
├── ligand_receptor_resource.csv.gz  # OmniPath ligand-receptor prior knowledge
├── dataset_info.txt            # Dataset metadata
└── sample_submission.csv       # ligand/target/score template
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/manual/open-problems-cell-cell-communication-ligand-target/prepared
cp $PREP/public/*.h5ad data/ 2>/dev/null
cp $PREP/public/*.csv data/ 2>/dev/null

$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.manual.open_problems_cell_cell_communication_ligand_target.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data format

```python
import anndata as ad, pandas as pd

sc_data = ad.read_h5ad("data/sc_data.h5ad")
# sc_data.obs: cell type labels, batch info
# sc_data.var_names: gene names

train_pairs = pd.read_csv("data/train_pairs.csv")
# Columns: ligand, target, label (0/1), potentially cell_type info

test_pairs = pd.read_csv("data/test_pairs.csv")
# Columns: ligand, target (labels withheld)
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
2. **Save `submission.csv`** — scores for all ligand–target pairs in `adata.uns['ccc_test_pairs']` (columns: ligand, target, score).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** Odds Ratio (top 5% predictions vs ground truth) — higher is better

### Dataset structure

| Set | (ligand, target) pairs | Positives | Negatives | Notes |
|-----|------------------------|-----------|-----------|-------|
| `ccc_train` | 81 | 17 | 64 | All pairs disjoint from test |
| `ccc_test_pairs` | 731 | withheld | withheld | All pairs disjoint from train |

- 28 unique ligands appear in both train and test; 28 unique target cell types in train, 29 in test (3 test-only: `Mature.Luminal`, `Myoepithelial`, `PVL.Immature`).
- **Train and test (ligand, target) pairs are completely disjoint** — there is no direct overlap to memorise.
- The method is unsupervised/statistical: score all pairs using gene expression + prior knowledge, then use the 81 training labels only to sanity-check ranking quality.

### Validation

Because train and test pairs are disjoint and the method is unsupervised, a random 80/20 split of the 81 training labels is the only available validation. It is a noisy signal (only ~16 val rows, ~3–4 positives) but sufficient to detect gross failures:

```python
import anndata as ad
import numpy as np
import pandas as pd

adata = ad.read_h5ad("data/tnbc_data.h5ad")
ccc_train = adata.uns["ccc_train"]  # 81 rows: ligand, target, response

val = ccc_train.sample(frac=0.2, random_state=42)
tr  = ccc_train.drop(val.index)

# ... build scoring method using gene expression + prior knowledge (tr labels optional) ...
# val_scores: 1-D array of predicted scores for each row in val

def odds_ratio_top5pct(scores, labels):
    threshold = np.percentile(scores, 95)
    top5 = scores >= threshold
    tp = np.sum(top5 & (labels == 1))
    fp = np.sum(top5 & (labels == 0))
    fn = np.sum(~top5 & (labels == 1))
    tn = np.sum(~top5 & (labels == 0))
    return (tp * tn) / (fp * fn) if fp > 0 and fn > 0 else np.inf

score = odds_ratio_top5pct(val_scores, val["response"].values)
print(f"Val Odds Ratio: {score:.4f}")
```

**Note:** With only ~16 val rows the Odds Ratio is extremely noisy. Treat it as a sanity check, not a reliable optimisation target. The scoring method should be driven primarily by biological prior knowledge (OmniPath ligand-receptor pairs, ligand expression levels, receptor expression in target cell types) rather than by fitting to these 81 labels.

Report this val score to the ClawLab leaderboard after each iteration.

---

## Final Submission

This task uses an unsupervised/statistical scoring approach — there is no model to retrain on all data. Apply your best method to score all pairs in `adata.uns['ccc_test_pairs']`.

```python
adata = ad.read_h5ad("data/tnbc_data.h5ad")
test_pairs = adata.uns["ccc_test_pairs"]  # DataFrame: ligand, target

# ... score all test pairs using your method ...
submission = test_pairs[["ligand", "target"]].copy()
submission["score"] = test_scores
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against held-out interaction labels using Odds Ratio.

---

## Output Format

```csv
ligand,target,score
TGFB1,SMAD2,0.892
EGFR,STAT3,0.123
...
```

---



---

## Background

# Cell-Cell Communication (Ligand-Target) Task

## Overview
Cell-cell Communication (ligand-target): Predicting cell-cell communication from single-cell transcriptomics data using supervised learning

## Task Description
The growing availability of single-cell data has sparked an increased interest in the inference of cell-cell communication (CCC), with an ever-growing number of computational tools developed for this purpose.

Different tools propose distinct preprocessing steps with diverse scoring functions, that are challenging to compare and evaluate. Furthermore, each tool typically comes with its own set of prior knowledge. To harmonize these, [Dimitrov et al, 2022](https://openproblems.bio/bibliography#dimitrov2022comparison) recently developed the [LIANA](https://github.com/saezlab/liana) framework, which was used as a foundation for this task.

The challenges in evaluating the tools are further exacerbated by the lack of a gold standard to benchmark the performance of CCC methods. In an attempt to address this, Dimitrov et al use alternative data modalities, including the spatial proximity of cell types and downstream cytokine activities, to generate an inferred ground truth. However, these modalities are only approximations of biological reality and come with their own assumptions and limitations. In time, the inclusion of more datasets with known ground truth interactions will become available, from which the limitations and advantages of the different CCC methods will be better understood.

**This subtask evaluates the methods' ability to predict interactions, the corresponding cytokines of which, are inferred to be active in the target cell types. This subtask focuses on the prediction of interactions from steady-state, or single-context, single-cell data.**

## Dataset: TNBC Data
We use the **Triple-Negative Breast Cancer (TNBC)** dataset from Wu et al., 2021. This dataset provides single-cell RNA-seq data from human breast cancer samples, offering insights into cell-cell communication within the tumor microenvironment.

### Dataset Overview:
- **Reference**: Wu et al., 2021 - Single-cell analysis of triple-negative breast cancer
- **Tissue**: Human breast cancer
- **Technology**: Single cell RNA sequencing (scRNA-seq)
- **Cell types**: Multiple tumor and immune cell types
- **Ground truth**: Inferred cytokine activities serving as proxy for cell-cell communication

## Data Format
All data is provided in H5AD (AnnData) format with the following structure:

### Input Data
- **obs**:
  - `label`: Cell type annotations
- **var**: Gene metadata
- **layers**:
  - `counts`: Raw count matrix
  - `normalized`: Log-transformed count matrix
- **uns**:
  - `ccc_train`: Training set DataFrame with columns:
    - `ligand`: Gene symbol of the ligand
    - `target`: Target cell type name
    - `response`: Binary (0/1) indicating interaction occurrence
  - `ccc_test_pairs`: Test set DataFrame with columns:
    - `ligand`: Gene symbol of the ligand
    - `target`: Target cell type name
    - (Note: response values are withheld for evaluation)
  - `target_organism`: NCBI taxonomy ID for species conversion

### Data Split
- The dataset is split 10/90 into training and test sets
- Training data includes ground truth responses for model development
- Test set responses are withheld and used for evaluation
- Stratified split ensures balanced class distribution

**NOTE**: We have purposely given a very small number of labels to you. This is because you are NOT expected to actually train a supervised model. The goal is to develop an unsupervised method / statistical approach to predict interactions. We have allowed you a small number of labels so that you can sanity check your method.

### Output Format
Methods must produce predictions for the test set interactions as a CSV file with:
- `ligand`: Gene symbol of the ligand (must match test set)
- `target`: Target cell type name (must match test set)
- `score`: Predicted interaction strength (-inf to +inf)

## Evaluation Metric
We evaluate methods using **Odds Ratio**, which:
- Compares true/false positive ratios in top-ranked predictions vs remaining interactions
- Uses top 5% of predictions by default
- Applies sigmoid transformation for normalization
- Quantifies association strength between method prioritization and positive interactions

Formula: OR = (TP × TN) / (FP × FN)
Where predictions are evaluated within top-ranked interactions.

## Input/Output Specification

### Input
Methods receive:
1. H5AD file (`tnbc_data.h5ad`) containing:
   - Cell type annotations and gene expression data
   - Training set with 50% of interactions (including ground truth responses)
   - Test set pairs (without responses) to predict
2. Prior knowledge ligand-receptor resource (`ligand_receptor_resource.csv.gz`):
   - OmniPath consensus database of known ligand-receptor interactions
   - Key columns: `source_genesymbol` (ligand), `target_genesymbol` (receptor)
   - Includes metadata like `secreted_intercell_source` (if ligand is secreted)
   - Aggregated from CellPhoneDB, CellChatDB, ICELLNET, connectomeDB2020, CellTalkDB
3. Gene symbol reference mapping (if available)

### Output
Methods must produce a CSV file with:
- Required columns: ligand, target, score
- Must contain exactly the same ligand-target pairs as in `ccc_test_pairs`
- Scores representing predicted interaction probability (0-1)

## Implementation Notes

### Using the Prior Knowledge Resource
The ligand-receptor resource (`ligand_receptor_resource.csv.gz`) provides biological constraints:
```python
import pandas as pd
lr_resource = pd.read_csv('ligand_receptor_resource.csv.gz')
# Key columns:
# - source_genesymbol: ligand gene
# - target_genesymbol: receptor gene  
# - secreted_intercell_source: True if ligand is secreted
```

It is the Omnipath database downloaded from LIANA's R package (Aug 2025).

## Important Considerations
1. **Prior knowledge integration**: Methods should leverage the ligand-receptor resource to:
   - Focus on biologically plausible interactions
   - Check if ligands are secreted (for cell-cell communication)
   - Handle multi-subunit complexes (subunits separated by `_`)
2. **Training data usage**: Use the provided training set to learn patterns
3. **Score interpretation**: Higher scores indicate stronger predicted interactions
4. **Cell type matching**: Target must be valid cell type from dataset
5. **Expression-based filtering**: Consider filtering interactions based on:
   - Ligand expression in any cell type
   - Receptor expression in the target cell type

---
