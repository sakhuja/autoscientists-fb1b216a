---
task_type: biomlbench
name: open-problems-spatially-variable-genes
description: >
  Identify spatially variable genes in mouse cortex spatial transcriptomics (SlideSeqV2, 210 genes).
  Maximize Kendall's tau between predicted spatial scores and ground truth binary labels.
---

## Constraints

- **Python:** `/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python`  
  or `/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/.venv/bin/python`
- **210 genes only:** Small gene set from SlideSeqV2.
- **Compute:** See `program_biomlbench.md` for wall-clock limit and GPU/CPU settings.
- **Unsupervised methods only:** Your method must compute spatial variability scores using only expression data and spatial coordinates — no SVG labels may be used during scoring. See the critical constraint below.

---

Your role in this task is to build an **unsupervised** method for detecting SVGs. Your method must compute spatial scores from expression data and spatial coordinates alone — no SVG labels may be used at any point in the scoring pipeline.

> **⚠️ CRITICAL — UNSUPERVISED METHODS ONLY**
>
> Your `train.py` must produce cortex spatial scores **without using any SVG labels at any point**
> in the scoring pipeline. This means:
>
> - **Do not** train, fine-tune, or fit any model on `cerebellum_labels.h5ad`
> - **Do not** use cerebellum labels for feature selection, hyperparameter search, or weight tuning
> - **Do not** use `data/private/answers.csv` for any purpose whatsoever
>
> Concretely: `score_dataset(adata)` or equivalent must take only expression data and spatial
> coordinates as input and return scores. Labels must never flow into that function.
>
> **How to evaluate during development:** Use the cerebellum dataset as a **read-only sanity check**.
> Run your unsupervised method on the cerebellum expression + coordinates, then compare the resulting
> scores to `cerebellum_labels.h5ad` (`var["true_spatial_var_score"]`) using Kendall's tau. This
> tells you whether your unsupervised statistics correlate with ground truth — but the labels must
> not feed back into the method itself. See the **Iterative Leaderboard** section below for the
> exact evaluation code.

> **⚠️ CRITICAL — DO NOT USE `data/private/answers.csv` FOR EVALUATION**
>
> `data/private/answers.csv` contains the held-out `true_spatial_var_score` values for the 210 genes
> in `data/train.h5ad` (the mouse cortex SlideSeqV2 dataset — this is the actual test set you must
> score). It is used **only** by the external grader to score your final `submission.csv`. **Never
> read, open, or use this file in `train.py` or any intermediate evaluation.** Doing so would
> invalidate your score.

---

# OpenProblems: Spatially Variable Gene Detection

**Metric:** Kendall's tau correlation — higher is better  
**Difficulty:** Medium  
**Data:** Mouse cortex SlideSeqV2 spatial transcriptomics, 210 genes  
**GPU:** Recommended 16GB, ~30 min runtime

---

## The Problem

Spatially variable genes (SVGs) show expression patterns correlated with spatial position in tissue — they mark cell type regions, gradients, or spatial domains. Detecting SVGs from spatial transcriptomics is key to understanding tissue architecture. Ground truth labels are continuous (values 0.0–1.0 in steps of 0.05, representing gene-level SVG scores); predict a continuous spatial score. Kendall's tau handles continuous labels correctly.

**Input:** Spatial gene expression data (H5AD) with spatial coordinates  
**Output:** Continuous `spatial_score` (0–1) for each gene  
**Evaluation:** Kendall's tau correlation between predicted scores and ground truth continuous labels

---

## Data

### Location
```
data/
├── train.h5ad              # AnnData with expression + spatial coordinates (training/all data)
├── cerebellum_train.h5ad   # Additional cerebellum spatial data
├── cerebellum_labels.h5ad  # Cerebellum ground truth labels
└── sample_submission.csv   # gene_id + spatial_score template
```

### How to prepare data
```bash
PYTHON=/n/holylabs/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/bin/python
BIOMLBENCH=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench

PREP=/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/biomlbench/data/manual/open-problems-spatially-variable-genes/prepared
cp $PREP/public/*.h5ad data/ 2>/dev/null
cp $PREP/public/sample_submission.csv data/ 2>/dev/null

$PYTHON -c "
import sys; sys.path.insert(0, '$BIOMLBENCH')
from biomlbench.tasks.manual.open_problems_spatially_variable_genes.prepare import prepare
from pathlib import Path
prepare(Path('data/raw'), Path('data'), Path('data/private'))
"
```

### Data format

```python
import anndata as ad
import numpy as np

# Main spatial dataset
adata = ad.read_h5ad("data/train.h5ad")
# Expression layers are sparse — use .toarray() before numpy operations:
# expr_matrix = np.array(adata.layers['normalized'].toarray())  # shape (cells, genes)
# Spatial coordinates: adata.obsm['spatial']
# Gene names: adata.var_names

# Additional cerebellum dataset
cerebellum = ad.read_h5ad("data/cerebellum_train.h5ad")
cerebellum_labels = ad.read_h5ad("data/cerebellum_labels.h5ad")

print(f"Main dataset: {adata.shape}")        # (cells, genes)
print(f"Cerebellum: {cerebellum.shape}")
print(f"Keys available: obs={list(adata.obs.columns)}, layers={list(adata.layers.keys())}")
print(f"obsm keys: {list(adata.obsm.keys())}")  # should include 'spatial'
```

---

## How to Run

**`train.py` is NOT provided. You must write it from scratch.**

Your workflow:

1. **Write `train.py`** — implement a method that uses the provided datasets,
   evaluates on the cerebellum val set, prints the val metric, and saves `submission.csv`.
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
1. **Evaluate locally** — run your unsupervised method, evaluate on the cerebellum val set, print the val metric.
2. **Save `submission.csv`** — spatial variability scores for all 210 genes in `data/train.h5ad` cortex dataset (columns: `gene_id`, `spatial_score`).
3. **Copy `train.py` to the output directory** — after saving `submission.csv`, copy the current script to the same directory: `import shutil, __file__ as _f; shutil.copy(_f, 'train.py')` or equivalent, so `train.py` and `submission.csv` are always co-located.


---

## Iterative Leaderboard (ClawLab API)

**Metric:** Kendall's tau — higher is better

The cerebellum dataset has ground truth labels and the same anonymised gene names as the cortex data — use it to evaluate your method's performance during development.

```python
import anndata as ad
from scipy.stats import kendalltau

cerebellum_labels = ad.read_h5ad("data/cerebellum_labels.h5ad")

# ... produce scores: dict mapping gene_id -> spatial_score ...

true_labels = cerebellum_labels.var["label"]  # check actual column name
pred_scores = [scores[g] for g in true_labels.index]
tau, _ = kendalltau(pred_scores, true_labels.values)
print(f"Cerebellum Kendall tau: {tau:.4f}")
```

Report this cerebellum tau to the ClawLab leaderboard after each iteration.

**Small-N regime:** With only 210 genes, the cerebellum tau estimate reflects performance on a small gene set. Be cautious about over-interpreting small differences in tau — they may not reflect meaningful differences in method quality.
**Cross-tissue transfer:** The cerebellum and cortex are different tissues. A method that achieves high cerebellum tau but generalises poorly to cortex may be sensitive to tissue-specific properties of the cerebellum data. Consider whether your approach is robust to tissue context.

---

## Final Submission

Produce spatial variability scores for all 210 genes in `data/train.h5ad` (the cortex dataset).

```python
import pandas as pd
import anndata as ad

cortex = ad.read_h5ad("data/train.h5ad")
# ... produce scores: dict mapping gene_id -> spatial_score ...

submission = pd.DataFrame({"gene_id": cortex.var_names, "spatial_score": [scores[g] for g in cortex.var_names]})
submission.to_csv("submission.csv", index=False)
```

The final submission is graded once against held-out cortex ground truth labels using Kendall's tau.

---

## Output Format

```csv
gene_id,spatial_score
GENE1,0.892
GENE2,0.756
GENE3,0.234
...
```

Score should be in [0, 1], higher = more spatially variable.

---



---

## Background

# Spatially Variable Genes Task

## Overview
Spatially variable genes (SVGs) are genes whose expression levels vary significantly across different spatial regions within a tissue or across cells in a spatially structured context.

## Task Description
Recent years have witnessed significant progress in spatially-resolved transcriptome profiling techniques that simultaneously characterize cellular gene expression and their physical position, generating spatial transcriptomic (ST) data. The application of these techniques has dramatically advanced our understanding of disease and developmental biology. One common task for all ST profiles, regardless of the employed protocols, is to identify genes that exhibit spatial patterns. These genes, defined as spatially variable genes (SVGs), contain additional information about the spatial structure of the tissues of interest, compared to highly variable genes (HVGs).

Identification of spatially variable genes is crucial to for studying spatial domains within tissue microenvironmnets, developmental gradients and cell signaling pathways. In this task we attempt to evaluate various methods for detecting SVGs using a number of realistic simulated datasets with diverse patterns derived from real-world spatial transcriptomics data using scDesign3. Synthetic data is generated by mixing a Gaussian Process (GP) model and a non-spatial model (obtained by shuffling mean parameters of the GP model to remove spatial correlation between spots) to generate gene expressions with various spatial variability. For more details, please refer to our [manuscript](https://www.biorxiv.org/content/10.1101/2023.12.02.569717v1) and [Github](https://github.com/pinellolab/SVG_Benchmarking).


## Dataset: Mouse Cortex (SlideSeqV2)
This task uses spatial transcriptomics data from mouse cortex captured using SlideSeqV2 technology. The dataset includes:

- **Technology**: SlideSeqV2
- **Organism**: Mouse
- **Tissue**: Cortex
- **Number of genes evaluated**: 210 genes
- **Spatial pattern generation**: Gaussian Process (GP) based synthetic patterns

The dataset contains a mix of:
- Genes with true spatial variability (score = 1.0)
- Genes without spatial patterns (score = 0.0)

Note: While the ground truth scores in this dataset are binary (0.0 or 1.0), the task requires predicting continuous spatial variability scores that will be evaluated using correlation metrics.

## Data Format
All data is provided in H5AD (AnnData) format:

- `dataset.h5ad`: Training data containing gene expression and spatial coordinates
  - Expression matrix in `layers['counts']` or `layers['normalized']` (log-transformed)
  - Spatial coordinates in `.obsm['spatial']`
  - Gene information in `.var` (with anonymized gene identifiers GENE1, GENE2, etc.)
  
## Evaluation Metric
The task uses **Kendall's tau correlation coefficient** to evaluate the performance of spatially variable gene detection methods. This non-parametric correlation metric measures the ordinal association between predicted and true spatial variability scores, making it robust to outliers and appropriate for ranking-based evaluations.

## Input/Output Specification

### Input
Agents receive:
- `train.h5ad`: Spatial transcriptomics data (mouse cortex) with:
  - Gene expression matrix
  - Spatial coordinates for each spot/cell
  - Gene metadata (anonymized as GENE1, GENE2, etc.)
- `cerebellum_train.h5ad`: Additional spatial transcriptomics data (mouse cerebellum) with ground truth labels available
- `cerebellum_labels.h5ad`: Ground truth labels for the cerebellum dataset to test your method before submission

Note: All gene names have been anonymized to prevent information leakage. Genes are consistently named across all datasets (e.g., GENE1 in cortex data corresponds to GENE1 in cerebellum data).

### Output
Agents must produce a CSV file (`submission.csv`) with:
- `gene_id`: Gene identifier (GENE1, GENE2, etc., matching the anonymized names in the training data)
- `spatial_score`: Continuous score indicating the degree of spatial variability (0.0-1.0)
  - Higher scores indicate stronger spatial variability
  - Scores will be correlated with true spatial variability scores using Kendall's tau

### Sample Submission Format
```csv
gene_id,spatial_score
GENE1,0.123
GENE2,0.456
GENE3,0.789
...
```


Available files:
- `dataset.h5ad`: Training data
- `solution.h5ad`: Ground truth labels
- `simulated_dataset.h5ad`: (Not used - combination of dataset and labels)
- `state.yaml`: Dataset metadata

# A Primer on Spatial Variability

Spatial transcriptomics enables the measurement of gene expression and positional information in tissues. The evolution of spatial transcriptomics technologies advanced the reconstruction of tissue structure and provided profound insights into developmental biology, physiology, cancer, and other fields. However, the complexity and high dimensionality of spatial transcriptomics (ST) data pose new challenges and requirements for analytical approaches. One crucial analytical challenge in spatial transcriptomics studies is the identification of spatially variable genes (SVGs) whose expressions correlate with spatial location, also known as SE genes (genes with spatial expression patterns). Identifying SVGs promotes characterizing spatial patterns within tissues and predicting spatial domains. Several methods have been developed for detecting SVGs. Trendsceek models the data as marked point processes and tests the significant dependency between spatial distributions and expression levels of pairwise points. SpatialDE decomposes gene expression variability into a spatial component and an independent noise term based on Gaussian process regression and tests statistical significance by comparing the SpatialDE model to a null model without the spatial variance component. SPARK, an extension of SpatialDE, uses the Gaussian process regression as the underlying data model and ten different spatial kernels to represent common spatial patterns in biological data, thereby improving statistical power. SPARK-X tests the dependence of gene expressions and spatial locations based on the covariance test framework. scGCO applies graph cuts in computer vision to address SVG identification. It utilizes the hidden Markov random field to identify candidate regions with spatial dependence for individual genes and tests their dependence under the complete spatial randomness framework. Squidpy uses Moran's I to determine SVG and calculates the p-value based on standard normal approximation from 100 random permutations.

Trendsceek, SpatialDE, and SPARK have limited applicability for large-scale datasets due to their high computational complexity. Trendsceek employs the permutation strategy to compute multiple statistics of different paired points, which requires extensive computational work and is only scalable to small-scale datasets. The Gaussian process framework hinders the detection of SVGs and model parameter convergence in SpatialDE and SPARK when analyzing high-dimensional and sparse ST data. SPARK-X offers significantly faster computational speed than the aforementioned methods, but its effectiveness depends heavily on how well the constructed spatial covariance matrix matches the true underlying spatial patterns. The above four methods identify SVGs by searching for predefined relationships between expressions and locations. They have limited generalizability to a wide range of spatial patterns due to the arbitrary nature of the true spatial pattern of SVGs and the resulting uncertainty in the relationship between expression and coordinates. scGCO has the capability to identify SVGs with unknown exact locations and shapes, however, it suffers from false negatives due to the limited accuracy of the graph cuts algorithm in identifying candidate regions for SVGs, especially in sparse ST datasets. The accuracy of Squidpy depends on the number of random permutations. Increasing the number of random permutations enhances the reliability of the results; however, it comes at the cost of increased time consumption, making the process more time-intensive.

Here is an example of one method:

It simulates a diffusion process and evaluates the time it takes to reach a uniform state (convergence). It is a formulation of Fick’s second law to a regular graph (grid). It is defined as:

u\left( {x,y,t + dt} \right) = u\left( {x,y,t} \right) + D{\Delta}u\left( {x,y,t} \right)dt

Where u(x,y,t) is the concentration (for example gene expression on a node in x,y coordinates), D is the diffusion coefficient, t is the update time and ∆(u(x,y,t)dt) is the laplacian on the graph (see elsewhere42 for an extended formulation). Convergence is reached if the change in entropy is below a given threshold:

H\left( {u\left( t \right)} \right) - H\left( {u\left( {t - 1} \right)} \right) < {\it{\epsilon }}

The time t the gene takes to reach consensus is then used a ‘Sepal score’ and indicates the degree of spatial variability of the gene. 
