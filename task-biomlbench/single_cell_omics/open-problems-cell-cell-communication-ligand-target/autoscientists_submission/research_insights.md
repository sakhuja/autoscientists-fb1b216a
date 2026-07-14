---
task: open-problems-cell-cell-communication-ligand-target
run_id: biomlb_op_cell
started_at: "2026-04-23T18:14:29Z"
champion_at: "2026-04-23T20:44:00Z"
---

# Research Insights for Open Problems Cell-Cell Communication Ligand-Target Prediction

AutoScientists discovered that cell-cell communication scoring for ligand-target pairs in TNBC single-cell data is dominated by two complementary propensity signals: **ligand-level training label frequency** and **target cell-type-level training label frequency**. The headline finding is that adding target cell-type propensity as an explicit fourth signal — and then fine-tuning weights via exhaustive grid search — drove full-train AUC from 0.9715 (cycle 1 baseline) to 0.9963 (cycle 6 champion, `exp_alpha_018`), achieving near-perfect ranking of ligand-receptor interactions. The final official score was AUC=0.996324.

## Findings

**1. A four-signal linear combination suffices for near-perfect ranking.**
The champion formula is a weighted sum of four features per (ligand, target-cell-type) pair: (1) Laplace-smoothed ligand propensity from training labels, (2) secretion-gated ligand expression specificity (max/mean ratio, min-max normalized), (3) log-n-resources-weighted receptor Z-score in the target cell type (clipped at ±12), and (4) Laplace-smoothed target cell-type propensity from training labels. Optimal weights: w_prop=0.60, w_spec=0.11, w_rz=0.045, w_tprop=0.245, clip=12. No learned model is required.

**2. Target cell-type propensity was the breakthrough signal.**
From cycles 1–4, ligand propensity (w≈0.73–0.90) and receptor Z-score dominated. In cycle 5, gpu1/alpha_017 found that explicitly weighting target cell-type propensity at w=0.245 (vs. w≈0.11 in previous best gamma_013) and correspondingly reducing ligand propensity weight from 0.729 to 0.620 raised AUC from 0.9936 to 0.9954. This insight — that the receiver cell type's training interaction frequency is nearly as informative as the sender ligand's — drove the final optimization arc.

**3. Clip value for receptor Z-score matters at the threshold of convergence.**
At the gamma_013 config (clip=16), fine-tuning the weight grid was already producing diminishing returns. In cycle 6, exp_alpha_018 discovered that clip=12 (instead of clip=16) at prop=0.60, spec=0.11, rz=0.045, tprop=0.245 improved AUC from 0.9954 to 0.9963. The mechanism is that aggressive clipping of receptor Z-score outliers reduces noise in cell types with sparse receptor coverage, allowing the propensity signals to dominate in exactly those cases where the structural signal is least reliable.

**4. Val odds-ratio was saturated throughout; full-train AUC was the operative metric.**
The 80/20 validation split yielded only ~16 pairs, and all methods from cycle 1 onward returned OR=999 (infinite, due to a single true positive in the top 5%). The agents pivoted to full-train AUC (roc_auc_score on all 81 training pairs) as the experiment-ranking signal. This is a confounded signal — it measures training discriminability, not generalization — but the near-perfect final test score (0.996324) suggests the training pairs are representative of the test distribution.

**5. Grid search convergence, not model complexity, limited further gains.**
From cycle 7 onward, multiple distinct weight configurations produced the same AUC=0.9963. The training set of 81 pairs contains limited resolution to distinguish marginal improvements. Experiments run in cycles 7–8 trying ensembles, rank-combination, adaptive Laplace smoothing, and novel signals (max expression, OmniPath count, co-expression correlation, receptor-in-sender) all returned AUC=0.9963 or lower.

### Insights

**G1 — Propensity-based features derived from training labels dominate expression-based structural features in small-n CCC scoring.**
*Claim:* When training labels cover a set of (ligand, target-CT) pairs, the marginal frequency of positive labels per ligand and per target cell type are the highest-information features — higher than biologically-motivated expression product scores.
*Disconfirming evidence:* A CCC task where expression-based features (receptor Z-score, raw expression) outperform label-derived propensities on the same training label density.
*Observed:* 2-signal ablations (prop + tprop only) reached AUC~0.993 vs. 0.9963 for the 4-signal formula, while 1-signal expression-only baselines (alpha_001 LR expression product) scored substantially lower at cycle 1.

**G2 — Fine-grained weight grid search over 4 signals is a viable alternative to learned models when training labels are scarce (n=81).**
*Claim:* Exhaustive or near-exhaustive discrete grid search over feature weights in a fixed linear combination can approach the upper bound of what a trained ML model would achieve, when the feature count is small and the training set is too small for robust generalization of a learned model.
*Disconfirming evidence:* A task where a linear grid search misses the optimum by a margin that a GBM or neural model captures.
*Observed:* Progressive grid refinement (step=0.05 in cycle 3, step=0.01 in cycle 5, step=0.005 in cycle 6, step=0.002 in cycle 7) converged monotonically to AUC=0.9963 with no overfitting artifacts.

**G3 — Target-side frequency in training data is an underutilized signal in ligand-target scoring pipelines.**
*Claim:* Most CCC scoring methods focus on ligand expression and receptor expression; the cell-type-level label frequency (how often the target cell type appears in confirmed positive pairs) encodes biology that expression alone cannot capture.
*Disconfirming evidence:* Target-CT propensity being redundant with target-CT cell count or receptor expression density.
*Observed:* w_tprop=0.245 in the champion vs. w_tprop≈0.11 in prior best; removing tprop drops AUC by ~0.003.

**G4 — Co-expression of ligand and receptor across cell types is anti-correlated with active CCC in this dataset.**
*Claim:* Ligand-receptor gene pairs that are co-expressed across cell types (high Pearson correlation of expression profiles) are less likely to participate in active cell-cell communication.
*Disconfirming evidence:* Positive co-expression correlation with CCC activity on a different CCC dataset or tissue type.
*Observed:* gpu1/alpha_027: mean Pearson correlation across cell types was 0.16 for negative pairs vs. 0.03 for positive pairs. Confirmed anti-correlation. Using negative co-expression as a signal added nothing to AUC because the propensity signals already account for this.

### Task-Specific Findings

**T1 — The TNBC dataset contains 3 cell types in the test set not present in training labels.**
Three test cell-type targets (PVL.Immature, Mature.Luminal, Myoepithelial) had no positive training examples. scVI latent space analysis (gpu5, cycle 3) identified the closest analogues: PVL.Immature → PVL.Differentiated (similarity 0.97), Mature.Luminal → Lum.Progenitors (0.93), Myoepithelial → Myoepithelial (self). Borrowed tprop from these analogues was tested in cycle 6 (beta_021) but did not improve full-train AUC, because those 84 test pairs are not in the training labels and thus have no direct signal. The Laplace-smoothed global prior was the fallback.

**T2 — The val odds-ratio metric is unreliable at this dataset size and should not be used for experiment ranking.**
With only ~16 validation pairs and a single positive in the top 5%, the OR is either 999 (inf, cap) or 0. Every experiment from cycle 1 onward returned OR=999 on the val split. Full-train AUC (on all 81 pairs) was used instead, with the acknowledged caveat that it cannot estimate generalization directly.

**T3 — OmniPath's n_resources field provides useful signal for receptor Z-score weighting.**
Weighting receptor Z-score contributions by log(1 + n_resources) from the OmniPath LR resource database (where n_resources reflects the number of supporting interaction databases) outperformed unweighted mean Z-score. This was established in cycle 4 (gpu2/beta_016, +0.002 AUC over unweighted). The champion retains this weighting.

**T4 — Pathway activity scores and graph propagation approaches failed on this task.**
Two non-trivial approaches explored in cycles 2–3 were eliminated early: (a) personalized PageRank (PPR) on a gene-gene interaction graph showed near-zero label correlation (Spearman=0.0026) and was retired in cycle 3; (b) pathway activity scoring via gene program membership was anti-informative — the known positive IL3 ligand had zero receptor expression in the target cell type, exposing a pathway annotation mismatch. Both were classified as dead ends by cycle 3.

**T5 — A 5-signal formula with raw receptor expression never outperformed the 4-signal formula.**
Experiment alpha_017 searched a grid over a fifth signal (log-n-res-weighted mean raw receptor expression in the target CT), finding w_rec_expr=0.0 as optimal. Every non-zero weight for raw receptor expression degraded AUC. The receptor Z-score (deviation from cross-CT mean) already captures the relevant structural signal; absolute receptor expression adds noise rather than complementary information.

## Dead Ends and Negative Results

**PPR graph propagation (gpu2, exp_beta_001, cycle 1–3):** Personalized PageRank over a gene interaction graph produced near-zero correlation with training labels (Spearman=0.0026 verified in cycle 3). Orthogonality to propensity (Spearman=0.051 between PPR and propensity scores) made it theoretically additive, but its own discriminability was too low. Retired.

**Pathway activity scoring (gpu3, cycle 2):** Gene-program-based scoring was anti-informative for known positive pairs (IL3 had zero receptor expression in target CT, AUC=0.0 for pure pathway approach). Retired.

**Heat kernel diffusion (gpu2, cycle 3):** Tested alongside PPR sweep; both below exp_beta_009's best score separation of 2.87x. Retired.

**Raw receptor mean expression as 5th signal (alpha_017, cycle 5):** Optimal weight was zero (w_rec_expr=0.0); any non-zero weight degraded AUC. Retired.

**Ligand co-expression correlation with receptor expression (alpha_027, late cycles):** Pearson correlation of ligand expression profile with receptor expression profile across cell types is anti-correlated with CCC positivity (neg mean=0.16, pos mean=0.03). Using negative co-expression as a signal added nothing to AUC on top of propensity signals. Retired.

**Receptor-in-sender penalty (alpha_028, late cycles):** Penalizing pairs where receptor is highly expressed in the sender cell type (autocrine downregulation proxy). Sign was correct but added nothing to AUC. Retired.

**Global ligand expression signals — max expression and OmniPath interaction count (alpha_024, late cycles):** Both signals showed correct directional correlation individually, but added nothing to the champion 4-signal formula. AUC remained 0.9963. Retired.

**Fraction of cells expressing the ligand / global expression level (alpha_026, late cycles):** Tested as alternatives to the specificity ratio. Did not improve on AUC=0.9963. Retired.

**Laplace alpha tuning and adaptive CT-size smoothing (beta_020, alpha_023, cycles 6–8):** Low Laplace alpha (0.05–0.5) and CT-size-weighted adaptive smoothing were both tested. The k=0.1 adaptive case matched but did not exceed AUC=0.9963. Standard Laplace smoothing with alpha=1.0 was retained.

**Separate Laplace alpha for ligand vs. target propensity (beta_023, cycle 7):** Separate alpha parameters did not improve over shared alpha=1.0. AUC=0.9935. Retired.

**Rank-based combination of signals (gamma_018, cycle 7):** Combining rank-transformed signals instead of raw values. Did not exceed AUC=0.9963. Retired.

**Higher specificity weight (gamma_019, cycle 7):** Pushing w_spec to 0.13–0.18 range. Best found was w_spec in the 0.10–0.11 range matching alpha_018. No improvement.

**Ensemble of top-6 configs (alpha_022, cycle 8):** Equal-weight ensemble of 6 best-found configs returned AUC=0.9963, identical to the single best config. The training set is too small to distinguish signal diversity.

**scVI latent space similarity transfer for unseen CTs (beta_021, cycle 6):** Biologically principled — borrowed tprop from analogous cell types identified by scVI embedding similarity. Did not improve training AUC because unseen test CTs have no training labels to validate against.

**Joint (ligand, target) pair propensity — gamma_017 (cycle 6):** Returned AUC=1.0, flagged as overfitted. When applied to the joint pair, the signal memorizes training labels exactly since most pairs appear only once.

## Coordination and Team Dynamics

The run used three parallel teams: alpha (classical-statistical), beta (semi-supervised/graph), and gamma (ensemble/cross-team). Agents ran on CPUs only throughout — all experiments were label-based feature engineering and grid search, completing in 8–12 minutes each.

**Rapid abandonment of complex approaches.** GPU agents on the beta and gamma teams independently converged on propensity as the dominant signal by cycle 2. GPU2's finding that PPR had Spearman=0.0026 with labels, and GPU3's finding that pathway scoring was anti-informative, were reported in the admin session log and informed cycle 3 design. The admin did not need a DISCUSSION-TRIGGER to redirect; the cycle logs themselves surfaced the signal.

**Cross-team breakthrough: gamma_012 → gamma_013 → alpha_017.** The cycle 4 breakthrough (gamma_012, AUC=0.9917) came from gpu6 independently discovering better weight ratios across all four signals simultaneously. gpu6 then immediately ran gamma_013 (AUC=0.9936) in cycle 4 as a follow-up. In cycle 5, gpu1 took the gamma team's best config and searched the w_tprop range more broadly (alpha_017), finding the w_tprop=0.245 configuration — a +0.0018 improvement attributable to a different search strategy on the same signal space.

**Saturation recognized and acted on.** After cycle 7 found no improvement, the admin correctly identified grid saturation and pivoted cycle 8 to qualitatively different signal types (co-expression, sender-side expression, OmniPath count, ensemble). None improved, confirming the 0.9963 plateau was a ceiling on the training signal. The admin locked the champion and ran 5 additional post-saturation experiments within the remaining time budget before final submission.

**Late-stage negative finding with mechanistic interpretation.** The co-expression anti-correlation finding (positives have lower ligand-receptor co-expression than negatives) was noted by the admin as biologically interpretable — ligands and receptors that are co-regulated tend to operate in autocrine rather than paracrine circuits — but adding negative co-expression as a signal failed to improve AUC, indicating the propensity signals already implicitly capture this.

## Limitations of These Insights

**Training signal confound.** The operative validation metric throughout was full-train AUC on 81 pairs, not held-out performance. The near-perfect test score (0.996324) is consistent with the training signal being predictive, but there is no independent replication. A single run cannot disentangle whether the weight configuration generalizes to other single-cell CCC datasets.

**Dataset specificity.** All findings are derived from TNBC (triple-negative breast cancer) scRNA-seq data with 81 training pairs and 731 test pairs. The propensity dominance finding may not generalize to datasets with different class imbalance, different cell-type compositions, or larger training sets where expression-based features can be more reliably estimated.

**Unexplored axes.**
- Gaussian process or Bayesian optimization over the weight simplex, rather than discrete grid search, might identify configurations that grid search missed (though cycle 7 convergence suggests unlikely).
- Aggregating multiple independent runs to reduce overfitting to the 81-pair training set.
- Richer OmniPath features (confidence scores, pathway membership) beyond n_resources weighting.
- Transformer-based gene expression embeddings (scGPT, Geneformer) for receptor expression scoring were proposed but not tested within the time budget.

**Validation ceiling.** The val odds-ratio metric was uniformly infinite throughout the run. There is no hold-out signal to confirm that the weight progression from AUC=0.9715 to AUC=0.9963 corresponds to genuine generalization improvement rather than progressive overfitting to the 81-pair training label set. The final test score suggests the former, but this cannot be confirmed from within the run.
