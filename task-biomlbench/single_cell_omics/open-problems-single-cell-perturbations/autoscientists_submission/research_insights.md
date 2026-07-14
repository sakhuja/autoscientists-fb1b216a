---
task: open-problems-single-cell-perturbations
run_id: biomlb_op_scpert_2
started_at: "2026-04-22T22:37Z"
champion_at: "2026-04-23T00:59Z"
---

# Research Insights for Open Problems Single-Cell Drug Perturbation Response

AutoScientists converged on a family of per-gene Ridge regression models trained via compound leave-one-out cross-validation (LOOCV) on a semi-supervised pool of all available B cell and Myeloid cell labeled data. The headline finding is that **semi-supervised training via compound LOOCV on all 96 B/Myeloid labeled rows** (not just the ~48 official training rows) was the foundational breakthrough, and that **encoding drug-similarity (max Tanimoto) as a 4th per-gene feature** alongside the three cross-cell-type expression features (T cells, NK cells, Tregs) yielded the largest single-experiment gain in the second half of the run. The final champion scored **0.8440 MRRMSE** after a progression of 0.8552 -> 0.8538 -> 0.8510 -> 0.8504 -> 0.8451 -> 0.8441 -> 0.8440.

## Findings

**1. Semi-supervised compound LOOCV on all labeled B/Myeloid rows is the foundational design choice.**
The initial cycle models trained only on the official split=train rows (~48 B/M compounds). Agent gpu1 discovered that since test compounds are unseen, the "public_test" B/Myeloid rows can be included as additional training data by using compound-level LOOCV: when predicting compound i, all other B/M compounds (including previously held-out val rows) are used for training. This change alone dropped MRRMSE from ~0.94 to 0.8604 — the largest single jump in the run.

**2. Per-gene Ridge with cell-type expression features dramatically outperforms global Ridge on 18,211-gene targets.**
Rather than fitting one Ridge model with compound-level features (Morgan FP + aggregate profiles) to predict all 18,211 genes jointly, fitting one closed-form 3D Ridge per gene using [T_g, NK_g, Treg_g] as features (i.e., the same gene's expression in T cells, NK cells, and Tregs as predictors) is dramatically more effective. This per-gene formulation uses the cross-cell-type expression as a gene-specific inductive bias, requiring only ~95 training samples with 3-4 features per gene rather than 18,211 features.

**3. Adding max Tanimoto similarity as a 4th per-gene feature improves performance and is the key architectural advance of cycles 4-5.**
The champion experiment (exp_gamma_010) augmented the per-gene Ridge from 3D [T_g, NK_g, Treg_g] to 4D [T_g, NK_g, Treg_g, tanimoto_max], where tanimoto_max is the maximum Morgan fingerprint Tanimoto similarity of the query compound to all LOOCV training compounds. This gave MRRMSE 0.8504 vs. 0.8538 for the 3D version, confirming that the structural similarity of a compound to the training set is an informative gene-level signal.

**4. Global Ridge regularization was massively over-calibrated; the optimal alpha is 0.1, not 5000.**
Early experiments fixed the global Ridge alpha at 5000 based on rough grid search. Cycle 5 HP tuning (exp_gamma_013) found that alpha=0.1 is optimal (MRRMSE 0.8451 vs. 0.8504 for alpha=5000). At alpha=5000, the global model's predictions were almost entirely shrunk toward zero, effectively eliminating its contribution. At alpha=0.1, the global model (compound-level features: Morgan FP + T/NK/Treg aggregate profiles) contributes as an informative ensemble component.

**5. Cross-cell-type expression as a 5th per-gene feature (Myeloid_g for B prediction, B_g for Myeloid prediction) provides a small additional gain.**
Experiment exp_gamma_015/016 added the paired cell type's expression of the same gene as a 5th feature in the per-gene Ridge. Since the training set contains both B cell and Myeloid data for all compounds, this feature is available without leakage when proper compound-LOOCV is used (both B_i and Myeloid_i for compound i are excluded together). This 5D model improved from 0.8451 to 0.8441.

### Insights

**G1 — Compound LOOCV enables valid semi-supervised training when test compounds are structurally novel.**
*Claim:* When test targets are novel compounds (not cell-type splits), all available labeled rows for the target cell types can be used as training data under compound-level LOOCV, because the held-out validation compounds are also unseen test compounds. This avoids train/val compound overlap without wasting labeled signal.
*Disconfirming evidence:* A task where val and test compounds overlap, or where the compound LOOCV estimate is optimistic relative to true test generalization.
*Observed:* Including val B/M rows in training via compound LOOCV dropped MRRMSE from ~0.94 to 0.8604. The final test submission used all available B/M rows for retraining.

**G2 — Per-gene regression with cross-cell-type expression features is a strong inductive bias for transcriptional perturbation prediction.**
*Claim:* When multiple cell types share the same drug treatment, the drug-response profile in the observed cell types (e.g., T cells, NK cells, Tregs) contains gene-specific signal predictive of responses in unobserved cell types (B cells, Myeloid). Fitting one low-dimensional Ridge per gene with these cross-cell-type features as predictors exploits this regularity efficiently.
*Disconfirming evidence:* A perturbation dataset where different cell type responses are largely uncorrelated, making cross-cell-type features uninformative.
*Observed:* Per-gene 3D Ridge [T_g, NK_g, Treg_g] substantially outperformed global Ridge (FP + aggregate profiles) in all experiments from cycle 2 onward.

**G3 — Max Tanimoto similarity to the training set is an informative gene-level feature for per-gene models.**
*Claim:* For a query compound, its maximum structural similarity to the LOOCV training compounds provides a useful gene-level regularization signal. Compounds with high max Tanimoto can be predicted more accurately via interpolation; compounds with low max Tanimoto should be predicted more conservatively. Encoding this as a per-gene Ridge feature allows the model to learn these gene-specific calibration patterns.
*Disconfirming evidence:* A task where structural similarity (Tanimoto on Morgan FP) is poorly correlated with transcriptional similarity.
*Observed:* exp_gamma_010 (4D: +tanimoto_max) improved over 3D per-gene Ridge from 0.8538 to 0.8504.

**G4 — Interaction terms and nonlinear per-gene models do not improve over linear Ridge with n_compounds ~ 95.**
*Claim:* At ~95 B/M training compounds per cell type, the per-gene model has only 95 samples per gene. Adding pairwise interaction terms (T*NK, T*Treg, NK*Treg) or replacing Ridge with XGBoost or ExtraTrees per gene increases model complexity without sufficient training data, causing overfitting relative to LOOCV MRRMSE.
*Disconfirming evidence:* A task with similar n_compounds where interaction terms or nonlinear models reduce LOOCV error.
*Observed:* exp_beta_013 (6D pairwise interactions): 0.8593 vs. champion 0.8504. Per-gene XGBoost (exp_alpha_012): not competitive. Per-gene ExtraTrees (gpu3/gpu4 cycle 4): 0.8755.

**G5 — Cell-type-specific ensemble weights improve over a single global weight.**
*Claim:* B cells and Myeloid cells may have different degrees of predictability from T/NK/Treg profiles or from the global compound-feature model. Tuning separate ensemble weights (w_global, w_pergene) for each target cell type via LOOCV outperforms a single shared weight.
*Disconfirming evidence:* A task where per-CT weight tuning consistently produces the same weights as a shared global weight.
*Observed:* The champion uses B cells: w_global=0.5, w_pg=0.5 and Myeloid cells: w_global=0.65, w_pg=0.35. Separate tuning improved over shared weights across all experiments that tested both.

### Task-Specific Findings

**T1 — The "public_test" split in de_train.parquet contains labeled B cell and Myeloid data for novel compounds that are fair game for semi-supervised training.**
The dataset structure includes a split column with values train, control, and public_test. Compounds in public_test are also drug-treated compounds with measured B cell and Myeloid gene expression. Since actual test compounds (in id_map.csv) are a separate disjoint set, using public_test rows as additional training data under compound LOOCV is valid and is the single most impactful design decision of the run.

**T2 — The target prediction problem is exclusively B cells and Myeloid cells; T cells, NK cells, and Tregs serve as feature sources only.**
id_map.csv contains only B cell and Myeloid cell (cell_type, sm_name) pairs. de_train.parquet contains 5 cell types. The winning approach uses T cells, NK cells, and Tregs exclusively as feature donors and predicts only B cells and Myeloid cells.

**T3 — The global Ridge alpha was massively over-regularized at alpha=5000; optimal is alpha=0.1.**
Searching global Ridge alpha over [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0] via LOOCV on val compounds found alpha=0.1 consistently optimal. At high alpha, the compound-level global Ridge predictions are effectively zero and its ensemble weight goes to 1 (pure per-gene model). At alpha=0.1, both components contribute meaningfully to the ensemble.

**T4 — B cell and Myeloid cell responses are correlated: adding Myeloid_g as a feature for B cell prediction (and vice versa) improves performance.**
Both B cells and Myeloid cells are myeloid lineage cells (with Myeloid referring to classical myeloid cells in this dataset). The same drug compound produces correlated expression changes in both. Using one cell type's labeled expression as a feature for predicting the other (with compound-LOOCV excluding both B_i and Myeloid_i simultaneously to prevent leakage) improved MRRMSE from 0.8451 to 0.8441. The feature is available at train time because all compounds in the B/M training set have paired B and Myeloid measurements.

**T5 — Deep learning approaches (VAE, Chemprop MPNN, MLP, GNN) underperformed linear Ridge throughout all cycles.**
Multiple neural architectures were tested: adversarial VAE (exp_gamma_001/002, ~0.945 MRRMSE), Chemprop MPNN (exp_beta_001, 0.9212), GNN-style MLP with graph smoothing (exp_beta_001 from gpu3, 0.9537), contrastive perturbation learning MLP (gpu5 cycle 2, 0.9011 on a different variant). None outperformed the per-gene Ridge + global Ridge ensemble at any stage. The dataset has ~95 labeled B/M compounds — too few for neural architectures to overcome their capacity disadvantage relative to closed-form per-gene linear models.

**T6 — Stacked generalization (XGBoost meta-learner on LOOCV base predictions) produced a deceptively good score due to data leakage.**
Experiment exp_gamma_008 (gpu6, cycle 4) reported MRRMSE=0.5159, which was immediately flagged by the admin as data leakage. The meta-learner was trained on full OOF predictions (including val compound targets) and then evaluated on those same val compounds, directly leaking val labels into meta-learner training. This was identified and excluded from the champion.

## Dead Ends and Negative Results

**Deep learning (VAE, MPNN, MLP, GNN):** Tested across cycles 1-3. MRRMSE range 0.9011-0.9619. All substantially worse than linear baselines. The small training set (~95 B/M compounds) and the 18,211-dimensional output space make neural approaches inefficient compared to closed-form per-gene Ridge. Retired after cycle 3.

**Tanimoto KRR (kernel ridge regression with Tanimoto kernel):** Used in cycles 2-4 as a third ensemble component alongside global Ridge and per-gene Ridge. Consistently received near-zero ensemble weight in LOOCV optimization. Dropped in exp_gamma_010. The compound-level Tanimoto signal is better incorporated as a per-gene feature than as a separate KRR model.

**Pairwise interaction features in per-gene Ridge [T*NK, T*Treg, NK*Treg]:** exp_beta_013 scored 0.8593 vs. champion 0.8504. Adding 3 interaction terms to the 3D per-gene model increases the feature space from 3 to 6 dimensions while n_compounds ~ 95, causing overfitting in LOOCV. Dead end.

**Per-gene XGBoost (exp_alpha_012):** Not competitive; LOOCV MRRMSE substantially worse than per-gene Ridge at same feature set. Non-linear per-gene models overfit on 95 compound-LOOCV samples.

**Per-gene ExtraTrees (gpu3/gpu4, cycle 4):** MRRMSE 0.8755. Consistent with XGBoost result; decision-tree-based per-gene models do not benefit from 3-4 input features at n=95.

**Reduced Rank Regression (PCA-compressed multi-output Ridge, alpha team):** Tested extensively in cycles 1-2 (exp_alpha_009/010). Best result ~0.9464. The low-rank approximation of the 18,211-gene output space is a useful prior but was outpaced by the per-gene Ridge formulation which avoids the PCA approximation entirely.

**Pooled per-gene Ridge with cell-type indicator (exp_gamma_018):** Pooling B and M rows with a CT indicator feature (1.0 for B, 0.0 for Myeloid) scored 0.8836, much worse than the separate per-gene models (0.8441). Pooling loses CT-specific coefficient structure; separate models per CT are essential.

**T*NK interaction as 6th per-gene feature (exp_gamma_019):** Scored 0.8493, worse than the champion 5D model. The T*NK cross-term adds no predictive signal and increases regularization pressure.

**kNN Tanimoto-weighted average (cycle 5, kNN approach):** MRRMSE 0.8981. Instance-based predictor received near-zero weight (optimal w_global=0.9, w_kNN=0.1) in ensemble search, meaning it adds essentially no signal beyond the per-gene Ridge. The approach listed as 'kNN-Tanimoto-weighted-avg-ensemble' in the approach registry was completed but not competitive.

**Adaptive per-gene alpha (exp_alpha_013):** Using variance-based gene-specific alpha (high-variance genes get lower regularization) scored worse than the uniform per-CT alpha. The LOOCV alpha search already captures what adaptive alpha attempts to do, and the additional complexity of per-gene alpha degrades LOOCV reliability.

**Corrected stacked generalization with nested meta-LOOCV (exp_alpha_015):** Proposed to fix the leakage from exp_gamma_008's stacking. Computationally slow (N-squared LOOCV for meta-training) and did not complete within the time budget. The approach is listed in the approach registry as "stacked-generalization-nested-LOOCV" and was retired due to runtime.

**Finer HP search beyond champion (exp_gamma_013):** Extended per-gene alpha grid [0.01-100.0] and finer weight grid [0.3-0.8] found no improvement over champion's configuration. The HP landscape is saturated at the champion's settings.

## Coordination and Team Dynamics

**Three teams, one emergent winner.** Three teams (alpha, beta, gamma) were assigned at run start. The alpha team (gpu1, gpu2, analyst1, analyst2) focused on compound-level features, Reduced Rank Regression, and HP search. The beta team (gpu3, gpu4) explored neural approaches and interaction-term extensions. The gamma team (gpu5, gpu6) owned the per-gene Ridge paradigm and drove most champion advances. The champion's architecture (exp_gamma_010, exp_gamma_015, exp_gamma_020) is exclusively from the gamma team.

**Semi-supervised LOOCV discovery propagated immediately.** The breakthrough experiment (compound LOOCV on all 96 B/M rows, MRRMSE 0.8604) was run by the alpha team's gpu1 in cycle 3. Admin promoted it to champion immediately and directed all teams to build on this protocol. From cycle 3 onward, all experiments used compound-LOOCV on all B/M rows.

**Data leakage caught in real time.** When gpu6 reported MRRMSE=0.5159 (exp_gamma_008 stacking), the specific line causing leakage was identified within 2 JSONL messages ("Train meta-learner per gene on ALL OOF pairs" — val targets available during meta-learner training). The legitimate champion's submission.csv was restored from a backup and the leaky result was excluded.

**Analyst proposals were queued but many went unexecuted.** Analyst1 proposed exp_alpha_012 (per-gene XGBoost) and exp_alpha_013 (adaptive per-gene alpha) in cycle 4. These were queued and eventually run, but neither improved on champion. Analyst3 proposed exp_gamma_008 and exp_gamma_009 — the gamma team picked up the stacking concept (which led to the leakage incident). The analyst-to-GPU pipeline worked for queuing but proposals were sometimes delayed by compute contention.

**The beta team did not contribute a champion-competitive experiment.** Team beta's best result was 0.8755 (gpu4's ExtraTrees experiment). The beta team's focus on neural approaches (GNN, Chemprop, interaction terms) and slower experiments (ExtraTrees took ~50 minutes per run) left it behind the gamma team's closed-form Ridge iterations that could complete in under 10 minutes each.

**GPU compute was underutilized in cycles 1-2 due to team formation delays.** Analysts reported "no team found" early in the run because local AGENT.md files were not updated with team assignments at launch. The admin had to manually patch all agents' AGENT.md files. This caused some agents to exit prematurely in cycles 1-2, reducing effective parallelism.

## Limitations of These Insights

**Single run, no independent replication.** All results come from one 4-hour session. The champion score (0.8440 MRRMSE on the public_test split used as validation) may be optimistic due to LOOCV variance over ~48 val compounds. The actual Kaggle public leaderboard score for this submission is unknown.

**Validation is the public_test rows in de_train, not the actual test set.** The compound LOOCV procedure evaluates on the same rows used for semi-supervised training (in leave-one-out mode). The MRRMSE reported (0.8440) is a LOOCV estimate over these ~48 compounds. The generalization to the actual test compounds in id_map.csv may differ.

**The deep learning results are preliminary.** Neural approaches (VAE, Chemprop, MLP) were explored with limited hyperparameter search in cycles 1-2 before being retired. More extensive tuning, pretraining on external transcriptomics data, or foundation models for single-cell biology may overcome the small-data limitation.

**Per-gene Ridge saturated early; unexplored directions remain:**
- Gene co-expression structure (e.g., graph-regularized per-gene models) was not explored.
- Drug mechanism of action features (beyond Morgan fingerprint Tanimoto) were not tried.
- Broader ensemble across experiment trails (e.g., averaging multiple HP configurations) was not attempted.
- The "mean T-cell expression across genes" compound potency feature (used in exp_gamma_020) was added only in the final minutes and its contribution was not isolated.

**Scope of insights.** The per-gene Ridge findings are specific to the cross-cell-type perturbation structure of this dataset (T/NK/Treg as feature donors, B/Myeloid as targets). Transferability to other perturbation tasks with different cell type structures is uncertain.
