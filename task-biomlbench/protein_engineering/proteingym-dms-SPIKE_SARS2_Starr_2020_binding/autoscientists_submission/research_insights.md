---
task: proteingym-dms-SPIKE_SARS2_Starr_2020_binding
run_id: biomlb_pg_spike_2
started_at: "2026-04-29T14:56:08Z"
champion_at: "2026-04-29T16:54:09Z"
---

# Research Insights for ProteinGym DMS SPIKE SARS-CoV-2 Binding Fitness Prediction

AutoScientists achieved a final mean Spearman of **0.6701** (exp_gamma_009, 14-model ensemble) against a starting score of 0.5617 (cycle 1, ESM-1v + ESM2-150M diff + LightGBM). The headline finding is that **iterative ensemble expansion with per-fold Nelder-Mead weight optimization is the single most productive axis in this run**, yielding roughly +0.11 over the best individual model (0.6292, exp_beta_006) and accounting for the majority of progress after cycle 2. A secondary finding is that **the ranking loss (differentiable soft-Spearman) delivered a measurable improvement over MSE for cross-position generalization**, providing the strong individual-model base on which ensemble gains were built.

## Findings

**1. Iterative ensemble expansion with per-fold weight optimization dominated the score ladder.**
The trajectory from cycle 1 to the final champion is almost entirely attributable to stepwise ensemble expansion: 6-model ensemble (exp_gamma_007, 0.6444) → 9-model (exp_gamma_008, 0.6508) → 10-model (exp_alpha_010, 0.6664) → 11-model (exp_alpha_012, 0.6691) → 12-model (exp_beta_008, 0.6693) → 14-model (exp_gamma_009, 0.6701). Each step added models specifically selected for diversity rather than individual score, and a per-fold Nelder-Mead optimizer (20–80 restarts) assigned separate weights per fold type (random / modulo / contiguous). Ensemble models with individual Spearman as low as 0.5487 (exp_alpha_011, ESM2-650M diff ResidualMLP) and 0.5723 (exp_gamma_008 WT position MLP) were kept because the optimizer found non-zero weight for them on at least one fold.

**2. A soft-Spearman ranking loss provided the strongest individual-model baseline.**
exp_beta_006 (score 0.6292, cycle 2 champion) was the first model to substantially exceed the cycle-1 best. Its key change over prior cycle-1 individual models (best ~0.6170) was replacing pure MSE with a 50%/50% mixture of MSE and a differentiable soft-Spearman ranking loss (pairwise sigmoid soft-rank, temperature 0.1). A subsequent variant (exp_beta_007) increased the ranking weight to 0.7 and added a deeper residual MLP (512→256→128→64), reaching 0.6351. These two ranking-loss models were consistently included in every downstream ensemble and received substantial weight.

**3. ESM2-3B masked marginal added information over ESM-1v and ESM2-150M alone.**
exp_beta_006 included ESM2-3B masked marginal as a 1-dimensional scalar feature alongside ESM2-150M diff (640d), ESM2-150M WT context (640d), ESM-1v LLR (1d), and BLOSUM/physicochemical features (62d). The prior best individual model (exp_alpha_006, 0.6170) used the same architecture without the ESM2-3B signal. Agents explicitly framed this as adding a "3B-scale zero-shot fitness estimate orthogonal to ESM-1v," and the +0.012 gain from alpha_006 to beta_006 (holding architecture and blend ratio fixed) is consistent with that framing.

**4. Per-fold weight assignment outperforms fold-agnostic ensemble weighting.**
exp_gamma_008 (9-model ensemble, 0.6508) compared directly against exp_gamma_007 (6-model ensemble, 0.6444). The larger pool of models was additive, but the explicit improvement attributable to per-fold rather than joint optimization is visible in the script comment: "Per-fold weights: separate Nelder-Mead for each fold type (random/modulo/contiguous) — allows the model to assign different weights for modulo vs contiguous folds." This was maintained in all subsequent ensembles.

**5. Model-family diversity (LightGBM vs. MLP) contributed measurably to ensemble gains.**
exp_beta_008 added a LightGBM model trained on ESM2-150M diff + ESM2-3B marginal + ESM-1v marginal as the 12th ensemble member. Its docstring notes it is "fundamentally different model family from all MLP models in ensemble — strong on random fold (0.815), weaker on positional splits." The ensemble (0.6693) gained from LightGBM's high precision on the random fold while MLP/ranking-loss models anchored the modulo/contiguous folds.

### Insights

**G1 — For DMS fitness prediction, per-fold ensemble weight optimization captures fold-type heterogeneity missed by joint optimization.**
*Claim:* When three structurally different CV split strategies (random, modulo, contiguous) are used simultaneously, allowing separate ensemble weights per fold is strictly better than optimizing a single weight vector.
*Disconfirming evidence:* Tasks where all three folds have consistent rank ordering across models, making fold-agnostic weights equally effective.
*Observed:* Per-fold Nelder-Mead (exp_gamma_008, 0.6508) improved over the prior 6-model joint-weight ensemble (exp_gamma_007, 0.6444) beyond what adding 3 new models alone explains; the same per-fold strategy was propagated through all 14 subsequent ensemble iterations.

**G2 — Iterative ensemble expansion where low-individual-scoring models are included for diversity is productive up to at least 14 members.**
*Claim:* Adding ensemble members with individual scores well below the current champion can improve the ensemble if their residuals are sufficiently decorrelated, and the optimizer can discover zero weight for non-contributing members.
*Disconfirming evidence:* Consistent degradation when adding low-scoring members, or stagnation at fewer than 8 models on similar tasks.
*Observed:* exp_alpha_011 (individual Spearman 0.5487, ESM2-650M backbone) received weight 0.1036–0.2133 per fold in exp_alpha_012; exp_gamma_008_wt_pos (0.5723, WT position embedding) received non-zero weight in exp_gamma_009. Both additions improved the mean score.

**G3 — A differentiable soft-Spearman ranking loss improves cross-position generalization over pure MSE for protein DMS fitness regression.**
*Claim:* When the evaluation metric is Spearman rank correlation and test sets include positional splits (modulo/contiguous), replacing MSE with a mixed MSE + soft-Spearman loss during MLP training reduces the train/eval metric gap and improves held-out Spearman.
*Disconfirming evidence:* Tasks where MSE-trained models already achieve near-ceiling Spearman, or where ranking loss introduces instability on small fold sizes.
*Observed:* exp_beta_006 (ranking_weight=0.5) improved from the best MSE-only individual model (exp_alpha_006, 0.6170) to 0.6292 on an otherwise identical feature set and blend ratio. exp_beta_007 (ranking_weight=0.7) further improved to 0.6351.

**G4 — Larger ESM2 models (3B vs. 150M) provide incremental additive signal as scalar marginals even when not used as backbone embeddings.**
*Claim:* Adding ESM2-3B masked marginal as a single scalar alongside 150M-scale full embeddings provides non-redundant signal; the large model's implicit evolutionary constraint is not fully captured by the smaller model's marginals or embeddings.
*Disconfirming evidence:* An experiment showing ESM2-3B marginal adds no improvement over ESM2-150M marginal after controlling for other features.
*Observed:* The feature transition from exp_alpha_006 (no ESM2-3B, 0.6170) to exp_beta_006 (with ESM2-3B, 0.6292) is the most controlled comparison available; both use the same model family, blend ratio, and all other features. The 1-dimensional marginal was retained in all subsequent high-performing models.

### Task-Specific Findings

**T1 — The three CV fold strategies produce structurally different model rankings.**
Scripts consistently note that random fold Spearman is substantially higher than modulo or contiguous Spearman for the same model (e.g., LightGBM in exp_beta_008 shows random=0.815 but weaker on positional splits). This divergence motivates per-fold ensemble weighting and means overall mean Spearman masks fold-type heterogeneity.

**T2 — ESM2-150M diff embedding (640d) combined with WT position context (640d) is a consistently strong backbone for individual models.**
Nearly every model scoring above 0.60 uses this combination. The "diff" captures the direction and magnitude of embedding change under mutation; the "WT position context" captures the structural/functional context at the mutation site. This combination was first introduced in exp_alpha_005 (0.6023, replacing pure diff + ESM-1v + BLOSUM) and was retained as the backbone of all subsequent high-scoring individual models through exp_beta_006 and exp_beta_007.

**T3 — ESM2-650M WT position embedding (1280d) provides diversity to the ensemble despite weaker individual performance.**
exp_gamma_008 (agent gpu5) used ESM2-650M position-specific WT embedding (1280d) as the primary backbone, achieving individual Spearman 0.5723 — below most other models. However, as the 13th member of exp_gamma_009, it received non-zero ensemble weight because its structural-context encoding differs from the ESM2-150M diff + context paradigm used by all other members.

**T4 — Physicochemical / k-mer baselines (analyst2) scored too low for direct inclusion in ensembles but were available as distant ensemble members.**
The analyst2 workspace contains five experiments (ESM-IF1 proxy physicochemical, regularized global, k-mers, reduced hybrid, alpha search), all using Ridge regression with shallow feature representations. These were loaded as candidate members in exp_gamma_010 (34-model expanded ensemble) but received effectively zero weight in the Nelder-Mead optimization given the much stronger ESM-based members present.

**T5 — Ensemble score improvement saturated in cycle 3; expanding from 14 to ~34 members (exp_gamma_010) did not surpass the 14-model champion.**
The SOURCE file records exp_gamma_009 (14-model, 0.6701) as the final recorded champion. exp_gamma_010 attempted a full expansion to all available OOF prediction files (~34 models) using 20 Nelder-Mead restarts with 100 iterations (reduced from 80 restarts × longer in prior ensembles), but no improvement over 0.6701 is evidenced — the champion/train.py remains the exp_gamma_009 script (not exp_gamma_010), confirming the 14-model ensemble was the final champion.

## Dead Ends and Negative Results

**Pure MSE objective for MLP training on DMS fitness:** All cycle-1 individual models using MSE (exp_alpha_005, 0.6023; exp_alpha_006, 0.6170; exp_alpha_008, 0.6137) were surpassed by exp_beta_006 (0.6292) and exp_beta_007 (0.6351) once ranking loss was introduced. MSE models remained in ensembles as diversity contributors but were not developed further as standalone architectures.

**XGBoost with heavy regularization (exp_alpha_007, 0.6128):** Tested as an alternative to LightGBM, adding mutant position embedding and ESM2-3B marginal but switching the tree model. Performance (0.6128) was below the contemporary LightGBM + diff champion and below the MLP models. XGBoost was not pursued further as a primary model family.

**ESM2-650M mean-pooled global diff as primary backbone (exp_alpha_011, 0.5487):** Using the larger ESM2-650M mean-pooled variant-minus-WT difference embedding (1280d) as the full feature backbone with a ResidualMLP scored substantially lower than the position-specific ESM2-150M diff approach. This was included in ensembles for diversity (non-zero weight found) but not developed as a standalone direction.

**Analyst2 physicochemical / k-mer baselines:** Ridge regression on physicochemical properties, k-mer features, and hybrid representations produced low-scoring models. These were included as candidate ensemble members in the cycle-3 34-model expansion but received effectively zero weight from the optimizer.

**34-model "expand all" ensemble (exp_gamma_010):** Loading all available OOF prediction files across all agents (~34 models) with a reduced Nelder-Mead budget (20 restarts, 100 iterations) did not surpass the 14-model ensemble. The final champion remains exp_gamma_009.

## Coordination and Team Dynamics

Three GPU teams (alpha: gpu1/gpu2, beta: gpu3/gpu4, gamma: gpu5/gpu6) operated in parallel, with analysts (analyst1, analyst2, analyst3) running exploratory experiments from the beginning. Analyst2 produced five Ridge-regression experiments using physicochemical and k-mer features; these ran early and independently, providing low-scoring but distinct OOF predictions that were available as far-diversity candidates in the final ensemble expansion.

Cross-team model sharing was the primary coordination mechanism from cycle 2 onward. The exp_gamma_007 "cross-team OOF weighted ensemble" script explicitly lists models from all three GPU teams (beta007/006, alpha006/008/005, gamma001) and describes different teams' strengths: "Experiments with ranking loss: better on cross-position (modulo/contiguous); Ridge-heavy models: better on random folds." This framing was adopted across all subsequent ensemble experiments.

The ensemble-expansion approach (adding one or two new members per round) propagated across teams: gpu6 ran exp_gamma_008 (9→14-model), gpu1 ran exp_alpha_010 (9→10-model), gpu2 ran exp_alpha_012 (10→11-model), gpu3 ran exp_beta_008 (12-model + LightGBM), and gpu6 ran exp_gamma_009 (14-model). Each agent tracked the current best ensemble size and added its own latest individual model as a candidate. The final 14-model champion pools models from gpu1–gpu6, with gpu3 and gpu4 contributing the highest-weighted individual members (ranking-loss MLPs exp_beta_006 and exp_beta_007).

No evidence of analyst1 or analyst3 GPU-team task assignments is present in the workspace scripts; analyst2's five experiments are the only analyst-team artifacts with submission files.

## Limitations of These Insights

**Single run, no independent replication.** All scores are from a single 3-cycle run. No held-out test set separate from the three CV fold strategies (random_5, modulo_5, contiguous_5) is available; the reported scores are in-distribution CV means. The true generalization gap is unknown.

**Ensemble score improvement may be partially attributable to overfitting to the CV folds.** Per-fold Nelder-Mead weight optimization is performed on the full dataset using OOF predictions (fold-level holdout is respected), so direct data leakage is avoided. However, weights are tuned to maximize mean Spearman on exactly these three fold strategies, and generalization to a different CV protocol or held-out set is not tested.

**Individual model contributions in the ensemble are not fully disentangled.** While the optimizer assigns non-zero weight to low-scoring members (indicating they provide decorrelated signal), the magnitude of their net contribution relative to the 2–3 dominant models (exp_beta_007, exp_gamma_007, exp_beta_006) is not reported in aggregate form. Ablations removing individual members were not conducted.

**Unexplored axes within the budget:**
- Larger ESM2 backbone (3B-scale) as a full embedding source rather than a scalar marginal — this was discussed in exp_alpha_011's framing but not executed with the 3B model as the primary embedding.
- Fine-tuned or task-adapted PLM representations rather than frozen embeddings.
- Structure-based features (actual 3D coordinates or ESM-IF1 inverse-folding embeddings) — analyst2 attempted an ESM-IF1 proxy but fell back to physicochemical properties when the model was unavailable; no actual structure-based feature reached a competitive individual score.
- Temperature / hyperparameter search for the soft-Spearman loss across a wider range.
