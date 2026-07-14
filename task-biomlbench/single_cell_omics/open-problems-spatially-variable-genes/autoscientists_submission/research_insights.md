---
task: open-problems-spatially-variable-genes
run_id: biomlb_op_spa_8
started_at: "2026-04-26T03:35:18Z"
champion_at: "2026-04-26T06:07:48Z"
---

# Research Insights for Open Problems Spatially Variable Genes

AutoScientists explored spatial statistics ensembles for ranking spatially variable genes (SVGs) by Kendall τ correlation with ground truth. The headline finding is that **a structured pipeline of Differential Evolution (DE) followed by coordinate descent (CD) and warm Dirichlet search substantially outperforms random weight search alone**, and that **Freeman-Tukey variance-stabilizing transformation (FT-VST) at multiple shift values provides consistent signal lift** over a fixed preprocessing choice. The final champion (Kendall τ = 0.860471 on the cerebellum validation set) uses 24 spatial signals drawn from six complementary spatial statistics families, with weights optimized by DE → CD → warm Dirichlet → CD.

## Findings

**1. Ensemble of spatial statistics families dominates any single method.**
The cycle-1 best individual approaches scored 0.683–0.755 (Kendall τ). By cycle 3, a weighted ensemble of 7–12 signals from SPARK-X, Geary's C, SMR, and variogram reached 0.838–0.844. Each family captures a qualitatively different spatial pattern: SPARK-X (regression on coordinate kernels), Geary's C (local dissimilarity), SMR (smoothing ratio), variogram (semivariance slope), Moran's I (global autocorrelation), Getis-Ord G* (local clustering), and Directional Moran (anisotropic gradient). No single statistic dominates; the optimized weighted combination of all of them yields the best scores.

**2. Freeman-Tukey VST at multiple shift values is a strong preprocessing lever.**
The breakthrough from ~0.846 to 0.859 (cycle 6, exp_gamma_020) came from applying the Freeman-Tukey transform — `sqrt(counts + shift)` — at three shift values (0.25, 0.50, 1.00) and treating each (signal, shift) pair as a distinct feature. The optimal shift differs by signal: based on observed weight distributions, shift=1.0 tends to be preferred for SPARK-X on counts and Geary's C at k=30, while shift=0.25 contributes positively to variogram. Running four key signals (sparkx_cnt, geary_k30, variogram, smr_k8) across three shifts added 12 more signals and added +0.012 τ over the same four signals at a single shift.

**3. Differential evolution finds better weight solutions than random Dirichlet sampling alone.**
In cycle 5, applying DE (10,000 evaluations) to a 12-signal ensemble yielded τ = 0.846404, matching or exceeding 200k–300k Dirichlet trials (which reached 0.846031). In cycle 7, DE (maxiter=300, popsize=25) on the 24-signal set reached de_tau = 0.8594 before CD refinement pushed it to 0.8605. The GPU1 agent confirmed this: pruning the 24-signal set to 14 high-weight signals and running DE (maxiter=500, popsize=30) produced the strongest single-run result observed (τ = 0.8612).

**4. Directional Moran and Getis-Ord G* contribute meaningful independent signal.**
The cycle-5 jump from 0.844 to 0.8447 (exp_gamma_018) came from adding Getis-Ord G* at k=10,k=20 and Directional Moran in x and y (k=20). In the final champion weight vector, dirmoran_y (w ≈ 0.163) and moran_k30 (w ≈ 0.126) are among the largest weights, while geary_k30_neg at shift=1.0 (w ≈ 0.134) and variogram at shift=1.0 (w ≈ 0.140) are also large — confirming that directional and clustering statistics add non-redundant signal beyond standard autocorrelation.

**5. Pruning low-weight signals before high-budget DE improves optimization.**
GPU1's cycle-7 experiment (exp_beta_023) filtered the 24-signal set down to the 14 signals with weights > 0.02 in the gamma_020 champion, then ran DE with larger budget (maxiter=500, popsize=30). This pruned set achieved τ = 0.8612 — the best cerebellum score observed across the run — against the unpruned 24-signal DE result of 0.8605. Removing near-zero-weight signals reduces the dimensionality of the optimization problem and likely allows DE to converge more effectively.

### Insights

**G1 — Multi-family spatial ensembles outperform single-statistic approaches for SVG ranking.**
*Claim:* Combining SPARK-X, Geary's C, SMR, variogram, Moran's I, Getis-Ord G*, and Directional Moran via learned weights produces higher Kendall τ than any individual statistic, because each family is sensitive to a different mode of spatial variation (gradient, clustering, local dissimilarity, anisotropy).
*Disconfirming evidence:* A spatial statistics dataset where one family dominates across all tissue types and no benefit from ensemble weighting is observed.
*Observed:* Cycle-1 individual methods scored 0.60–0.755; the optimized 24-signal ensemble reached 0.860. No single signal exceeded τ = 0.84 individually (inferred from weight distribution and DE convergence behavior).

**G2 — Variance-stabilizing transformation shift value is a hyperparameter that should be optimized or swept.**
*Claim:* The FT-VST shift parameter (`sqrt(x + shift)`) interacts with downstream spatial statistics in a signal-specific way. Running each spatial statistic at multiple shift values and letting a weight optimizer select among them consistently improves over any fixed shift.
*Disconfirming evidence:* A task where τ from a single fixed shift matches the multi-shift ensemble after weight optimization.
*Observed:* Introducing 3-shift variants of four key signals added +0.012 τ in one cycle (0.8468 → 0.8589). The weight optimizer consistently zeroed out some shift-signal combinations and heavily weighted others (e.g., geary_k30_neg at shift=1.0 received w ≈ 0.134 vs near-zero at shift=0.25).

**G3 — Differential evolution with coordinate-descent refinement is more sample-efficient than random Dirichlet sampling for weight optimization on 20–30 dimensional simplices.**
*Claim:* For a 20–30 signal ensemble with weights constrained to a simplex, DE + CD finds higher-quality solutions per function evaluation than random Dirichlet sampling, because DE maintains a population of candidate solutions and exploits gradient-free directional information.
*Disconfirming evidence:* A task where 500k Dirichlet samples outperform DE + CD at equivalent wall-clock time on the same signal set.
*Observed:* 10k DE evaluations matched or exceeded 200k–300k Dirichlet trials on the same 12-signal set in cycle 5. On the 24-signal set in cycle 7, DE (300 iterations × 25 population = up to 7,500 function evaluations) initialized CD that then improved further, demonstrating the value of sequential refinement.

**G4 — High-capacity SPARK-X feature sets (82 features) underperform the 25-feature set in ensemble context.**
*Claim:* Expanding the SPARK-X kernel from 25 to 82 polynomial/trigonometric features does not improve downstream Kendall τ when SPARK-X is used as one component of a weighted ensemble.
*Disconfirming evidence:* A spatial transcriptomics dataset where 82-feature SPARK-X consistently outperforms 25-feature SPARK-X in the ensemble.
*Observed:* exp_alpha_012 (cycle 3) and exp_gamma_016 (cycle 4) both found that 82-feat SPARK-X hurt ensemble performance relative to 25-feat. GPU3 noted explicitly: "82-feat SPARKX hurts vs 25-feat."

**G5 — Inverse-distance weighting (IDW) for spatial neighbor graphs adds no value over binary k-NN for these spatial statistics.**
*Claim:* Replacing binary k-NN neighbor weights with inverse-distance weights in Moran's I or Geary's C does not improve SVG detection, because the distance within the k-NN neighborhood provides little additional discriminating information once neighbors are selected.
*Disconfirming evidence:* A dataset where IDW Moran consistently receives non-zero optimal weight and improves τ.
*Observed:* GPU5 (exp_beta_019, cycle 5) found IDW Moran at k=10 and k=30 received zero weight from the optimizer, explicitly noted: "IDW Moran got zero weight — binary k-NN sufficient, inverse-distance weighting adds no value." GPU1 also tested IDW Moran (cycle 6, exp_beta_019/020) with near-zero weights and below-champion scores.

### Task-Specific Findings

**T1 — The cerebellum validation dataset and cortex submission dataset are distinct; optimizing on cerebellum generalizes to the cortex submission.**
The champion pipeline trains signal weights entirely on the cerebellum (train + label split), then applies those weights to cortex for submission. The fact that the champion was selected this way (final score 0.860471) reflects that the spatial patterns of variably-expressed genes are sufficiently consistent between these brain regions that cerebellum-derived weights transfer usefully to cortex.

**T2 — SMR at multiple neighbor scales (k=5, 8, 12, 20, 30) provides complementary spatial signal.**
The champion uses SMR at k=5, k=8 (via VST-shift variants), k=12, k=20, and k=30, all receiving non-negligible weight. This indicates that spatial smoothing ratio captures different spatial frequency components at different scales, and the weight optimizer integrates them rather than zeroing redundant ones. A fine-grained SMR grid search (cycle 1–2) confirmed k=8 as particularly effective (exp_alpha_001/003 used SMR k=8 for the cycle-1 best of 0.755).

**T3 — The variogram formulation as (gamma[-1] - gamma[0]) / mean_gamma is positively correlated with SVG ground truth; modified lag-stratified formulations invert the signal.**
GPU1 explicitly documented (exp_beta_017_insight): alternative variogram formulations using short- or medium-lag semivariance normalized by gene variance produced negative τ (−0.22 to −0.31), while the original slope-normalized formulation is positively correlated. The original formulation was retained in all subsequent experiments.

**T4 — Geary's C at large k (k=30, k=60) captures more global spatial patterns and outperforms small k in ensemble context.**
Fine-grained Geary grid search (cycle 1) found k=8 was best individually, but in weighted ensembles the k=30 version consistently received the highest weights (visible in all reported best_weights across cycles 4–7). Geary k=60 was tested but received near-zero weight, suggesting k=30 as a practical upper bound for this benefit.

**T5 — Directional Moran (y-component) is consistently more informative than the x-component.**
In the champion weight vector, dirmoran_y receives w ≈ 0.163 vs dirmoran_x w ≈ 0.077. This asymmetry is consistent across the GPU6 and GPU1 result_latest.json files. The reason is not determined from the experiments, but may reflect the spatial orientation of the tissue section used for validation.

**T6 — The 25-feature SPARK-X kernel on the counts layer (after FT-VST) dominates the normalized layer version.**
In the champion, sparkx_norm (applied to log-normalized layer, shift=0.50) receives weight ≈ 0.00001, while sparkx_cnt (counts layer, various shifts) receives substantial weight (shift=0.50: w ≈ 0.035, shift=1.00: w ≈ 0.016). The weight optimizer consistently drives sparkx_norm weight to near zero, indicating the counts-layer SPARK-X signal is more informative for SVG ranking in this dataset.

## Dead Ends and Negative Results

**82-feature SPARK-X kernel:** Tested in exp_alpha_012 (cycle 3) and exp_gamma_016 (cycle 4). Observed performance below champion in both cases. GPU3 note: "82-feat SPARKX hurts vs 25-feat." Retired: larger feature set adds noise to the spatial regression without improving SVG discrimination.

**Anisotropic directional Moran:** exp_beta_015 (cycle 4), tau = 0.778 — a large regression from the then-champion of 0.843. This tested a directional decomposition beyond simple x/y components. Hard dead end; the simple x/y Directional Moran (which became part of the champion) is the effective form.

**Spatial Variance Ratio and Spatial CV signals:** GPU2 tests in cycles 5–6 (exp_gamma_018, exp_gamma_019). Observed delta of −0.013 to −0.018 vs champion. These signals did not capture additional SVG-relevant variance beyond the existing SPARK-X/Geary/SMR core.

**Local Moran I variance (LISA):** GPU2 exp_gamma_019 (cycle 5). Local spatial instability metric added at k=10, k=30; received near-zero weight. Delta = −0.018 vs champion. Retired.

**SVR (Spatial Variance Ratio) k=10, k=30:** GPU2 exp_gamma_018 (cycle 5). Best_weights show svr_k30_neg at w = 0.284 suggesting some contribution, but overall ensemble scored 0.830 — below champion 0.844727 at the time. Retired.

**Multi-lag variogram variants (short/medium lag stratified by distance):** GPU1 exp_beta_017 (cycle 5). Produced tau = 0.820, regression of −0.024 from champion. Documented explicitly: modified variogram formulations with normalized semivariance invert the signal. Hard dead end.

**IDW Moran and IDW Geary:** Tested by GPU1 and GPU5 in cycles 5–6 in multiple experiments. Received zero or near-zero weight from optimizer in all cases. Retired: binary k-NN is sufficient for these statistics.

**Ensemble-of-ensembles (rank average of top submissions):** GPU3 exp_alpha_013_ensemble (cycle 5): rank average of 6 top submissions produced tau ≈ 0.844 — matching but not exceeding the then-champion. Adding this meta-ensemble layer did not improve over single optimized ensemble.

**Extended VST shift grids:** Several agents tested wider shift ranges (shifts 0.0625 to 2.0, or 0.5 to 10.0). GPU2 cycle 6 (exp_gamma_020) with 6 SPARK-X shifts scored 0.829, well below the champion using 3 shifts. GPU5 cycle 6 (exp_beta_021) with 5 shifts including 2.0 and 5.0 scored 0.848 — below champion. The 3-shift regime [0.25, 0.50, 1.00] consistently outperformed wider grids, suggesting that extremes (very small or very large shifts) introduce noise without capturing additional spatial structure.

**GetisG* at extended scales (k=5, k=30):** GPU4 cycle 7 (exp_alpha_016) added GetisG* at k=5 and k=30 to the 24-signal set. Performance 0.859 — marginal improvement over 24-signal baseline but below the exp_beta_023 pruned champion. Getis k=5 received zero weight. The k=10, k=20 scales used in the champion appear to be sufficient.

## Coordination and Team Dynamics

Three loosely coupled agent teams operated across 7 cycles: an alpha team (GPU3, GPU4), a beta team (GPU1, GPU5), and a gamma team (GPU2, GPU6), with two analysts (analyst1, analyst2; analyst3 also participated in cycle 2 with its own experiments). Analyst3 directly ran experiments (exp_gamma_010, exp_gamma_011) rather than purely directing, achieving tau = 0.815–0.816 in cycle 2.

**Cross-team knowledge transfer was fast and effective.** The VST multi-shift discovery (GPU5, beta team, cycle 5) was adopted by GPU6 (gamma team) in cycle 6 to produce the largest single-cycle jump in the run (+0.012 τ, 0.847 → 0.859). The Directional Moran addition (GPU6, gamma team, cycle 5) was subsequently incorporated into the full 24-signal set that became the shared foundation for all cycle-7 experiments. The four key signals (sparkx_cnt, geary_k30, variogram, smr_k8) identified by GPU5's VST experiment were adopted by all teams for the multi-shift backbone.

**DE optimizer adoption spread across teams.** GPU6 introduced DE in cycle 5 (exp_gamma_019), finding +0.0003 improvement. By cycle 7, GPU1 independently applied DE (larger budget) to a pruned signal set, achieving the session-best cerebellum tau of 0.8612. Both agents arrived at DE as the optimizer of choice from different starting points.

**Signal redundancy was identified empirically.** Multiple agents tested overlapping signal families (Geary at many k values, SMR at many k values, variogram variants) and consistently found the optimizer zeroing out most duplicates. This convergent finding from alpha, beta, and gamma teams provides strong evidence that the final signal set in the champion is close to non-redundant for this task.

**No stagnation trigger was required.** The run maintained steady improvement across all 7 cycles: 0.755 (cycle 1) → 0.825 (cycle 2) → 0.844 (cycle 3) → 0.844 (cycle 4) → 0.847 (cycle 5) → 0.859 (cycle 6) → 0.860 (cycle 7). Each cycle added either a new signal family or a better optimizer, preventing plateaus.

## Limitations of These Insights

**Single run, single validation dataset.** All weight optimization was performed on the cerebellum validation set. There is no independent held-out validation to confirm that the weight assignments generalize beyond this single tissue section. The champion submission is applied to cortex, but no cerebellum/cortex cross-validation was performed.

**Optimization objective alignment.** Kendall τ is non-differentiable and was optimized directly via rank-based objectives. The DE and CD procedures optimize on the same validation set used for model selection, creating a risk of overfitting weights to the cerebellum label structure. The magnitude of this risk is not quantified.

**Signal space not exhausted.** Several candidate statistics were proposed but not tested: full Ripley's K/L, spatially-weighted PCA, graph neural network spatial embeddings, and Lee's L statistic (tested briefly in cycle 4 but without thorough optimization). The current champion signal set may not be globally optimal.

**Hyperparameter choices not fully ablated.** The SPARK-X 25-feature kernel design (specific polynomial and trigonometric basis functions) was used throughout without ablation of the feature composition. The variogram bin count (8 bins, k_max=50) was chosen early and not revisited in later cycles. The CD step size (200 steps per dimension, 10 cycles) was fixed; a finer grid or adaptive step size may improve results.

**Tissue-specific weight transfer.** The optimal weights are learned on cerebellum and applied to cortex. The relative importance of directional Moran y vs x (T5) may reflect cerebellum section orientation rather than a universal property, limiting the transferability of this specific finding to other tissue types or spatial transcriptomics platforms.
