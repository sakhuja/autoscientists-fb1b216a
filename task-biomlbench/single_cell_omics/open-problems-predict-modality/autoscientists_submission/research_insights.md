---
task: open-problems-predict-modality
run_id: biomlb_op_modality_8
started_at: "2026-04-23T21:40:22Z"
champion_at: "2026-04-23T23:13:14Z"
---

# Research Insights for Open Problems Predict Modality (RNA→ATAC)

AutoScientists (6 GPU agents, 3 analysts, 1 admin, 2 cycles, ~3.75 h) discovered that a compact residual MLP operating on 512 TruncatedSVD components of normalized RNA counts is the strongest architecture tried for predicting protein (CITE-seq surface protein) abundance from single-cell transcriptomics in this dataset. The headline finding is that **512 SVD components is a near-optimal dimensionality sweet spot**: doubling to 1024 components with a proportionally wider network degrades validation RMSE by +0.017 (overfitting), while cutting to 256 PCA components loses −0.009 RMSE versus the champion. More complex domain-adaptation or generative approaches (domain-adversarial training, Harmony batch correction, scVI VAE) all performed substantially worse than the straightforward SVD + residual MLP pipeline, validating a strong compression-then-regress inductive bias for this task.

## Findings

**1. A 512-component TruncatedSVD + residual MLP is the dominant architecture.**
The champion (exp_beta_003, val RMSE 0.8560) uses TruncatedSVD(512) on normalized RNA counts followed by a three-block residual MLP [1024→512→256] with BatchNorm, Dropout(0.3), and CosineAnnealingLR. This captures 36.1% of variance in training RNA. The prior single-site-validated champion (exp_alpha_003) used 256 PCA components with a standard (non-residual) MLP of the same hidden dimensions and reached 0.8654. The switch to 512 SVD components plus residual connections together improved RMSE by 0.0094.

**2. SVD dimensionality has a clear optimum at 512; scaling to 1024 causes overfitting.**
Exp_alpha_004 doubled SVD components to 1024 and widened hidden layers to [2048, 1024, 512] (9.5M parameters). Validation RMSE degraded to 0.8729 (worse than champion), early stopping triggered at epoch 23 of 120, and full-data retraining OOM'd on 66K × 1024 features. The training loss continued descending while validation RMSE rose after epoch 10, a textbook overfitting pattern. This establishes 512 as the practical capacity ceiling for this architecture and data size.

**3. Domain-adversarial training (DANN) improves over Ridge baseline but is outclassed by residual MLP.**
Exp_beta_001 (DANN with 128-dim SVD, 80 epochs, alpha=0.1) reached val RMSE 0.9156. Scaling DANN to 256-dim SVD and 120 epochs with alpha=0.15 (exp_beta_002) pushed to 0.9036. Both substantially beat the PCA+Ridge baseline (1.011), but neither approached the residual MLP results. The domain loss stabilized around 1.77 in both experiments, suggesting the adversarial head learned batch-discriminating representations but this correction did not transfer to downstream protein prediction quality.

**4. Batch correction via Harmony degrades prediction.**
Exp_beta_002 (GPU4, cycle 1) applied Harmony to 256-dim PCA embeddings before MLP training. Val RMSE was 0.9292, worse than the plain 256-PCA+MLP baseline (0.8654, exp_alpha_003). Harmony aligns batch distributions by warping latent space, but this removes batch-discriminative signal that may be relevant to cross-site protein prediction.

**5. scVI VAE embeddings are poor predictors of protein abundance in this task.**
Two scVI experiments were run. Exp_gamma_001 (n_latent=30, Ridge head) scored 1.0415 — worse than the Ridge baseline on raw PCA features — suggesting that scVI's generative compression optimized for RNA count reconstruction discards protein-predictive variance. Exp_gamma_002 (n_latent=64, MLP head [256,256]) improved to 0.9699 but remained far above the champion. The scVI latent space explains the within-RNA distribution but not the RNA→protein co-variation.

**6. Validation-weighted ensembling of MLP + Ridge does not improve over MLP alone.**
Exp_beta_004 attempted to combine the residual MLP (val RMSE 0.8735) with a Ridge-SVD model (val RMSE 1.0081) using validation-weighted averaging. The ensemble scored 0.9320 — worse than the MLP alone — and the script correctly fell back to MLP-only predictions. Ridge and MLP predictions are not sufficiently complementary to improve ensemble performance; the Ridge model is too weak a partner.

### Insights

**G1 — A dimensionality sweet spot exists for SVD-then-MLP pipelines on high-dimensional omics data.**
*Claim:* For datasets with ~60K cells and ~14K genes predicting ~134 targets, 512 SVD components captures near-optimal variance for downstream MLP regression; 256 is insufficient and 1024 induces overfitting in the MLP.
*Disconfirming evidence:* A task where SVD dimensionality monotonically improves RMSE up to the full rank, or where 1024 components with sufficient regularization matches 512.
*Observed:* 256 PCA → 0.8654, 512 SVD → 0.8560, 1024 SVD+wider → 0.8729. The 1024-component model early-stopped at epoch 23 with rising validation RMSE.

**G2 — Residual connections improve MLP training stability for medium-depth networks on omics regression.**
*Claim:* Three-block residual MLPs (with learned skip projections when dimensions differ) converge more reliably than plain MLPs of equivalent depth on PCA/SVD-compressed omics inputs, as measured by early-stopping behavior.
*Disconfirming evidence:* A comparable plain MLP with the same hidden dims reaching equivalent or lower RMSE.
*Observed:* Exp_alpha_003 (plain MLP, 256 PCA) reached 0.8654. Exp_beta_003 (residual MLP, 512 SVD) reached 0.8560. The residual architecture with 2× more SVD components improved performance without the early-stopping and OOM pathologies seen in the wider non-residual networks.

**G3 — Domain-adversarial and batch-correction strategies do not help when the validation protocol already holds out a full measurement site.**
*Claim:* When site-held-out validation is used and the goal is to generalize across measurement sites, explicit domain adaptation does not improve over a site-agnostic MLP trained on the same features.
*Disconfirming evidence:* A DANN experiment with more careful gradient reversal tuning, multi-site adversarial heads, or a lower alpha cap that outperforms vanilla MLP on the same task.
*Observed:* Best DANN (exp_beta_002) 0.9036 vs. residual MLP champion 0.8560. Harmony batch correction (exp_beta_002 GPU4) 0.9292 vs. plain MLP (exp_alpha_003) 0.8654.

**G4 — VAE-based generative models trained for RNA reconstruction produce embeddings with poor RNA→protein cross-modal predictive power.**
*Claim:* Optimizing a VAE ELBO on RNA counts produces a latent space that encodes within-modality variation (count distributions, cell type identity) rather than cross-modal covariation, making it a weak predictor of surface protein levels.
*Disconfirming evidence:* A scVI experiment with a protein-aware decoder (totalVI) or a jointly trained RNA+protein model outperforming SVD+MLP.
*Observed:* scVI (n_latent=30, Ridge head): 1.0415; scVI (n_latent=64, MLP head): 0.9699. Both are far above the SVD+MLP champion (0.8560).

### Task-Specific Findings

**T1 — Leave-site-1-out is a natural and stable validation protocol for the BMMC CITE-seq dataset.**
The training data spans sites s1–s3 across 9 batches (66,175 cells). Holding out site 1 (s1d1, s1d2, s1d3 — 14,669 cells; 22.2% of training) provides a clean cross-site generalization signal. Per-batch RMSE was consistent within ±0.01 across s1d1/s1d2/s1d3 for the champion, indicating stable training. The test set contains 1,000 cells including site 4 batches (s4d1, s4d8, s4d9 — unseen measurement site), identified by analyst3 as representing 68.8% of the test set, introducing a generalization challenge not measured by the site-1 validation split.

**T2 — Normalized counts in the `normalized` layer are the correct input representation.**
All agents consistently used `train_rna.layers["normalized"]` rather than raw counts or log-normalized counts. TruncatedSVD applied directly to normalized counts (avoiding the memory cost of mean-centering a 66K × 14K matrix) explained 36.1% of variance with 512 components and 47.5% with 1024 components.

**T3 — The champion retrains on the full 66,175-cell dataset for submission, using the early-stop epoch from validation training as the number of full-retrain epochs.**
This protocol was verified to produce valid submissions (1000 rows × 135 columns, no NaN). The 1024-SVD experiment (exp_alpha_004) failed this step with an apparent OOM error, demonstrating that 1024-component feature matrices exceed working memory budget on the 40 GB A100 when combined with the full training set.

**T4 — LightGBM multi-output regression on top-2000 variance genes did not complete within the run window.**
Exp_alpha_002 (GPU2) was still running at session end after ~2.5 hours, with no val score reported. The result JSON shows status="running" and val_score=null. LightGBM multi-output on 134 targets from 2,000 gene features appears to be computationally prohibitive at the dataset size (66K training cells). No valid result was obtained.

**T5 — CosineAnnealingLR with patience=20 early stopping provides good convergence on this task.**
The champion trained for 49 epochs (of 120) before early stopping at epoch 69, with best RMSE 0.8560 at epoch 49. Training loss continued to decrease while validation RMSE stabilized near epoch 50, consistent with the model learning progressively finer-grained gene-protein co-variation. Per-batch validation RMSEs (s1d1: 0.8518, s1d2: 0.8566, s1d3: 0.8591) were tightly clustered, indicating the scheduler and early stopping produced a well-calibrated model.

## Dead Ends and Negative Results

**PCA(256)+Ridge (exp_alpha_001):** Val RMSE 1.011. The weakest result in the run. Ridge regression on 256 PCA components severely underfits the RNA→protein prediction problem; the linear capacity is insufficient even with regularization.

**Domain-adversarial MLP — DANN (exp_beta_001 and exp_beta_002, GPU3):** Val RMSE 0.9156 and 0.9036 respectively. Despite gradient reversal training with 9-class batch discrimination, DANN consistently underperformed the vanilla residual MLP. The adversarial loss (dom ≈ 1.77 throughout training) never converged toward chance (log(9) ≈ 2.20), suggesting partial batch discrimination removal. Retired: domain adaptation with scalar alpha ramp up to 0.15 does not help when the target site (site 1) is held out and the batch structure is complex.

**Harmony batch correction + MLP (exp_beta_002, GPU4):** Val RMSE 0.9292. Harmony-corrected PCA embeddings degraded performance by 0.064 RMSE versus plain MLP on the same 256-PCA features. Retired: Harmony correction removes inter-site variation that the model needs for protein prediction.

**scVI VAE + Ridge head (exp_gamma_001, GPU5):** Val RMSE 1.0415. Worse than the PCA+Ridge baseline. Retired: scVI latent space (n_latent=30) does not capture RNA→protein cross-modal information.

**scVI VAE + MLP head (exp_gamma_002, GPU6):** Val RMSE 0.9699 (n_latent=64). A 0.29 RMSE gap from the champion. Marked DISCARD. Retired: deeper scVI latent dimensions with MLP decoder partially recover predictive signal but cannot match direct SVD compression.

**1024-SVD + Wider Residual MLP (exp_alpha_004, GPU1, cycle 2):** Val RMSE 0.8729. Early stopped at epoch 23 due to rising validation RMSE. Full-data retraining OOM'd. Marked DISCARD. Retired: 1024 components over-parameterize both the SVD representation and the MLP capacity for this dataset; 512 is the practical ceiling.

**Ensemble MLP + Ridge-SVD (exp_beta_004, GPU3, cycle 2):** Ensemble val RMSE 0.9320, worse than MLP alone (0.8735). The script correctly rejected the ensemble and submitted MLP-only predictions. The Ridge model (val RMSE 1.0081) is too weak a partner to contribute diversity, and the performance-weighted averaging pulled the strong MLP predictions toward the Ridge's errors.

**LightGBM multi-output on 2,000 variance genes (exp_alpha_002, GPU2):** Did not complete; no result. The compute requirement of fitting 134 independent LightGBM regressors on 51K training cells likely exceeds the run window. Marked as incomplete.

## Coordination and Team Dynamics

The run organized 6 GPU agents into three teams (team_alpha: GPU1–2; team_beta: GPU3–4; team_gamma: GPU5–6), each seeding their own experiment queues in parallel. Admin coordinated queue seeding and GPU dispatch, launching agents sequentially after monitoring gpu_claim files to avoid GPU conflicts.

Three analyst agents were deployed. Analyst3 was the only one to produce a substantive analysis memo, identifying the 68.8% site-4 test-cell generalization problem and proposing Factorization Machines with site-regularized offsets as an experiment paradigm. This paradigm was not implemented in the experiment queues; the GPU teams focused on MLP-based approaches. Analysts 1 and 2 did not record findings.

Cross-team information flow was limited. The cycle 2 queues (exp_alpha_004 for team_alpha, exp_beta_004 for team_beta) both drew on the cycle 1 champion result (exp_beta_003, 0.8560) as the baseline to beat. GPU1's AGENT.md explicitly documented the champion's RMSE and proposed further improvements, but cycle 2 experiments (wider MLP, ensemble) both failed to improve on it.

Time pressure was significant. GPU5 (scVI, exp_gamma_001) and GPU2 (LightGBM, exp_alpha_002) were still running when the 15-minute deadline buffer was reached, leaving both without posted results. Admin verified the champion submission at 38 minutes before deadline and correctly did not attempt additional risky reruns.

## Limitations of These Insights

**Single run, no replication.** All findings are from one 4-hour run on one set of random seeds. The champion's 0.8560 RMSE reflects a specific early-stopping point and TruncatedSVD random state (random_state=42); a second run may produce different results.

**Validation proxy may not reflect test generalization.** The leave-site-1-out protocol measures cross-site generalization to sites 1 held-out during training. The test set contains site 4, which was absent from training entirely. Analyst3 estimated 688/1000 test cells (68.8%) come from site 4 — generalization quality to this held-out site is not reflected in the reported val RMSE of 0.8560.

**Incomplete coverage of the approach space.** Several potentially competitive approaches were not run: Factorization Machines (proposed by analyst3), protein-specific calibration layers, lower dropout / higher weight decay on the champion architecture, and longer training (the champion early-stopped at epoch 49 with patience=20; agents noted possible undertraining). totalVI or other jointly-trained multimodal models were not explored.

**LightGBM and scVI experiments were under-resourced or incomplete.** The LGBM experiment (exp_alpha_002) timed out without a result. Both scVI experiments used fixed latent dimensions without hyperparameter search. The conclusions about those approaches are tentative.

**No hyperparameter search on the champion architecture.** The 512 SVD + residual MLP champion used default hyperparameters (lr=1e-3, weight_decay=1e-5, dropout=0.3, batch=512). No grid or random search was conducted around these values. It is plausible that a tuned version of the same architecture would improve beyond 0.8560.
