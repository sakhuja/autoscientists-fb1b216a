---
task: open-problems-label-projection
run_id: biomlb_op_label
started_at: "2026-04-23T13:34:14Z"
champion_at: "2026-04-23T11:52:13Z"
---

# Research Insights for Open Problems Label Projection

AutoScientists found that **a weighted ensemble of scANVI latent representations, logistic regression on scANVI latent embeddings, and logistic regression on normalized HVG features substantially outperforms any single model**, reaching 0.9761 val weighted F1 on the held-out batch (gpu2, `exp_scanvi_lr_ensemble_v1`). The headline finding is that **combining generative VAE latent space with direct HVG logistic regression covers complementary aspects of the batch-correction and classification problem**, and that lightweight ensemble weight search (Nelder-Mead or grid over a small recipe space) consistently adds 0.002-0.005 F1 on top of single models. The submitted champion script (`exp_geneformer_probe_v0`, gpu5) is a CPU-fallback PCA+HVG frozen-probe approach that scored 0.936 val F1 — substantially below the best scANVI ensemble results — due to a mismatch between the champion selection infrastructure and what was actually written to the submission path.

## Findings

**1. scANVI with HVG logistic regression ensemble is the dominant approach on this cross-batch cell type transfer task.**
The best single-batch val score observed was 0.9761 (gpu2, `avg_scanvi_lrhvg` with learned shift, held-out batch `control_1`). The scANVI component alone scored 0.9650 on that batch; adding HVG LR predictions pushed it to 0.9744, and a learned per-class shift correction added another +0.0017.

**2. Choice of held-out validation batch matters substantially for score estimates.**
gpu1 held out `healthy_4` (val scANVI direct: 0.9562); gpu2 and gpu3 held out `control_1` (val scANVI direct: 0.9650 and 0.9671 respectively). The same underlying model architecture produced apparent val scores that differ by ~0.01 depending on which batch is held out. This reflects genuine dataset heterogeneity: some donor batches are harder to transfer to than others.

**3. Three-way weight sweep over scANVI probs / LR-on-latent / LR-on-HVG outperforms equal weighting.**
gpu2 log shows: `avg_all3` F1=0.9739, `avg_scanvi_lrhvg` (best two of three) F1=0.9744, and a learned shift on top: 0.9761. Equal three-way averaging was not optimal. Learned or grid-searched weights consistently found HVG LR to be the most reliable component, with scANVI probabilities and LR-on-latent contributing additional diversity.

**4. Longer scANVI training (120 scVI + 40 scANVI epochs) produced better val F1 than shorter schedules.**
gpu1's `exp_stack_scanvi_harmony_v1` trained for 120 scVI + 40 scANVI epochs and reached 0.9723 val (with a harmony+XGBoost fourth component). gpu3's bagged approach trained 50+20 epochs across 3 seeds and reached 0.9721 (negligible gain from bagging vs. single-seed). gpu2's approach trained 80 scVI + 30 scANVI and produced the highest individual val score (0.9761 on control_1). The relationship is not monotonic — gpu2's architecture differences (transductive + shift correction) contributed beyond the epoch count alone.

**5. The Geneformer/foundation-model paradigm was not executable in this environment; the cpu-fallback proxy scored substantially lower.**
gpu5 (`exp_geneformer_probe_v0`) implemented the "frozen embedding + linear probe" paradigm using precomputed PCA embeddings and HVG features, since Geneformer and scGPT were not installed. This proxy scored 0.9362 val F1 — roughly 0.036 below the best scANVI ensemble. The gap represents the difference between a pretrained genomics transformer embedding and a classical PCA projection, without fine-tuning.

### Insights

**G1 — Ensemble of scANVI soft probabilities with a direct HVG logistic regression classifier is reliably better than either alone on cross-batch scRNA-seq label transfer.**
*Claim:* The two components are complementary because scANVI captures batch-corrected latent structure while HVG LR exploits direct discriminative signal in expression space; averaging predicted class probabilities reduces errors from either alone.
*Disconfirming evidence:* A cross-batch transfer task where scANVI alone already achieves near-ceiling F1 and HVG LR adds noise rather than signal.
*Observed:* scANVI direct (0.9650) + HVG LR (0.9693) → avg 0.9700, best two-component blend 0.9744, best with shift 0.9761 (gpu2, `control_1` held out). Similar pattern on gpu1 and gpu3.

**G2 — Per-class prediction shift correction (label-prior calibration) adds small but reliable gains on top of ensemble averaging.**
*Claim:* When reference and query datasets have different class frequency distributions, a learned per-class additive shift on top of averaged softmax probabilities can recover calibration and improve weighted F1.
*Disconfirming evidence:* Tasks where reference and query share near-identical label distributions, so the shift converges to zero.
*Observed:* gpu2 shift tuning: baseline 0.9744 → post-shift 0.9761, a +0.0017 gain via 2 iterations of grid search on 10 held-out batches.

**G3 — Classical PCA-based linear probes are substantially below scANVI for cross-dataset single-cell label transfer (~3-4 point F1 gap).**
*Claim:* scANVI's VAE + classifier joint training with batch-label conditioning provides a richer, batch-corrected representation than PCA alone, even when PCA uses the same HVG feature set.
*Disconfirming evidence:* A dataset where batch effects are negligible and PCA captures the cell type structure as well as a VAE latent space.
*Observed:* gpu5 PCA+HVG LR probe val F1 = 0.9362 vs. best scANVI ensemble 0.9761. The gap of ~0.040 is consistent across held-out batches seen in gpu5 logs.

### Task-Specific Findings

**T1 — 13 cell types across 10 reference batches (control, diabetic, healthy) with 1 test batch; the batch variable is correlated with disease state, making transfer non-trivial.**
The data contains 33,898 training cells across 10 batches (control_1/2, diabetic_1-5, healthy_4/5/6) and 5,278 test cells in batch `control_3`. This single-test-batch structure means the val score under a leave-one-batch-out split reflects transfer to a held-out control condition, which may or may not generalize to diabetic batches in a hypothetical extended evaluation.

**T2 — HVG selection of 3,000-5,000 genes was stable; results did not improve significantly when varying HVG count in that range.**
gpu1 and gpu3 used 3,000 HVGs; gpu1's v2/v3 runs used 5,000 HVGs. The val scores on comparable architectures differed by less than 0.002, suggesting the information content saturates well below the full 27,980-gene matrix.

**T3 — Harmony batch correction provides moderate standalone accuracy but falls behind scANVI when used as a PCA-reduction step for downstream classifiers.**
gpu3's ensemble included `lr_harmony` (F1=0.9604) and `knn_harmony` (F1=0.9646) as components alongside `lr_hvg` (F1=0.9693). Harmony PCA was weaker than direct HVG LR in every combination tested. The best ensemble weight assigned `lr_hvg` a 0.60 share.

**T4 — Bagging scANVI across 3 seeds (seeds 0, 1, 2) provided negligible improvement over single-seed.**
gpu3 bagged result: 0.9721 val F1. gpu1 single-seed result: 0.9723 val F1 (different architecture, but comparable). The seed variance for scANVI on this dataset is small enough that bagging does not meaningfully reduce variance.

**T5 — XGBoost on Harmony PCA features is the weakest single-model component, scoring ~0.9646 val F1.**
gpu3 included `harmony_xgb` (F1=0.9646) in the ensemble sweep. Despite being the weakest, it contributes mild diversity and was retained at small positive weight in the best ensemble. This is consistent with the observation that tree-based classifiers on batch-corrected PCA offer different failure modes than logistic regression.

## Dead Ends and Negative Results

**Geneformer / scGPT frozen embedding probe (exp_geneformer_probe_v0, gpu5):** The foundation-model paradigm was not executable because neither Geneformer nor scGPT was installed in the environment. The CPU fallback using PCA as a "frozen embedding" scored 0.9362 val F1 — the lowest of any approach tried. The result demonstrates that precomputed PCA is a poor substitute for a pretrained single-cell transformer, rather than being informative about the foundation model paradigm itself.

**Meta-learner (OOF stacking) on top of scANVI + Harmony + HVG components:** gpu1's `exp_stack_scanvi_harmony_v1` tried a LogisticRegression OOF meta-learner and scored 0.9686 val F1, below the best static weight blend of 0.9723 on the same held-out batch. The meta-learner did not outperform grid-searched static weights, consistent with the pattern that OOF meta-features require more training data than the per-batch val folds provide.

**Nelder-Mead weight search with 4-component ensembles (gpu1, beefy_v3):** An extended 4-component ensemble (scANVI + LR-latent + LR-HVG + Harmony-XGB) optimized with Nelder-Mead from multiple starting seeds produced a best avg val F1 of 0.9563 (averaged over two batches: control_1, healthy_4). While this is lower than the single-batch peak of 0.9761, the cross-batch average performance reflects a harder generalization criterion. The Nelder-Mead best weight vector placed ~30% on LR-HVG and ~29% on Harmony-XGB — a larger XGB contribution than expected, but still within a plausible range.

**Transductive scANVI (gpu2, scanvi_transductive_v2):** A second gpu2 run applied scANVI in transductive mode (joint train+test embedding), which produced a submission but its val score was superseded by the earlier `exp_scanvi_lr_ensemble_v1` result (0.9761). It did not appear to significantly improve over the inductive approach.

## Coordination and Team Dynamics

The run was registered at cycle 0 and remained there throughout. Three initial approaches were claimed from the registry simultaneously at launch — `celltypist_lr_hvg` (gpu4), `scanvi_vae` (gpu1), `geneformer_frozen_probe` (gpu5) — and all three ran in parallel without explicit analyst-mediated coordination between cycles.

Several agents (gpu1, gpu2, gpu3) went beyond their initially assigned paradigm and iterated on ensemble and stacking variants within their own workspace. This self-directed iteration produced a richer set of results than the three registered paradigms, but without a formal cycle increment or orchestrated champion-promotion step, the best result (gpu2's 0.9761) was not propagated to the shared champion. The task's `result_latest.json` recorded gpu5's proxy experiment (0.9362) as the final result, which is inconsistent with the best val scores observed in agent workspaces.

The gpu5 agent explicitly noted the unavailability of Geneformer/scGPT in the script docstring, and implemented a fallback — demonstrating appropriate engineering judgment under constraint — but the fallback's lower score was not overridden by orchestration given the cycle-0 stall.

No analyst-mediated knowledge transfer events are evident in the registry or agent outputs. gpu1, gpu2, and gpu3 all independently discovered that scANVI + HVG LR ensembling outperforms scANVI alone, reaching similar conclusions via different architectures.

## Limitations of These Insights

**Cycle 0 run:** The approach registry never advanced past cycle 0. Only the three initial paradigms were formally registered; subsequent within-workspace iterations by gpu1, gpu2, and gpu3 were not coordinated through the orchestration layer. The champion submission reflects the first experiment completed (gpu5's PCA+HVG probe, val F1 = 0.9362) rather than the best experiment found (gpu2's scANVI ensemble with shift, val F1 = 0.9761).

**Single held-out batch validation:** Val scores are all derived from a single held-out batch per run. The leave-one-batch-out setup creates high variance in absolute F1 estimates across agents because different held-out batches have different difficulty. Cross-batch average val F1 (seen only in gpu1's beefy_v3: 0.9563 averaged over two batches) provides a more conservative estimate.

**No test-set ground truth:** All reported scores are validation scores. The actual test-set weighted F1 is not known from these logs. The val-test gap for this type of cross-dataset benchmark is typically 0.005-0.02 depending on distributional shift between held-out reference batch and the actual test batch.

**Environment constraints unexplored:** The Geneformer/scGPT gap (0.040 F1) is an estimate based on a PCA proxy rather than an actual foundation model. True Geneformer performance on this task is unknown from this run.

**Unexplored axes:**
- Cell embedding models (scGPT, UCE, scFoundation) if environment constraints were lifted
- Per-batch classifier training or source-free domain adaptation approaches
- Larger scANVI latent dimension (only 30-40 latent dims tested)
- Temperature scaling or Platt calibration for the per-class shift correction
- Cross-batch ensemble weight selection (gpu1's approach was closest to this, but still a single-run estimate)
