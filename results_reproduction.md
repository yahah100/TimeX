# TimeX synthetic reproduction results

All six methods are retained. Each main row is the highest-AUPRC complete five-fold configuration observed at base seed **42**; AUP and AUR come from that same run. Exact ties use the traceable completed run. This is a retrospective selection of observed configurations, not a new uniformly configured benchmark.

Cells show **mean ± fold SE / paper (difference)**. Higher is better. “Within ±0.05” requires all three metric differences to satisfy that bound; it describes numerical agreement only. CoRTX’s multivariate recipe remains unresolved. Results are historical measurements; the cleaned implementation has not yet been rerun on the cluster.

Recipes and source artifacts are linked in each row. The compact [measurement snapshot](experiments/reproduction_results.json) preserves fold values and source hashes even when the large local archives are absent. [Findings and discarded variants](reproduction_findings.md) explain the remaining gaps.

## Table 1: Univariate attribution

| Dataset | Method | AUPRC | AUP | AUR | Paper metric agreement | Recipe / source |
|---|---|---:|---:|---:|---|---|
| FreqShapes | TimeX | 0.8401 ± 0.0202 / 0.8324 (+0.0077) | 0.7440 ± 0.0246 / 0.7219 (+0.0221) | 0.6309 ± 0.0208 / 0.6381 (-0.0072) | Within ±0.05 | [original-timex](timex_21712429/seed_42/results/freqshape_ours_evaluation.log) |
| FreqShapes | IG | 0.7846 ± 0.0447 / 0.7516 (+0.0330) | 0.7290 ± 0.0427 / 0.6912 (+0.0378) | 0.5777 ± 0.0071 / 0.5975 (-0.0198) | Within ±0.05 | [ig-original](timex_21712429/seed_42/results/freqshape_ig_evaluation.log) |
| FreqShapes | Dynamask | 0.2415 ± 0.0204 / 0.2201 (+0.0214) | 0.3391 ± 0.0483 / 0.2952 (+0.0439) | 0.4949 ± 0.0108 / 0.5037 (-0.0088) | Within ±0.05 | [dyna-original](timex_21712429/seed_42/results/freqshape_dyna_evaluation.log) |
| FreqShapes | WinIT | 0.5048 ± 0.0218 / 0.5071 (-0.0023) | 0.5611 ± 0.0137 / 0.5546 (+0.0065) | 0.4494 ± 0.0130 / 0.4557 (-0.0063) | Within ±0.05 | [winit-original](timex_21712429/seed_42/results/freqshape_winit_evaluation.log) |
| FreqShapes | CoRTX | 0.5537 ± 0.0003 / 0.6978 (-0.1441) | 0.3720 ± 0.0007 / 0.4938 (-0.1218) | 0.5000 ± 0.0000 / 0.3261 (+0.1739) | Outside ±0.05 | [reconstruction-original](results/table1/freqshape_cortx_results.json) |
| FreqShapes | SGT + Grad | 0.2900 ± 0.0726 / 0.5312 (-0.2412) | 0.2026 ± 0.0849 / 0.4138 (-0.2112) | 0.3004 ± 0.0461 / 0.3931 (-0.0927) | Outside ±0.05 | [ce-detached-kl-final](results/table1/freqshape_sgt_grad_results.json) |
| SeqComb-UV | TimeX | 0.6831 ± 0.0230 / 0.7124 (-0.0293) | 0.9022 ± 0.0483 / 0.9411 (-0.0389) | 0.2989 ± 0.0572 / 0.3380 (-0.0391) | Within ±0.05 | [original-timex](timex_21712429/seed_42/results/seqcomb_uv_ours_evaluation.log) |
| SeqComb-UV | IG | 0.5089 ± 0.0137 / 0.5760 (-0.0671) | 0.7501 ± 0.0171 / 0.8157 (-0.0656) | 0.3415 ± 0.0150 / 0.2868 (+0.0547) | Outside ±0.05 | [ig-original](timex_21712429/seed_42/results/seqcomb_uv_ig_evaluation.log) |
| SeqComb-UV | Dynamask | 0.4363 ± 0.0086 / 0.4421 (-0.0058) | 0.8727 ± 0.0105 / 0.8782 (-0.0055) | 0.1047 ± 0.0008 / 0.1029 (+0.0018) | Within ±0.05 | [dyna-original](timex_21712429/seed_42/results/seqcomb_uv_dyna_evaluation.log) |
| SeqComb-UV | WinIT | 0.4577 ± 0.0258 / 0.4568 (+0.0009) | 0.7718 ± 0.0247 / 0.7872 (-0.0154) | 0.2426 ± 0.0069 / 0.2253 (+0.0173) | Within ±0.05 | [winit-original](timex_21712429/seed_42/results/seqcomb_uv_winit_evaluation.log) |
| SeqComb-UV | CoRTX | 0.4105 ± 0.0004 / 0.5643 (-0.1538) | 0.5852 ± 0.0005 / 0.8241 (-0.2389) | 0.3383 ± 0.0005 / 0.1749 (+0.1634) | Outside ±0.05 | [reconstruction-original](results/table1/seqcomb_uv_cortx_results.json) |
| SeqComb-UV | SGT + Grad | 0.4402 ± 0.0128 / 0.5731 (-0.1329) | 0.8453 ± 0.0124 / 0.7828 (+0.0625) | 0.1536 ± 0.0067 / 0.2136 (-0.0600) | Outside ±0.05 | [ce-detached-kl-final](results/table1/seqcomb_uv_sgt_grad_results.json) |

## Table 2: Multivariate attribution

| Dataset | Method | AUPRC | AUP | AUR | Paper metric agreement | Recipe / source |
|---|---|---:|---:|---:|---|---|
| SeqComb-MV | TimeX | 0.3735 ± 0.0363 / 0.6878 (-0.3143) | 0.7308 ± 0.0407 / 0.8326 (-0.1018) | 0.2938 ± 0.0155 / 0.3872 (-0.0934) | Outside ±0.05 | [original-timex](cluster_runs/timex_table2_21846132/seed_42/results/seqcomb_mv_ours_results.json) |
| SeqComb-MV | IG | 0.2837 ± 0.0159 / 0.3298 (-0.0461) | 0.6724 ± 0.0296 / 0.7483 (-0.0759) | 0.3172 ± 0.0323 / 0.2581 (+0.0591) | Outside ±0.05 | [ig-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/seqcomb_mv_ig_results.json) |
| SeqComb-MV | Dynamask | 0.3277 ± 0.0116 / 0.3136 (+0.0141) | 0.6199 ± 0.0306 / 0.5481 (+0.0718) | 0.2177 ± 0.0158 / 0.1953 (+0.0224) | Outside ±0.05 | [dyna-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/seqcomb_mv_dyna_results.json) |
| SeqComb-MV | WinIT | 0.2900 ± 0.0358 / 0.2809 (+0.0091) | 0.6030 ± 0.0791 / 0.7594 (-0.1564) | 0.2792 ± 0.0169 / 0.2077 (+0.0715) | Outside ±0.05 | [winit-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/seqcomb_mv_winit_results.json) |
| SeqComb-MV | CoRTX | 0.3623 ± 0.0007 / 0.3629 (-0.0006) | 0.5196 ± 0.0017 / 0.5625 (-0.0429) | 0.3453 ± 0.0006 / 0.3457 (-0.0004) | Within ±0.05; provenance unresolved | [reconstruction-original](cluster_runs/timex_table2_21846132/seed_42/results/seqcomb_mv_cortx_results.json) |
| SeqComb-MV | SGT + Grad | 0.1265 ± 0.0094 / 0.4893 (-0.3628) | 0.2596 ± 0.0198 / 0.4970 (-0.2374) | 0.0852 ± 0.0072 / 0.4289 (-0.3437) | Outside ±0.05 | [poly1-kl-validation](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/seqcomb_mv_sgt_grad_results.json) |
| LowVar | TimeX | 0.8547 ± 0.0186 / 0.8673 (-0.0126) | 0.6076 ± 0.0367 / 0.5451 (+0.0625) | 0.9062 ± 0.0199 / 0.9004 (+0.0058) | Outside ±0.05 | [lowvar-timex](cluster_runs/timex_table2_21926721/seed_42/results/connectivity-rollback-v1/seed_42/lowvar_ours_results.json) |
| LowVar | IG | 0.8326 ± 0.0523 / 0.8691 (-0.0365) | 0.4526 ± 0.0500 / 0.4827 (-0.0301) | 0.7991 ± 0.0184 / 0.8165 (-0.0174) | Within ±0.05 | [ig-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/lowvar_ig_results.json) |
| LowVar | Dynamask | 0.1240 ± 0.0105 / 0.1391 (-0.0151) | 0.1309 ± 0.0177 / 0.1640 (-0.0331) | 0.1944 ± 0.0149 / 0.2106 (-0.0162) | Within ±0.05 | [dyna-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/lowvar_dyna_results.json) |
| LowVar | WinIT | 0.1781 ± 0.0101 / 0.1667 (+0.0114) | 0.1381 ± 0.0032 / 0.1140 (+0.0241) | 0.3859 ± 0.0046 / 0.3842 (+0.0017) | Within ±0.05 | [winit-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/lowvar_winit_results.json) |
| LowVar | CoRTX | 0.1124 ± 0.0015 / 0.4983 (-0.3859) | 0.0590 ± 0.0003 / 0.3281 (-0.2691) | 0.6667 ± 0.0000 / 0.4711 (+0.1956) | Outside ±0.05; provenance unresolved | [reconstruction-qualified](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/lowvar_cortx_results.json) |
| LowVar | SGT + Grad | 0.4149 ± 0.0347 / 0.3449 (+0.0700) | 0.2406 ± 0.0245 / 0.2133 (+0.0273) | 0.5970 ± 0.0233 / 0.3528 (+0.2442) | Outside ±0.05 | [poly1-kl-validation](cluster_runs/timex_table2_21899369/seed_42/results/repaired-v1/seed_42/lowvar_sgt_grad_results.json) |

Table 1 TimeX/IG/Dynamask/WinIT fold SE is reconstructed from fold means rounded to four decimal places in the original logs; other SE values use full-precision JSON. SE describes variation across folds, not across independent training seeds.

SeqComb-MV TimeX and CoRTX retain the original single-attempt predictors, including weak folds 3/4. IG, Dynamask and WinIT use the validation-qualified predictors. SGT trains its own classifier. The predictor differences are part of the selected recipes.

LowVar TimeX improves over the original run from 0.8371 to 0.8547 AUPRC and from 0.5070 to 0.6076 AUP. Its AUP is higher than the paper by 0.0625, so the row remains outside the numerical-agreement tolerance. SeqComb-MV TimeX remains below the paper: 0.3735 versus 0.6878 AUPRC.

## Additional LowVar TimeX seeds

Both completed follow-up seeds are shown separately; neither replaces seed 42.

| Seed | Folds | AUPRC | AUP | AUR | Source |
|---|---:|---:|---:|---:|---|
| 43 | 5 | 0.8711 ± 0.0105 | 0.6452 ± 0.0344 | 0.8757 ± 0.0122 | [Cluster result](cluster_runs/timex_table2_21927128/seed_43/results/connectivity-rollback-v1/seed_43/lowvar_ours_results.json) |
| 44 | 5 | 0.8704 ± 0.0162 | 0.6651 ± 0.0275 | 0.8748 ± 0.0136 | [Cluster result](cluster_runs/timex_table2_21927128/seed_44/results/connectivity-rollback-v1/seed_44/lowvar_ours_results.json) |

## SeqComb-UV IoU (Table 10)

| Method | Reproduction | Paper | Difference |
|---|---:|---:|---:|
| TimeX | 0.4943 | 0.5214 | -0.0271 |
| IG | 0.3183 | 0.3750 | -0.0567 |
| Dynamask | 0.2940 | 0.2958 | -0.0018 |

## Reproduce

Use the same datasets, seed 42 and all five folds. The runners select the fixed recipe for each method automatically; no protocol switches are needed. Training budgets and dataset paths are documented in [Table 1](experiments/TABLE1.md) and [Table 2](experiments/TABLE2.md).

```bash
./run_table1.sh --seed 42
./run_table2.sh --seed 42
# Regenerate this Markdown from the recorded measurements:
PYTHONPATH=. uv run python experiments/evaluation/report_synth.py
```

Use fresh checkpoints from the cleaned code. Old checkpoint formats are not migrated. Cluster numerical verification is still required; CPU checks validate the retained recipes and workflow behavior, not a new five-fold result.
