# TimeX Paper Reproduction: Results & Interpretation

This document provides a comprehensive report on reproducing the synthetic benchmark results from the TimeX paper (**"Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency"**, [arXiv:2306.02109v2](2306.02109v2.pdf)).

It covers both **Table 1** (Univariate Attribution) and **Table 2** (Multivariate Attribution), evaluates reference predictor quality (**Table 15**), checks explanation overlap via IoU (**Table 10**), and provides an in-depth interpretation of the empirical findings.

---

## 1. Current status

The tables below preserve historical measurements. The **repaired Table 2 GPU
run has not been executed**; the user will submit the jobs. Historical numerical
agreement is not acceptance under the new provenance and validation gates.
Table 1 measurements are retained for reference and were not rerun in this repair.

The implementation repairs temporal connectivity, frozen-reference dropout,
clipping order, omitted checkpoint loss weights, predictor quality gates, and
SGT's loss/gradient flow. The runner now validates artifact and upstream hashes,
isolates protocols/seeds/folds, preserves failed attempts and supports archived
resume. See [the runnable workflow](experiments/TABLE2.md).

Focused CPU checks cover numerical invariances, gradients, actual reduced
training loops, checkpoint compatibility, predictor retries and runner resume
semantics. [Reloaded checkpoint evidence](experiments/table2_cpu_evidence.json)
confirms SeqComb-MV validation macro-F1 of 0.78716346/0.86871410 for folds 3/4.
LowVar fold 1 has 64.0584% of training values outside the sigmoid decoder's range,
with a bounded-reconstruction MSE lower bound of 0.48343837. CoRTX's multivariate
recipe remains unresolved. The earlier near-constant-mask claim is withdrawn:
the inspected output is heavily saturated with standard deviation 0.424000.

SGT's longer 1000/120-epoch budget and validation checkpoint selection are
explicit convergence repairs. CoRTX reconstruction and ten-epoch SGT already
appear in the author snapshot; they were not newly invented by the previous
runner. [Deviation notes](table2_deviations.md) separate confirmed defects,
protocol changes, hypotheses and unresolved provenance.

---

## 2. Reference Predictor Performance (Table 15)

In explainability benchmarks, the quality of attribution maps depends fundamentally on the performance of the underlying classifier being explained. Transformer predictors were evaluated over 5 folds:

| Dataset | Type | Reproduced Mean Test F1 | Paper Mean Test F1 | Difference | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **FreqShapes** | Univariate | **0.9740** (seed 42) / **0.9790** (seed 0) | 0.9716 ± 0.0034 | +0.0024 / +0.0074 | Fully aligned |
| **SeqComb-UV** | Univariate | **0.9184** | 0.9415 ± 0.0052 | -0.0231 | Slightly lower |
| **LowVar** | Multivariate | **0.9774** | 0.9748 ± 0.0056 | +0.0026 | Fully aligned |
| **SeqComb-MV** | Multivariate | **0.8866** | 0.9765 ± 0.0024 | -0.0899 | Discrepancy on folds 3 & 4 |

> **Key Predictor Insight**: The LowVar and FreqShapes predictors match or exceed the paper's target performance across all folds. SeqComb-MV trained successfully on splits 1, 2, and 5 (F1: `0.9539`, `0.9841`, `0.9639`), but underperformed on splits 3 and 4 (F1: `0.7787`, `0.7524`), which may contribute to the attribution gap; the causal effect awaits controlled pilots.

---

## 3. Table 1: Univariate Synthetic Attribution

Metrics are reported as `Reproduction / Paper (Difference)`. Higher is better.

| Dataset | Method | AUPRC | AUP | AUR | IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FreqShapes** | **TimeX** | **0.8401** / 0.8324 *(+0.0077)* | **0.7440** / 0.7219 *(+0.0221)* | **0.6309** / 0.6381 *(-0.0072)* | 0.5057 / — |
| | IG | 0.7846 / 0.7516 *(+0.0330)* | 0.7290 / 0.6912 *(+0.0378)* | 0.5777 / 0.5975 *(-0.0198)* | 0.4777 / — |
| | Dynamask | 0.2415 / 0.2201 *(+0.0214)* | 0.3391 / 0.2952 *(+0.0439)* | 0.4949 / 0.5037 *(-0.0088)* | 0.1837 / — |
| | WinIT | 0.5048 / 0.5071 *(-0.0023)* | 0.5611 / 0.5546 *(+0.0065)* | 0.4494 / 0.4557 *(-0.0063)* | 0.2575 / — |
| | CoRTX | 0.5537 / 0.6978 *(-0.1441)* | 0.3720 / 0.4938 *(-0.1218)* | 0.5000 / 0.3261 *(+0.1739)* | 0.3829 / — |
| | SGT + Grad | 0.2900 / 0.5312 *(-0.2412)* | 0.2026 / 0.4138 *(-0.2112)* | 0.3004 / 0.3931 *(-0.0927)* | 0.1206 / — |
| **SeqComb-UV** | **TimeX** | **0.6831** / 0.7124 *(-0.0293)* | **0.9022** / 0.9411 *(-0.0389)* | **0.2989** / 0.3380 *(-0.0391)* | **0.4943** / 0.5214 *(-0.0271)* |
| | IG | 0.5089 / 0.5760 *(-0.0671)* | 0.7501 / 0.8157 *(-0.0656)* | 0.3415 / 0.2868 *(+0.0547)* | 0.3183 / 0.3750 *(-0.0567)* |
| | Dynamask | 0.4363 / 0.4421 *(-0.0058)* | 0.8727 / 0.8782 *(-0.0055)* | 0.1047 / 0.1029 *(+0.0018)* | 0.2940 / 0.2958 *(-0.0018)* |
| | WinIT | 0.4577 / 0.4568 *(+0.0009)* | 0.7718 / 0.7872 *(-0.0154)* | 0.2426 / 0.2253 *(+0.0173)* | 0.2851 / — |
| | CoRTX | 0.4105 / 0.5643 *(-0.1538)* | 0.5852 / 0.8241 *(-0.2389)* | 0.3383 / 0.1749 *(+0.1634)* | 0.2517 / — |
| | SGT + Grad | 0.4402 / 0.5731 *(-0.1329)* | 0.8453 / 0.7828 *(+0.0625)* | 0.1536 / 0.2136 *(-0.0600)* | 0.2530 / — |

*(Note: In an earlier local run using seed 0, TimeX on FreqShapes reached 0.8713 AUPRC, 0.7898 AUP, 0.6266 AUR, and 0.5323 IoU).*

All six methods now have reproduced Table 1 numbers at base seed 42. CoRTX and SGT + Grad
were added last and use the reimplementations in `txai/baselines/synth_baselines.py`; both
fall materially short of the published values, and §5.3 explains why. **TimeX keeps rank #1
on every metric of both univariate datasets**, so the paper's central claim is unaffected —
but the baseline ordering below TimeX changes:

| Dataset | Metric | Published order | Reproduced order |
| :--- | :--- | :--- | :--- |
| FreqShapes | AUPRC | TimeX > IG > CoRTX > SGT > WinIT > Dynamask | TimeX > IG > CoRTX > WinIT > SGT > Dynamask |
| SeqComb-UV | AUPRC | TimeX > IG > SGT > CoRTX > WinIT > Dynamask | TimeX > IG > WinIT > SGT > Dynamask > CoRTX |

### Table 10: SeqComb-UV Intersection over Union (IoU)

The paper evaluates IoU to verify attribution map localization without thresholding bias:

| Method | Reproduced IoU | Published IoU | Difference | Published Rank | Reproduced Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TimeX** | **0.4943** | **0.5214** | -0.0271 | **#1** | **#1** |
| **IG** | 0.3183 | 0.3750 | -0.0567 | #2 | #2 |
| **Dynamask**| 0.2940 | 0.2958 | -0.0018 | #3 | #3 |

**Result**: The IoU ranking is 100% preserved. TimeX maintains a +0.1760 IoU lead over the closest baseline.

The paper reports no IoU for the remaining methods; the reproduction measures WinIT at 0.2851, SGT + Grad at 0.2530, and CoRTX at 0.2517 on SeqComb-UV, all below Dynamask and far below TimeX.

---

## 4. Table 2: historical comparison and repaired-run status

The following numbers are **historical**, from job 21846132 at seed 42, and use
`Historical mean ± fold SE / Paper (Difference)`. All three differences are
reported for each row. They are not repaired estimates.

| Dataset | Method | Historical all-three differences ≤0.05? | Repaired status |
|---|---|---|---|
| SeqComb-MV | TimeX | No | Pending GPU; 0/5 repaired folds |
| SeqComb-MV | IG | No | Pending GPU; 0/5 repaired folds |
| SeqComb-MV | Dynamask | Yes | Pending GPU; failed historical predictor gate |
| SeqComb-MV | WinIT | No | Pending GPU; 0/5 repaired folds |
| SeqComb-MV | SGT + Grad | No | Pending GPU; objective/budget repair |
| SeqComb-MV | CoRTX | Yes | Unresolved multivariate provenance; GPU pending |
| LowVar | TimeX | Yes | Pending GPU; 0/5 repaired folds |
| LowVar | IG | Yes | Pending GPU; 0/5 repaired folds |
| LowVar | Dynamask | Yes | Pending GPU; 0/5 repaired folds |
| LowVar | WinIT | Yes | Pending GPU; 0/5 repaired folds |
| LowVar | SGT + Grad | No | Pending GPU; objective/budget repair |
| LowVar | CoRTX | No | Unresolved multivariate provenance; GPU pending |

Acceptance requires five complete, traceable repaired folds, validation-qualified
predictors, supported provenance, and **each** AUPRC/AUP/AUR difference ≤0.05.
The runner generates a complete twelve-row CSV/Markdown comparison with those
statuses after every run, including failed or partially completed runs.

### LowVar (Multivariate)

| Method | AUPRC | AUP | AUR |
| :--- | :---: | :---: | :---: |
| **TimeX** (Ours) | **0.8371 ± 0.0231** / 0.8673 *(-0.0302)* | **0.5070 ± 0.0379** / 0.5451 *(-0.0381)* | **0.9031 ± 0.0121** / 0.9004 *(+0.0027)* |
| **IG** | 0.8326 ± 0.0523 / 0.8691 *(-0.0365)* | 0.4526 ± 0.0500 / 0.4827 *(-0.0301)* | 0.7991 ± 0.0184 / 0.8165 *(-0.0174)* |
| **WinIT** | 0.1781 ± 0.0101 / 0.1667 *(+0.0114)* | 0.1381 ± 0.0032 / 0.1140 *(+0.0241)* | 0.3859 ± 0.0046 / 0.3842 *(+0.0017)* |
| **Dynamask** | 0.1240 ± 0.0105 / 0.1391 *(-0.0151)* | 0.1309 ± 0.0177 / 0.1640 *(-0.0331)* | 0.1944 ± 0.0149 / 0.2106 *(-0.0162)* |
| **CoRTX** | 0.1124 ± 0.0015 / 0.4983 *(-0.3859)* | 0.0590 ± 0.0003 / 0.3281 *(-0.2691)* | 0.6667 ± 0.0000 / 0.4711 *(+0.1956)* |
| **SGT + Grad**| 0.1436 ± 0.0066 / 0.3449 *(-0.2013)* | 0.0701 ± 0.0047 / 0.2133 *(-0.1432)* | 0.3874 ± 0.0115 / 0.3528 *(+0.0346)* |

### SeqComb-MV (Multivariate)

| Method | AUPRC | AUP | AUR |
| :--- | :---: | :---: | :---: |
| **TimeX** (Ours) | **0.3735 ± 0.0363** / 0.6878 *(-0.3143)* | **0.7308 ± 0.0407** / 0.8326 *(-0.1018)* | **0.2938 ± 0.0155** / 0.3872 *(-0.0934)* |
| **CoRTX** | 0.3623 ± 0.0007 / 0.3629 *(-0.0006)* | 0.5196 ± 0.0017 / 0.5625 *(-0.0429)* | 0.3453 ± 0.0006 / 0.3457 *(-0.0004)* |
| **IG** | 0.2764 ± 0.0143 / 0.3298 *(-0.0534)* | 0.6452 ± 0.0415 / 0.7483 *(-0.1031)* | 0.3389 ± 0.0317 / 0.2581 *(+0.0808)* |
| **Dynamask** | 0.2895 ± 0.0232 / 0.3136 *(-0.0241)* | 0.5342 ± 0.0572 / 0.5481 *(-0.0139)* | 0.2393 ± 0.0197 / 0.1953 *(+0.0440)* |
| **WinIT** | 0.2793 ± 0.0309 / 0.2809 *(-0.0016)* | 0.6071 ± 0.0814 / 0.7594 *(-0.1523)* | 0.2622 ± 0.0248 / 0.2077 *(+0.0545)* |
| **SGT + Grad**| 0.1012 ± 0.0048 / 0.4893 *(-0.3881)* | 0.2014 ± 0.0220 / 0.4970 *(-0.2956)* | 0.1518 ± 0.0411 / 0.4289 *(-0.2771)* |

---

## 5. Interpretation and uncertainty

The historical tables do not isolate the causes of differences. SeqComb-MV
predictor failures are confirmed, but TimeX's healthy folds also miss recall.
Broken temporal connectivity, active frozen-reference dropout and ineffective
clipping are additional demonstrated implementation defects. Their numerical
impact remains to be separated in the fixed-hyperparameter pilots.

Baseline numerical agreement must be assessed across all three metrics. For
example, SeqComb-MV WinIT is close in AUPRC but misses AUP by −0.1523 and AUR by
+0.0545. Dynamask and WinIT both use the predictor and are affected by its quality.
SGT trains its own classifier; longer training does not guarantee that its
attributions will match the paper. Full-test macro-F1 will be recorded alongside
SGT's attribution metrics.

CoRTX's local and released cross-view InfoNCE formulas agree in forward loss and
both input gradients to 1e-12 in the tested float64 cases. Replacing InfoNCE is
therefore not a supported fix. The remaining multivariate reconstruction
adaptation and LowVar saturation issue are documented in
[the audit](table2_deviations.md). A close metric match cannot resolve provenance.

Fold standard error is the primary uncertainty: standard deviation of fold
means with `ddof=1`, divided by the square root of the number of folds. The
historical pooled convention treats individual explanations as observations and
uses `ddof=0`; it is retained as explicitly labelled pooled uncertainty. These
are different estimands, and the much smaller pooled SE is not evidence of
stability across folds or training seeds. Seeds 43/44 are conditional follow-ups
for affected supported rows and must both be reported, without selecting the
closest seed.

## 6. Artifact & File References

- **Table 1 Local Logs (FreqShapes, Seed 0)**: [`results/table1/`](results/table1/)
  - Predictor: [`results/table1/freqshape_predictor_train.log`](results/table1/freqshape_predictor_train.log)
  - TimeX: [`results/table1/freqshape_ours_evaluation.log`](results/table1/freqshape_ours_evaluation.log)
  - IG: [`results/table1/freqshape_ig_evaluation.log`](results/table1/freqshape_ig_evaluation.log)
  - WinIT: [`results/table1/freqshape_winit_evaluation.log`](results/table1/freqshape_winit_evaluation.log)
- **Table 1 Full Run (Seed 42)**: [`timex_21712429/seed_42/results/`](timex_21712429/seed_42/results/) & [`results.md`](results.md)
- **Historical Table 2 Run (Seed 42)**: [`timex_table2_21846132/seed_42/results/`](timex_table2_21846132/seed_42/results/)
  - Summary: [`timex_table2_21846132/seed_42/results/table2_summary.md`](timex_table2_21846132/seed_42/results/table2_summary.md)
  - CSV format: [`timex_table2_21846132/seed_42/results/table2_summary.csv`](timex_table2_21846132/seed_42/results/table2_summary.csv)
  - Slurm Job Log: [`timex-table2-21846132.out`](timex-table2-21846132.out)
- **Workflows & Runner Instructions**:
  - [`experiments/TABLE1.md`](experiments/TABLE1.md)
  - [`experiments/TABLE2.md`](experiments/TABLE2.md)

## 7. Local validation and user-run GPU commands

```bash
PYTHONPATH=. uv run python -m unittest discover -s tests -p 'test_table2_repairs.py' -v
PYTHONPATH=. uv run python experiments/evaluation/diagnose_table2.py --output results/table2_cpu_diagnosis.json
bash -n run_table2.sh sj_timex_table2
./run_table2.sh --datasets seqcomb_mv --folds 1 --methods ours --dry-run
```

The diagnosis requires `dataset/` and the historical archive. The focused tests
use generated CPU tensors and temporary artifacts. No GPU job was submitted.
The [workflow](experiments/TABLE2.md#gpu-pilots-and-full-run) gives all pilot,
full-run, archived-resume, and conditional seed-43/44 submission commands.
