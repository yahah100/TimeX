# Table 1 Reproduction Results

This document compares the completed five-fold reproduction run (base seed 42)
with the results reported in the TimeX paper. The run completed successfully for
TimeX, Integrated Gradients (IG), Dynamask, and WinIT on FreqShapes and
SeqComb-UV. CoRTX and SGT + Grad were not rerun because their committed scripts
do not currently support the complete five-fold workflow.

All metrics are reported as `reproduction / paper (difference)`, where the
difference is reproduction minus paper. Higher values are better.

| Dataset | Method | AUPRC | AUP | AUR |
|---|---|---:|---:|---:|
| FreqShapes | TimeX | 0.8401 / 0.8324 (+0.0077) | 0.7440 / 0.7219 (+0.0221) | 0.6309 / 0.6381 (-0.0072) |
| FreqShapes | IG | 0.7846 / 0.7516 (+0.0330) | 0.7290 / 0.6912 (+0.0378) | 0.5777 / 0.5975 (-0.0198) |
| FreqShapes | Dynamask | 0.2415 / 0.2201 (+0.0214) | 0.3391 / 0.2952 (+0.0439) | 0.4949 / 0.5037 (-0.0088) |
| FreqShapes | WinIT | 0.5048 / 0.5071 (-0.0023) | 0.5611 / 0.5546 (+0.0065) | 0.4494 / 0.4557 (-0.0063) |
| SeqComb-UV | TimeX | 0.6831 / 0.7124 (-0.0293) | 0.9022 / 0.9411 (-0.0389) | 0.2989 / 0.3380 (-0.0391) |
| SeqComb-UV | IG | 0.5089 / 0.5760 (-0.0671) | 0.7501 / 0.8157 (-0.0656) | 0.3415 / 0.2868 (+0.0547) |
| SeqComb-UV | Dynamask | 0.4363 / 0.4421 (-0.0058) | 0.8727 / 0.8782 (-0.0055) | 0.1047 / 0.1029 (+0.0018) |
| SeqComb-UV | WinIT | 0.4577 / 0.4568 (+0.0009) | 0.7718 / 0.7872 (-0.0154) | 0.2426 / 0.2253 (+0.0173) |

## Summary

The reproduction broadly supports the paper's main result. TimeX remains the
best of the four rerun methods on five of the six attribution metrics. The one
ranking change is SeqComb-UV AUR, where IG scores 0.3415 and TimeX scores
0.2989; the paper reports 0.2868 for IG and 0.3380 for TimeX.

- FreqShapes reproduces well. TimeX is 0.0077 higher on AUPRC and 0.0221
  higher on AUP, while AUR is 0.0072 lower than the paper.
- SeqComb-UV is weaker for TimeX. Its AUPRC, AUP, and AUR are lower by 0.0293,
  0.0389, and 0.0391, respectively.
- Dynamask and WinIT are particularly close to the published SeqComb-UV
  results. IG differs more substantially, with lower AUPRC and AUP but higher
  AUR in the reproduction.

## Predictor Performance

The reference transformer F1 scores provide useful context:

| Dataset | Reproduction mean F1 | Paper mean F1 | Difference |
|---|---:|---:|---:|
| FreqShapes | 0.9740 | 0.9716 | +0.0024 |
| SeqComb-UV | 0.9184 | 0.9415 | -0.0231 |

The weaker SeqComb-UV predictor is a plausible contributor to the attribution
gap, although this run alone does not establish a causal relationship.

## Intersection over Union

The paper reports SeqComb-UV IoU for TimeX, IG, and Dynamask in Table 10. The
reproduction preserves the published ranking:

| Method | Reproduction IoU | Paper IoU | Difference |
|---|---:|---:|---:|
| TimeX | 0.4943 | 0.5214 | -0.0271 |
| IG | 0.3183 | 0.3750 | -0.0567 |
| Dynamask | 0.2940 | 0.2958 | -0.0018 |

## Statistical Caveat

The paper describes its error bars as standard errors across five folds. The
current evaluator instead concatenates per-sample metric values from every fold
and computes `std / sqrt(n)` over that pooled collection. Consequently, the
reported means and method rankings are safer to compare directly than the
`+/-` values; the latter should not be used for formal significance claims
without first aligning the aggregation procedures.

## Sources

- Reproduction output: [`results/logs/timex-21712429.out`](results/logs/timex-21712429.out)
- Paper: [`2306.02109v2.pdf`](2306.02109v2.pdf), especially Tables 1, 10, and 15
- Evaluation aggregation: [`experiments/evaluation/saliency_exp_synth.py`](experiments/evaluation/saliency_exp_synth.py)
- Reproduction workflow: [`experiments/TABLE1.md`](experiments/TABLE1.md)
