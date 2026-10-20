========================================================================================================
CAUSATION JUDGE (120-ARG, 8-MODEL) -- PAIRED STATISTICS REPORT
Source figure: docs/manuscript/figures/causation_judge_120_FIXED.pdf
alpha=0.05 | bootstrap n=10000 (seed=42) | low-coverage threshold n<90
========================================================================================================

CONDITION MEANS vs. PUBLISHED FIGURE
--------------------------------------------------------------------------------------------------------
Model      Condition      n     mean      SEM     pub.       diff   status
LLaMA-2    vanilla      120    0.560    0.015    0.560    -0.0000     PASS
LLaMA-2    baseline     119    0.505    0.017    0.505    -0.0002     PASS
LLaMA-2    noise        118    0.521    0.015    0.521     0.0005     PASS
LLaMA-2    oracle       117    0.583    0.017    0.583    -0.0001     PASS
LLaMA-3    vanilla      117    0.565    0.013    0.565     0.0002     PASS
LLaMA-3    baseline     120    0.545    0.013    0.545     0.0000     PASS
LLaMA-3    noise        109    0.555    0.015    0.555     0.0004     PASS
LLaMA-3    oracle       119    0.566    0.013    0.566    -0.0002     PASS
Mistral    vanilla      120    0.591    0.013    0.591     0.0001     PASS
Mistral    baseline     120    0.539    0.014    0.539    -0.0001     PASS
Mistral    noise         90    0.534    0.016    0.534     0.0001     PASS
Mistral    oracle        79    0.603    0.016    0.603     0.0004     PASS
OLMo       vanilla      120    0.593    0.014    0.593    -0.0002     PASS
OLMo       baseline     120    0.594    0.014    0.594    -0.0001     PASS
OLMo       noise        119    0.557    0.017    0.557     0.0001     PASS
OLMo       oracle       120    0.641    0.014    0.641    -0.0004     PASS
Qwen3      vanilla      114    0.573    0.016    0.573     0.0001     PASS
Qwen3      baseline     120    0.571    0.014    0.571     0.0001     PASS
Qwen3      noise         84    0.498    0.020    0.498     0.0004     PASS
Qwen3      oracle        70    0.597    0.018    0.597     0.0001     PASS
Qwen2.5    vanilla      120    0.609    0.013    0.609     0.0004     PASS
Qwen2.5    baseline     120    0.576    0.014    0.576    -0.0004     PASS
Qwen2.5    noise         95    0.533    0.015    0.533    -0.0004     PASS
Qwen2.5    oracle        96    0.558    0.012    0.558    -0.0004     PASS
GPT-4o     vanilla      120    0.681    0.014    0.681     0.0001     PASS
GPT-4o     baseline     120    0.635    0.013    0.635    -0.0000     PASS
GPT-4o     noise        120    0.517    0.018    0.517     0.0002     PASS
GPT-4o     oracle       120    0.671    0.014    0.671     0.0001     PASS
GPT-5.5    vanilla      120    0.690    0.013    0.690     0.0000     PASS
GPT-5.5    baseline     120    0.698    0.013    0.698     0.0003     PASS
GPT-5.5    noise        120    0.553    0.020    0.553    -0.0002     PASS
GPT-5.5    oracle       120    0.704    0.012    0.704     0.0004     PASS

  Validation: 32/32 condition means PASS

CONTRASTS (all six condition pairs; every condition is scored by the identical
GPT-5.5 judge mechanism and paired at the same argument level, so all are
directly comparable regardless of which pipeline generated the underlying output)
--------------------------------------------------------------------------------------------------------
Contrast             Model         n mean_diff   t-test: CI95 / p (Holm/BH)         d_z           Wilcoxon: p (Holm/BH)      r_rb           lowN
oracle_vs_noise      LLaMA-2     115    +0.062   [+0.026,+0.098] p=0.0010         0.316           p=0.0020 H=0.006/BH=0.003   0.347               
                         t-test Holm/BH: H=0.003/BH=0.001 | shapiro(diff) p=0.102
oracle_vs_noise      LLaMA-3     108    +0.006   [-0.025,+0.037] p=0.6918         0.038           p=0.6246 H=0.625/BH=0.625   0.059               
                         t-test Holm/BH: H=0.692/BH=0.692 | shapiro(diff) p=0.022  [diffs non-normal -> favor Wilcoxon]
oracle_vs_noise      Mistral      60    +0.086   [+0.047,+0.124] p=0.0000         0.581           p=0.0001 H=0.000/BH=0.000   0.695            LOW
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.002  [diffs non-normal -> favor Wilcoxon]
oracle_vs_noise      OLMo        119    +0.083   [+0.052,+0.114] p=0.0000         0.487           p=0.0000 H=0.000/BH=0.000   0.538               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.177
oracle_vs_noise      Qwen3        49    +0.099   [+0.049,+0.150] p=0.0002         0.566           p=0.0006 H=0.002/BH=0.001   0.601            LOW
                         t-test Holm/BH: H=0.001/BH=0.000 | shapiro(diff) p=0.193
oracle_vs_noise      Qwen2.5      79    +0.031   [-0.001,+0.064] p=0.0582         0.216           p=0.0435 H=0.087/BH=0.050   0.275            LOW
                         t-test Holm/BH: H=0.116/BH=0.066 | shapiro(diff) p=0.005  [diffs non-normal -> favor Wilcoxon]
oracle_vs_noise      GPT-4o      120    +0.154   [+0.119,+0.188] p=0.0000         0.807           p=0.0000 H=0.000/BH=0.000   0.815               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.016  [diffs non-normal -> favor Wilcoxon]
oracle_vs_noise      GPT-5.5     120    +0.152   [+0.113,+0.190] p=0.0000         0.717           p=0.0000 H=0.000/BH=0.000   0.752               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.000  [diffs non-normal -> favor Wilcoxon]
oracle_vs_vanilla    LLaMA-2     117    +0.022   [-0.013,+0.058] p=0.2178         0.115           p=0.2664 H=1.000/BH=0.463   0.125               
                         t-test Holm/BH: H=1.000/BH=0.366 | shapiro(diff) p=0.211
oracle_vs_vanilla    LLaMA-3     116    -0.006   [-0.035,+0.023] p=0.6949        -0.037           p=0.3459 H=1.000/BH=0.463  -0.111               
                         t-test Holm/BH: H=1.000/BH=0.695 | shapiro(diff) p=0.001  [diffs non-normal -> favor Wilcoxon]
oracle_vs_vanilla    Mistral      79    +0.009   [-0.024,+0.042] p=0.5793         0.063           p=0.6156 H=1.000/BH=0.616   0.071            LOW
                         t-test Holm/BH: H=1.000/BH=0.662 | shapiro(diff) p=0.041  [diffs non-normal -> favor Wilcoxon]
oracle_vs_vanilla    OLMo        120    +0.048   [+0.017,+0.079] p=0.0027         0.280           p=0.0014 H=0.011/BH=0.006   0.356               
                         t-test Holm/BH: H=0.019/BH=0.011 | shapiro(diff) p=0.029  [diffs non-normal -> favor Wilcoxon]
oracle_vs_vanilla    Qwen3        67    +0.035   [-0.012,+0.081] p=0.1400         0.183           p=0.1248 H=0.749/BH=0.333   0.236            LOW
                         t-test Holm/BH: H=0.840/BH=0.366 | shapiro(diff) p=0.107
oracle_vs_vanilla    Qwen2.5      96    -0.058   [-0.090,-0.026] p=0.0005        -0.370           p=0.0014 H=0.011/BH=0.006  -0.404               
                         t-test Holm/BH: H=0.004/BH=0.004 | shapiro(diff) p=0.058
oracle_vs_vanilla    GPT-4o      120    -0.010   [-0.036,+0.016] p=0.4475        -0.070           p=0.4250 H=1.000/BH=0.486  -0.096               
                         t-test Holm/BH: H=1.000/BH=0.597 | shapiro(diff) p=0.009  [diffs non-normal -> favor Wilcoxon]
oracle_vs_vanilla    GPT-5.5     120    +0.014   [-0.009,+0.038] p=0.2289         0.110           p=0.3475 H=1.000/BH=0.463   0.108               
                         t-test Holm/BH: H=1.000/BH=0.366 | shapiro(diff) p=0.004  [diffs non-normal -> favor Wilcoxon]
noise_vs_vanilla     LLaMA-2     118    -0.037   [-0.070,-0.004] p=0.0274        -0.206           p=0.0203 H=0.061/BH=0.027  -0.262               
                         t-test Holm/BH: H=0.082/BH=0.036 | shapiro(diff) p=0.135
noise_vs_vanilla     LLaMA-3     106    -0.005   [-0.035,+0.025] p=0.7375        -0.033           p=0.8424 H=0.842/BH=0.842  -0.023               
                         t-test Holm/BH: H=0.738/BH=0.738 | shapiro(diff) p=0.085
noise_vs_vanilla     Mistral      90    -0.067   [-0.098,-0.035] p=0.0001        -0.445           p=0.0001 H=0.000/BH=0.000  -0.540               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.026  [diffs non-normal -> favor Wilcoxon]
noise_vs_vanilla     OLMo        119    -0.038   [-0.075,-0.001] p=0.0435        -0.187           p=0.0490 H=0.098/BH=0.056  -0.227               
                         t-test Holm/BH: H=0.087/BH=0.050 | shapiro(diff) p=0.107
noise_vs_vanilla     Qwen3        82    -0.087   [-0.127,-0.047] p=0.0000        -0.482           p=0.0001 H=0.000/BH=0.000  -0.517            LOW
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.076
noise_vs_vanilla     Qwen2.5      95    -0.088   [-0.122,-0.055] p=0.0000        -0.534           p=0.0000 H=0.000/BH=0.000  -0.603               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.075
noise_vs_vanilla     GPT-4o      120    -0.164   [-0.199,-0.129] p=0.0000        -0.842           p=0.0000 H=0.000/BH=0.000  -0.834               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.017  [diffs non-normal -> favor Wilcoxon]
noise_vs_vanilla     GPT-5.5     120    -0.137   [-0.177,-0.097] p=0.0000        -0.623           p=0.0000 H=0.000/BH=0.000  -0.689               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.000  [diffs non-normal -> favor Wilcoxon]
oracle_vs_baseline   LLaMA-2     116    +0.080   [+0.036,+0.124] p=0.0005         0.332           p=0.0023 H=0.018/BH=0.008   0.336               
                         t-test Holm/BH: H=0.004/BH=0.004 | shapiro(diff) p=0.368
oracle_vs_baseline   LLaMA-3     119    +0.019   [-0.011,+0.049] p=0.2140         0.115           p=0.3207 H=1.000/BH=0.428   0.119               
                         t-test Holm/BH: H=0.856/BH=0.342 | shapiro(diff) p=0.003  [diffs non-normal -> favor Wilcoxon]
oracle_vs_baseline   Mistral      79    +0.062   [+0.026,+0.098] p=0.0009         0.389           p=0.0038 H=0.021/BH=0.008   0.426            LOW
                         t-test Holm/BH: H=0.006/BH=0.004 | shapiro(diff) p=0.001  [diffs non-normal -> favor Wilcoxon]
oracle_vs_baseline   OLMo        120    +0.047   [+0.018,+0.075] p=0.0014         0.299           p=0.0031 H=0.021/BH=0.008   0.337               
                         t-test Holm/BH: H=0.008/BH=0.004 | shapiro(diff) p=0.061
oracle_vs_baseline   Qwen3        70    +0.010   [-0.030,+0.049] p=0.6338         0.057           p=0.4101 H=1.000/BH=0.469   0.125            LOW
                         t-test Holm/BH: H=1.000/BH=0.642 | shapiro(diff) p=0.064
oracle_vs_baseline   Qwen2.5      96    -0.017   [-0.051,+0.017] p=0.3314        -0.100           p=0.2830 H=1.000/BH=0.428  -0.135               
                         t-test Holm/BH: H=0.994/BH=0.442 | shapiro(diff) p=0.158
oracle_vs_baseline   GPT-4o      120    +0.036   [+0.013,+0.060] p=0.0029         0.277           p=0.0030 H=0.021/BH=0.008   0.354               
                         t-test Holm/BH: H=0.015/BH=0.006 | shapiro(diff) p=0.002  [diffs non-normal -> favor Wilcoxon]
oracle_vs_baseline   GPT-5.5     120    +0.006   [-0.020,+0.032] p=0.6425         0.042           p=0.7570 H=1.000/BH=0.757   0.036               
                         t-test Holm/BH: H=1.000/BH=0.642 | shapiro(diff) p=0.006  [diffs non-normal -> favor Wilcoxon]
noise_vs_baseline    LLaMA-2     117    +0.018   [-0.019,+0.055] p=0.3296         0.091           p=0.4559 H=0.912/BH=0.513   0.082               
                         t-test Holm/BH: H=0.790/BH=0.377 | shapiro(diff) p=0.355
noise_vs_baseline    LLaMA-3     109    +0.009   [-0.017,+0.035] p=0.4806         0.068           p=0.5131 H=0.912/BH=0.513   0.080               
                         t-test Holm/BH: H=0.790/BH=0.481 | shapiro(diff) p=0.025  [diffs non-normal -> favor Wilcoxon]
noise_vs_baseline    Mistral      90    -0.017   [-0.047,+0.013] p=0.2635        -0.119           p=0.2187 H=0.656/BH=0.292  -0.163               
                         t-test Holm/BH: H=0.790/BH=0.351 | shapiro(diff) p=0.078
noise_vs_baseline    OLMo        119    -0.037   [-0.070,-0.003] p=0.0331        -0.198           p=0.0392 H=0.157/BH=0.063  -0.237               
                         t-test Holm/BH: H=0.132/BH=0.053 | shapiro(diff) p=0.091
noise_vs_baseline    Qwen3        84    -0.073   [-0.114,-0.032] p=0.0007        -0.385           p=0.0004 H=0.002/BH=0.001  -0.501            LOW
                         t-test Holm/BH: H=0.004/BH=0.002 | shapiro(diff) p=0.057
noise_vs_baseline    Qwen2.5      95    -0.043   [-0.078,-0.008] p=0.0170        -0.249           p=0.0265 H=0.132/BH=0.053  -0.284               
                         t-test Holm/BH: H=0.085/BH=0.034 | shapiro(diff) p=0.057
noise_vs_baseline    GPT-4o      120    -0.118   [-0.151,-0.085] p=0.0000        -0.642           p=0.0000 H=0.000/BH=0.000  -0.666               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.024  [diffs non-normal -> favor Wilcoxon]
noise_vs_baseline    GPT-5.5     120    -0.146   [-0.187,-0.105] p=0.0000        -0.642           p=0.0000 H=0.000/BH=0.000  -0.693               
                         t-test Holm/BH: H=0.000/BH=0.000 | shapiro(diff) p=0.000  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  LLaMA-2     119    +0.057   [+0.017,+0.096] p=0.0050         0.262           p=0.0070 H=0.042/BH=0.019   0.300               
                         t-test Holm/BH: H=0.030/BH=0.013 | shapiro(diff) p=0.215
vanilla_vs_baseline  LLaMA-3     117    +0.021   [-0.006,+0.048] p=0.1233         0.144           p=0.0945 H=0.378/BH=0.151   0.197               
                         t-test Holm/BH: H=0.493/BH=0.197 | shapiro(diff) p=0.013  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  Mistral     120    +0.052   [+0.026,+0.078] p=0.0001         0.366           p=0.0002 H=0.002/BH=0.002   0.430               
                         t-test Holm/BH: H=0.001/BH=0.001 | shapiro(diff) p=0.009  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  OLMo        120    -0.001   [-0.032,+0.029] p=0.9427        -0.007           p=0.9677 H=1.000/BH=0.968  -0.004               
                         t-test Holm/BH: H=1.000/BH=0.970 | shapiro(diff) p=0.020  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  Qwen3       114    +0.001   [-0.030,+0.032] p=0.9703         0.003           p=0.7907 H=1.000/BH=0.904   0.031               
                         t-test Holm/BH: H=1.000/BH=0.970 | shapiro(diff) p=0.014  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  Qwen2.5     120    +0.034   [+0.005,+0.063] p=0.0211         0.213           p=0.0306 H=0.153/BH=0.061   0.249               
                         t-test Holm/BH: H=0.106/BH=0.042 | shapiro(diff) p=0.131
vanilla_vs_baseline  GPT-4o      120    +0.046   [+0.021,+0.072] p=0.0005         0.328           p=0.0007 H=0.005/BH=0.003   0.403               
                         t-test Holm/BH: H=0.003/BH=0.002 | shapiro(diff) p=0.012  [diffs non-normal -> favor Wilcoxon]
vanilla_vs_baseline  GPT-5.5     120    -0.008   [-0.036,+0.019] p=0.5526        -0.054           p=0.5961 H=1.000/BH=0.795  -0.065               
                         t-test Holm/BH: H=1.000/BH=0.737 | shapiro(diff) p=0.001  [diffs non-normal -> favor Wilcoxon]

NOTES
--------------------------------------------------------------------------------------------------------
  - Noise = the expert (A1) annotation of the next argument in the run, (i+1) mod N, given as pre-analysis notes in the same format as Oracle. The same rule is used for all eight models.
  - Baseline is the model's standard full-pipeline output rather than a --condition arm of the Vanilla/Noise/Oracle harness, generated by a separate script on a different date. It is scored by the identical GPT-5.5 judge mechanism and paired at the same argument level as the other three conditions, so contrasts against it are directly comparable -- which pipeline produced a given condition's model output does not affect the validity of a paired comparison on the resulting scores.
  - SEM in the condition-means table uses ddof=0, matching the source figure's own error-bar formula exactly (np.std(sc)/sqrt(n)).
  - Missingness varies substantially by model/condition (model_missing/gold_missing filtering) -- Mistral and Qwen3 drop to ~60-90 paired observations under Noise/Oracle contrasts; see the low_coverage flag (n below 90) on each row.
  - docs/manuscript/figures/causation_judge_120_FIXED.pdf on disk may be a stale 2026-09-25 regression (wrong baseline-sourcing logic); this script reads the underlying judge JSON directly, never that cached PDF, and validates against the figure's own printed values below.

========================================================================================================