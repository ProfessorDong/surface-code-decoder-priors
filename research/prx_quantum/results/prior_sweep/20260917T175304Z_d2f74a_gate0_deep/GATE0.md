# Gate 0 report: prior_sweep run 20260917T175304Z_d2f74a_gate0_deep

Gains are relative LER reductions versus the physical prior on the same 4e6-shot test set (positive = the deformed prior is better). p-values: exact McNemar; Holm over the class-search family.

                            cond        n  ler_phys  stale_gain    stale_p  uniform_gain  uniform_p  class_gain    class_p  class_p_holm                                                                           scales
si1000:google2025|d5|T15|p0.0015 16000000  0.004835    0.003051  5.351e-11      0.003839  1.986e-10     0.03088  8.519e-87     7.667e-86 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 si1000:google2025|d5|T15|p0.002 16000000   0.01137    0.001259    0.02204      0.005804   1.37e-31     0.02764 3.852e-162    4.237e-161 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
si1000:google2025|d5|T25|p0.0015 16000000  0.007868    0.003527  1.645e-23      0.004512  4.185e-22     0.03688 4.864e-199    5.836e-198 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 si1000:google2025|d5|T25|p0.002 16000000    0.0184   0.0003125     0.4793      0.005296  4.767e-42     0.02762 7.954e-261    1.034e-259                               (1.0, 0.9987492177719088, 1.0, 0.5912378761718231)
 si1000:google2025|d7|T7|p0.0015 16000000 0.0005142    0.006685   0.002686      0.006564   0.002237      0.0124   0.009749      0.009749 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.7684389151674537)
si1000:google2025|d7|T21|p0.0015 16000000  0.001468    0.003789   0.006105      0.003789   0.003941     0.01665  1.054e-08     2.107e-08 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.7684389151674537)
 si1000:google2025|d5|T15|p0.003 16000000   0.03669    0.007471 8.312e-140      0.008403 1.653e-113     0.02601          0             0 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
  si1000:google2025|d7|T7|p0.002 16000000  0.001653    0.004954    0.00499      0.005711   0.001221     0.02057  1.024e-22     4.097e-22                (0.9987492177719088, 1.0, 0.9987492177719088, 0.7684389151674537)
 si1000:google2025|d5|T5|p0.0015 16000000  0.001826    0.003321  9.071e-06      0.004451  4.003e-06     0.02421  1.445e-21     4.334e-21                               (1.0, 0.9987492177719088, 1.0, 0.5912378761718231)
  si1000:google2025|d5|T5|p0.002 16000000  0.004167     0.00252   0.004402      0.004904  4.336e-09     0.01602  6.037e-23     3.018e-22 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.7684389151674537)
 si1000:google2025|d5|T25|p0.003 16000000   0.05854    0.007159 4.496e-190      0.007747 1.428e-145     0.02726          0             0 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
  si1000:google2025|d5|T5|p0.003 16000000   0.01357    0.006282  4.735e-47      0.006987  5.716e-35     0.01731  1.929e-71     1.543e-70 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 si1000:google2025|d7|T21|p0.002 16000000  0.004788    0.003433   0.001093      0.004438  3.008e-08     0.01958  3.087e-34     2.161e-33                               (1.0, 1.2980862633468149, 1.0, 0.7684389151674537)
  si1000:google2025|d7|T7|p0.003 16000000  0.008318    0.006357  3.804e-19      0.006665  2.268e-22     0.01484      2e-29       1.2e-28 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 si1000:google2025|d7|T21|p0.003 16000000   0.02422    0.006019   3.25e-42      0.006306  2.416e-49     0.02111 2.312e-157    2.312e-156 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)

## Threshold location (physical prior)

si1000:google2025 (T=d): physical-prior LER by p (rows) and d (cols)
d              5         7
p                         
0.0015 1.826e-03 5.142e-04
0.0020 4.167e-03 1.653e-03
0.0030 1.357e-02 8.318e-03

si1000:google2025 (T=3d): physical-prior LER by p (rows) and d (cols)
d              5         7
p                         
0.0015 4.835e-03 1.468e-03
0.0020 1.137e-02 4.788e-03
0.0030 3.669e-02 2.422e-02

## D1 decomposition (stale p_cal=0.005 vs physical): offset-only vs structure-only gains

                            cond  d1_stale005_gain  d1_offset_gain  d1_structure_gain
si1000:google2025|d5|T15|p0.0015          -0.09115         -0.1133           0.008286
 si1000:google2025|d5|T15|p0.002          -0.06095        -0.07583           0.002589
si1000:google2025|d5|T25|p0.0015          -0.09134         -0.1173            0.01016
 si1000:google2025|d5|T25|p0.002          -0.06409        -0.08016           0.003085
 si1000:google2025|d7|T7|p0.0015           -0.1119         -0.1226           0.001702
si1000:google2025|d7|T21|p0.0015           -0.1142         -0.1226           0.003832
 si1000:google2025|d5|T15|p0.003          -0.02581        -0.03046           0.003351
  si1000:google2025|d7|T7|p0.002          -0.06777        -0.07723            0.00208
 si1000:google2025|d5|T5|p0.0015           -0.0745        -0.08525           0.006265
  si1000:google2025|d5|T5|p0.002          -0.05421        -0.06047           0.001845
 si1000:google2025|d5|T25|p0.003          -0.02469        -0.02953           0.003471
  si1000:google2025|d5|T5|p0.003          -0.02292        -0.02536           0.002179
 si1000:google2025|d7|T21|p0.002          -0.07162        -0.07861           0.002741
  si1000:google2025|d7|T7|p0.003          -0.02434        -0.02783           0.001878
 si1000:google2025|d7|T21|p0.003           -0.0246        -0.02822           0.002314

## Gate 0 decision inputs

- si1000:google2025|d5|T15|p0.0015: class-search gain 3.09% (Holm p=7.7e-86), best stale gain 0.31%, uniform gain 0.38%
- si1000:google2025|d5|T15|p0.002: class-search gain 2.76% (Holm p=4.2e-161), best stale gain 0.13%, uniform gain 0.58%
- si1000:google2025|d5|T25|p0.0015: class-search gain 3.69% (Holm p=5.8e-198), best stale gain 0.35%, uniform gain 0.45%
- si1000:google2025|d5|T25|p0.002: class-search gain 2.76% (Holm p=1e-259), best stale gain 0.03%, uniform gain 0.53%
- si1000:google2025|d7|T7|p0.0015: class-search gain 1.24% (Holm p=0.0097), best stale gain 0.67%, uniform gain 0.66%
- si1000:google2025|d7|T21|p0.0015: class-search gain 1.66% (Holm p=2.1e-08), best stale gain 0.38%, uniform gain 0.38%
- si1000:google2025|d5|T15|p0.003: class-search gain 2.60% (Holm p=0), best stale gain 0.75%, uniform gain 0.84%
- si1000:google2025|d7|T7|p0.002: class-search gain 2.06% (Holm p=4.1e-22), best stale gain 0.50%, uniform gain 0.57%
- si1000:google2025|d5|T5|p0.0015: class-search gain 2.42% (Holm p=4.3e-21), best stale gain 0.33%, uniform gain 0.45%
- si1000:google2025|d5|T5|p0.002: class-search gain 1.60% (Holm p=3e-22), best stale gain 0.25%, uniform gain 0.49%
- si1000:google2025|d5|T25|p0.003: class-search gain 2.73% (Holm p=0), best stale gain 0.72%, uniform gain 0.77%
- si1000:google2025|d5|T5|p0.003: class-search gain 1.73% (Holm p=1.5e-70), best stale gain 0.63%, uniform gain 0.70%
- si1000:google2025|d7|T21|p0.002: class-search gain 1.96% (Holm p=2.2e-33), best stale gain 0.34%, uniform gain 0.44%
- si1000:google2025|d7|T7|p0.003: class-search gain 1.48% (Holm p=1.2e-28), best stale gain 0.64%, uniform gain 0.67%
- si1000:google2025|d7|T21|p0.003: class-search gain 2.11% (Holm p=2.3e-156), best stale gain 0.60%, uniform gain 0.63%