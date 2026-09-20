# Gate 0 report: prior_sweep run 20260917T142535Z_fbc712_gate0

Gains are relative LER reductions versus the physical prior on the same 4e6-shot test set (positive = the deformed prior is better). p-values: exact McNemar; Holm over the class-search family.

                                   cond       n  ler_phys  stale_gain    stale_p  uniform_gain  uniform_p  class_gain    class_p  class_p_holm                                                                           scales
 legacy_from_physical_rate|d7|T7|p0.002 4000000 0.0001543    0.006483     0.3877      0.006483     0.4545     0.04862    0.02241       0.08962                                              (1.0, 1.0, 1.0, 0.4548991719712601)
 legacy_from_physical_rate|d7|T7|p0.003 4000000  0.000789     0.01141   0.005491       0.01267   0.004325     0.03485    0.00106      0.008482 (1.2980862633468149, 1.6871381895535231, 1.2980862633468149, 0.5912378761718231)
       si1000:google2025|d5|T15|p0.0015 4000000  0.004858    0.002573    0.00432      0.003963   0.009979     0.03165  1.411e-19     1.834e-18 (1.2980862633468149, 1.2980862633468149, 1.6871381895535231, 0.7684389151674537)
legacy_from_physical_rate|d5|T15|p0.005 4000000   0.02808    0.009848  9.829e-52       0.01003  6.631e-56     0.03479  6.37e-213    2.038e-211 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d7|T7|p0.005 4000000  0.005737     0.01129  4.936e-13       0.01233  3.373e-12     0.03626  5.662e-37     9.626e-36 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d7|T7|p0.007 4000000    0.0204     0.01833  1.115e-59       0.01874  5.626e-69     0.03611  9.805e-81     2.255e-79 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.5912378761718231)
         si1000:google2025|d7|T7|p0.001 4000000 9.325e-05           0          1      -0.01072      0.125    -0.06434   0.007084       0.04014                               (1.0, 1.2980862633468149, 1.6871381895535231, 1.0)
        si1000:google2025|d5|T15|p0.003 4000000   0.03651    0.007129  2.568e-32      0.007615  2.396e-24     0.02535   4.77e-97      1.24e-95 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
        si1000:google2025|d7|T7|p0.0015 4000000 0.0005288      0.0104    0.03155       0.01229   0.006674     0.01371    0.05479        0.1096                               (1.0, 1.2980862633468149, 0.9987492177719088, 1.0)
         si1000:google2025|d7|T7|p0.002 4000000  0.001659           0          1      0.001055      0.746    0.002561     0.7038        0.7038 (1.2980862633468149, 1.2980862633468149, 1.6871381895535231, 0.7684389151674537)
 legacy_from_physical_rate|d5|T15|p0.01 4000000     0.163     0.02197          0       0.02167          0     0.03664          0             0 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d5|T5|p0.002 4000000 0.0006352    0.001968     0.3593      0.002361     0.2863     0.02007    0.00669       0.04014                (0.9987492177719088, 1.2980862633468149, 1.0, 0.7684389151674537)
  legacy_from_physical_rate|d7|T7|p0.01 4000000    0.0684     0.01657 9.942e-146       0.01745 9.964e-151     0.03031 1.311e-156    3.934e-155 (0.9987492177719088, 1.2980862633468149, 1.2980862633468149, 0.5912378761718231)
 legacy_from_physical_rate|d5|T5|p0.003 4000000   0.00211    0.004502  0.0004752      0.004739   3.67e-05      0.0327  1.164e-10     1.281e-09 (1.2980862633468149, 1.6871381895535231, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d5|T5|p0.005 4000000  0.009308    0.008246  9.007e-17      0.008434  7.785e-18     0.03583  2.747e-47      5.22e-46 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.5912378761718231)
 legacy_from_physical_rate|d5|T5|p0.007 4000000   0.02345     0.01548  1.318e-74       0.01609  8.008e-75      0.0361 4.459e-109    1.204e-107 (0.9987492177719088, 1.2980862633468149, 0.9987492177719088, 0.5912378761718231)
         si1000:google2025|d7|T7|p0.003 4000000  0.008214    0.006696  2.648e-06      0.006727  1.079e-06     0.01543  1.048e-08     1.048e-07 (1.2980862633468149, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
  legacy_from_physical_rate|d5|T5|p0.01 4000000   0.05937     0.02256 9.236e-241       0.02196 1.564e-208      0.0402 6.031e-270    2.051e-268 (0.9987492177719088, 1.2980862633468149, 1.2980862633468149, 0.5912378761718231)
         si1000:google2025|d5|T5|p0.001 4000000 0.0005443           0          1      0.002297     0.5327     0.02343    0.02658       0.08962                (1.0, 0.7684389151674537, 1.2980862633468149, 0.5912378761718231)
        si1000:google2025|d5|T5|p0.0015 4000000  0.001798    0.004589   0.006053      0.004589    0.02736     0.01446  0.0003501      0.003151                               (1.2980862633468149, 1.2980862633468149, 1.0, 1.0)
         si1000:google2025|d5|T5|p0.002 4000000  0.004221    0.002666     0.1377      0.005153   0.002097     0.01001   0.004549       0.03184                (1.0, 1.2980862633468149, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d5|T5|p0.015 4000000    0.1484     0.02294          0       0.02306          0     0.03532          0             0 (1.2980862633468149, 1.2980862633468149, 1.6871381895535231, 0.7684389151674537)
         si1000:google2025|d5|T5|p0.003 4000000   0.01349    0.005597  2.791e-10      0.007395  1.174e-10      0.0205  5.855e-35     9.368e-34                (1.0, 0.9987492177719088, 1.2980862633468149, 0.7684389151674537)
         si1000:google2025|d7|T7|p0.005 4000000   0.05368    0.004256  3.103e-09       0.00489  7.555e-15     0.01548  1.549e-59     3.254e-58                               (1.0, 1.0, 1.2980862633468149, 0.7684389151674537)
 legacy_from_physical_rate|d3|T9|p0.005 4000000   0.03378    0.001724    6.6e-20      0.001717  1.022e-19     0.05322          0             0                   (2.192793615511396, 2.192793615511396, 2.192793615511396, 1.0)
         si1000:google2025|d5|T5|p0.005 4000000   0.05416    0.007782  5.394e-40      0.008355  1.821e-33     0.02087 1.177e-123    3.413e-122 (0.9987492177719088, 0.9987492177719088, 1.2980862633468149, 0.7684389151674537)
  legacy_from_physical_rate|d5|T5|p0.02 4000000     0.248     0.01914          0       0.01924          0     0.02738          0             0 (1.2980862633468149, 1.6871381895535231, 1.6871381895535231, 0.7684389151674537)
  legacy_from_physical_rate|d3|T9|p0.01 4000000    0.1138    0.007175 3.817e-235      0.007175  1.05e-237     0.04916          0             0                               (2.192793615511396, 1.6871381895535231, 2.85, 1.0)
        si1000:google2025|d3|T9|p0.0015 4000000   0.01346     0.06087 1.581e-322      0.002638   2.29e-21     0.06922          0             0                              (1.2980862633468149, 1.2980862633468149, 2.85, 1.0)
 legacy_from_physical_rate|d3|T3|p0.002 4000000  0.002204           0          1             0          1     0.05398  7.638e-45     1.375e-43                 (2.192793615511396, 1.6871381895535231, 1.6871381895535231, 1.0)
 legacy_from_physical_rate|d3|T3|p0.003 4000000  0.004812    0.001247  0.0007173      0.001247  0.0007173     0.05153  7.239e-87      1.81e-85                  (2.192793615511396, 1.6871381895535231, 2.192793615511396, 1.0)
 legacy_from_physical_rate|d7|T7|p0.015 4000000     0.206     0.01276 6.793e-143       0.01243 3.894e-176     0.02209 1.727e-231    5.698e-230                (1.0, 1.2980862633468149, 1.2980862633468149, 0.5912378761718231)
 legacy_from_physical_rate|d3|T3|p0.005 4000000   0.01271    0.001514  2.972e-08      0.001514  2.494e-08     0.04801 3.529e-199    1.094e-197                  (2.192793615511396, 1.6871381895535231, 2.192793615511396, 1.0)
 legacy_from_physical_rate|d3|T3|p0.007 4000000   0.02369   0.0009922  8.907e-07     0.0009922  8.907e-07     0.04746          0             0                  (2.192793615511396, 1.6871381895535231, 2.192793615511396, 1.0)
         si1000:google2025|d3|T9|p0.003 4000000   0.04655           0          1      0.004528  7.396e-49    0.005833  4.975e-58     9.949e-57                                (1.0, 1.0, 2.192793615511396, 0.7684389151674537)
  legacy_from_physical_rate|d3|T3|p0.01 4000000   0.04511     0.00286  4.388e-35       0.00286  2.792e-35     0.04317          0             0                               (2.192793615511396, 1.6871381895535231, 2.85, 1.0)
         si1000:google2025|d3|T3|p0.001 4000000  0.002485     0.02002  4.645e-18      0.001509   0.005925     0.02535   2.14e-30     2.996e-29                 (1.2980862633468149, 1.2980862633468149, 2.192793615511396, 1.0)
 legacy_from_physical_rate|d3|T3|p0.015 4000000   0.08963     0.01001 8.587e-275      0.009422 2.945e-299      0.0446          0             0                 (2.192793615511396, 1.6871381895535231, 1.6871381895535231, 1.0)
        si1000:google2025|d3|T3|p0.0015 4000000  0.005471     0.02079  4.247e-44      0.002422  1.967e-08      0.0255   2.82e-67     6.204e-66               (1.6871381895535231, 1.6871381895535231, 2.85, 1.2980862633468149)
         si1000:google2025|d3|T3|p0.002 4000000  0.009472     0.01874   9.18e-62       0.01692  7.545e-47     0.02238  1.356e-82     3.253e-81                              (1.2980862633468149, 1.2980862633468149, 2.85, 1.0)
         si1000:google2025|d3|T3|p0.003 4000000   0.01995           0          1      0.001742  3.441e-06    0.002919  2.819e-15     3.383e-14                               (1.0, 1.0, 1.2980862633468149, 0.7684389151674537)
  legacy_from_physical_rate|d3|T3|p0.02 4000000    0.1392     0.01186          0       0.01147          0     0.04311          0             0                               (2.192793615511396, 1.6871381895535231, 2.85, 1.0)
         si1000:google2025|d3|T3|p0.005 4000000   0.05027    0.002735  7.359e-33      0.003581  1.936e-25    0.003869  1.388e-34     2.082e-33                                             (1.0, 1.2980862633468149, 2.85, 1.0)
  legacy_from_physical_rate|d7|T7|p0.02 4000000    0.3465    0.007431  8.575e-67      0.007221  2.823e-77     0.01305 1.345e-121    3.766e-120 (0.9987492177719088, 1.2980862633468149, 1.2980862633468149, 0.5912378761718231)

## Threshold location (physical prior)

legacy_from_physical_rate (T=d): physical-prior LER by p (rows) and d (cols)
d             3         5         7
p                                  
0.002 2.204e-03 6.352e-04 1.543e-04
0.003 4.812e-03 2.110e-03 7.890e-04
0.005 1.271e-02 9.308e-03 5.737e-03
0.007 2.369e-02 2.345e-02 2.040e-02
0.010 4.511e-02 5.937e-02 6.840e-02
0.015 8.963e-02 1.484e-01 2.060e-01
0.020 1.392e-01 2.480e-01 3.465e-01

legacy_from_physical_rate (T=3d): physical-prior LER by p (rows) and d (cols)
d             3         5
p                        
0.005 3.378e-02 2.808e-02
0.010 1.138e-01 1.630e-01

si1000:google2025 (T=d): physical-prior LER by p (rows) and d (cols)
d              3         5         7
p                                   
0.0010 2.485e-03 5.443e-04 9.325e-05
0.0015 5.471e-03 1.798e-03 5.288e-04
0.0020 9.472e-03 4.221e-03 1.659e-03
0.0030 1.995e-02 1.349e-02 8.214e-03
0.0050 5.027e-02 5.416e-02 5.368e-02

si1000:google2025 (T=3d): physical-prior LER by p (rows) and d (cols)
d              3         5
p                         
0.0015 1.346e-02 4.858e-03
0.0030 4.655e-02 3.651e-02

## D1 decomposition (stale p_cal=0.005 vs physical): offset-only vs structure-only gains

                                  cond  d1_stale005_gain  d1_offset_gain  d1_structure_gain
legacy_from_physical_rate|d7|T7|p0.002          -0.05835        -0.05835           0.003241
legacy_from_physical_rate|d7|T7|p0.003          -0.03042        -0.03105           0.002852
      si1000:google2025|d5|T15|p0.0015          -0.08754         -0.1088           0.008543
legacy_from_physical_rate|d7|T7|p0.007           0.01262         0.01303         -0.0006005
        si1000:google2025|d7|T7|p0.001           -0.1475         -0.1743          -0.008043
       si1000:google2025|d5|T15|p0.003          -0.02525        -0.03037            0.00315
       si1000:google2025|d7|T7|p0.0015           -0.1078         -0.1206           0.007092
        si1000:google2025|d7|T7|p0.002           -0.0589        -0.06418           0.001958
legacy_from_physical_rate|d5|T15|p0.01           0.01987         0.02039          -0.001271
legacy_from_physical_rate|d5|T5|p0.002          -0.01377        -0.01456         -0.0003935
 legacy_from_physical_rate|d7|T7|p0.01           0.01657         0.01738          -0.002189
legacy_from_physical_rate|d5|T5|p0.003         -0.008768       -0.009123                  0
legacy_from_physical_rate|d5|T5|p0.007           0.01119         0.01135         -0.0005543
        si1000:google2025|d7|T7|p0.003          -0.02502        -0.02861           0.001918
 legacy_from_physical_rate|d5|T5|p0.01           0.02093         0.02149          -0.001916
        si1000:google2025|d5|T5|p0.001           -0.1176         -0.1456           0.005512
       si1000:google2025|d5|T5|p0.0015          -0.07774        -0.09192           0.005423
        si1000:google2025|d5|T5|p0.002          -0.04934        -0.05657          0.0008293
legacy_from_physical_rate|d5|T5|p0.015           0.02294         0.02323          -0.005068
        si1000:google2025|d5|T5|p0.003          -0.02204        -0.02467           0.001316
 legacy_from_physical_rate|d5|T5|p0.02           0.01914         0.01926          -0.007301
 legacy_from_physical_rate|d3|T9|p0.01          0.006054        0.006071         -4.395e-06
       si1000:google2025|d3|T9|p0.0015           0.05658         -0.0104            0.06689
legacy_from_physical_rate|d3|T3|p0.002         -0.001134       -0.001134                  0
legacy_from_physical_rate|d3|T3|p0.003        -0.0005714      -0.0005714                  0
legacy_from_physical_rate|d7|T7|p0.015           0.01276         0.01271          -0.002652
legacy_from_physical_rate|d3|T3|p0.007                 0               0         -3.167e-05
        si1000:google2025|d3|T9|p0.003         -0.003325       -0.003325                  0
 legacy_from_physical_rate|d3|T3|p0.01          0.002494        0.002494                  0
        si1000:google2025|d3|T3|p0.001           0.01408       -0.009959            0.02364
legacy_from_physical_rate|d3|T3|p0.015           0.00933         0.00933          -0.001093
       si1000:google2025|d3|T3|p0.0015           0.01736       -0.006214            0.02367
        si1000:google2025|d3|T3|p0.002           0.01475         0.01475            0.02138
        si1000:google2025|d3|T3|p0.003         -0.003834       -0.003847         -1.253e-05
 legacy_from_physical_rate|d3|T3|p0.02           0.01147         0.01146          -0.002877
 legacy_from_physical_rate|d7|T7|p0.02          0.006545        0.006107          -0.002741

## Gate 0 decision inputs

- legacy_from_physical_rate|d7|T7|p0.002: class-search gain 4.86% (Holm p=0.09), best stale gain 0.65%, uniform gain 0.65%
- legacy_from_physical_rate|d7|T7|p0.003: class-search gain 3.49% (Holm p=0.0085), best stale gain 1.14%, uniform gain 1.27%
- si1000:google2025|d5|T15|p0.0015: class-search gain 3.16% (Holm p=1.8e-18), best stale gain 0.26%, uniform gain 0.40%
- legacy_from_physical_rate|d5|T15|p0.005: class-search gain 3.48% (Holm p=2e-211), best stale gain 0.98%, uniform gain 1.00%
- legacy_from_physical_rate|d7|T7|p0.005: class-search gain 3.63% (Holm p=9.6e-36), best stale gain 1.13%, uniform gain 1.23%
- legacy_from_physical_rate|d7|T7|p0.007: class-search gain 3.61% (Holm p=2.3e-79), best stale gain 1.83%, uniform gain 1.87%
- si1000:google2025|d7|T7|p0.001: class-search gain -6.43% (Holm p=0.04), best stale gain 0.00%, uniform gain -1.07%
- si1000:google2025|d5|T15|p0.003: class-search gain 2.54% (Holm p=1.2e-95), best stale gain 0.71%, uniform gain 0.76%
- si1000:google2025|d7|T7|p0.0015: class-search gain 1.37% (Holm p=0.11), best stale gain 1.04%, uniform gain 1.23%
- si1000:google2025|d7|T7|p0.002: class-search gain 0.26% (Holm p=0.7), best stale gain 0.00%, uniform gain 0.11%
- legacy_from_physical_rate|d5|T15|p0.01: class-search gain 3.66% (Holm p=0), best stale gain 2.20%, uniform gain 2.17%
- legacy_from_physical_rate|d5|T5|p0.002: class-search gain 2.01% (Holm p=0.04), best stale gain 0.20%, uniform gain 0.24%
- legacy_from_physical_rate|d7|T7|p0.01: class-search gain 3.03% (Holm p=3.9e-155), best stale gain 1.66%, uniform gain 1.74%
- legacy_from_physical_rate|d5|T5|p0.003: class-search gain 3.27% (Holm p=1.3e-09), best stale gain 0.45%, uniform gain 0.47%
- legacy_from_physical_rate|d5|T5|p0.005: class-search gain 3.58% (Holm p=5.2e-46), best stale gain 0.82%, uniform gain 0.84%
- legacy_from_physical_rate|d5|T5|p0.007: class-search gain 3.61% (Holm p=1.2e-107), best stale gain 1.55%, uniform gain 1.61%
- si1000:google2025|d7|T7|p0.003: class-search gain 1.54% (Holm p=1e-07), best stale gain 0.67%, uniform gain 0.67%
- legacy_from_physical_rate|d5|T5|p0.01: class-search gain 4.02% (Holm p=2.1e-268), best stale gain 2.26%, uniform gain 2.20%
- si1000:google2025|d5|T5|p0.001: class-search gain 2.34% (Holm p=0.09), best stale gain 0.00%, uniform gain 0.23%
- si1000:google2025|d5|T5|p0.0015: class-search gain 1.45% (Holm p=0.0032), best stale gain 0.46%, uniform gain 0.46%
- si1000:google2025|d5|T5|p0.002: class-search gain 1.00% (Holm p=0.032), best stale gain 0.27%, uniform gain 0.52%
- legacy_from_physical_rate|d5|T5|p0.015: class-search gain 3.53% (Holm p=0), best stale gain 2.29%, uniform gain 2.31%
- si1000:google2025|d5|T5|p0.003: class-search gain 2.05% (Holm p=9.4e-34), best stale gain 0.56%, uniform gain 0.74%
- si1000:google2025|d7|T7|p0.005: class-search gain 1.55% (Holm p=3.3e-58), best stale gain 0.43%, uniform gain 0.49%
- si1000:google2025|d5|T5|p0.005: class-search gain 2.09% (Holm p=3.4e-122), best stale gain 0.78%, uniform gain 0.84%
- legacy_from_physical_rate|d5|T5|p0.02: class-search gain 2.74% (Holm p=0), best stale gain 1.91%, uniform gain 1.92%
- legacy_from_physical_rate|d7|T7|p0.015: class-search gain 2.21% (Holm p=5.7e-230), best stale gain 1.28%, uniform gain 1.24%
- legacy_from_physical_rate|d7|T7|p0.02: class-search gain 1.31% (Holm p=3.8e-120), best stale gain 0.74%, uniform gain 0.72%