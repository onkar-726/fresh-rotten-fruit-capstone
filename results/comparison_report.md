# Model comparison (generated automatically - do not edit by hand)

## test (originals only, PRIMARY)
| model        |   images |   source photos | accuracy [95% CI]    |   macro precision |   macro recall | macro F1 [95% CI]    |
|:-------------|---------:|----------------:|:---------------------|------------------:|---------------:|:---------------------|
| baseline_cnn |      215 |             215 | 0.753 [0.693, 0.805] |             0.758 |          0.752 | 0.746 [0.682, 0.795] |
| mobilenet_v2 |      215 |             215 | 0.935 [0.898, 0.967] |             0.941 |          0.92  | 0.928 [0.886, 0.965] |

Per-class F1:
|               |   baseline_cnn |   mobilenet_v2 |
|:--------------|---------------:|---------------:|
| freshapples   |          0.753 |          0.925 |
| freshbanana   |          0.857 |          0.958 |
| freshoranges  |          0.704 |          0.931 |
| rottenapples  |          0.76  |          0.943 |
| rottenbanana  |          0.848 |          0.962 |
| rottenoranges |          0.552 |          0.851 |

## test (all images)
| model        |   images |   source photos | accuracy [95% CI]    |   macro precision |   macro recall | macro F1 [95% CI]    |
|:-------------|---------:|----------------:|:---------------------|------------------:|---------------:|:---------------------|
| baseline_cnn |     1935 |             215 | 0.774 [0.721, 0.823] |             0.774 |          0.77  | 0.765 [0.704, 0.816] |
| mobilenet_v2 |     1935 |             215 | 0.918 [0.886, 0.946] |             0.924 |          0.898 | 0.908 [0.868, 0.938] |

Per-class F1:
|               |   baseline_cnn |   mobilenet_v2 |
|:--------------|---------------:|---------------:|
| freshapples   |          0.77  |          0.923 |
| freshbanana   |          0.855 |          0.961 |
| freshoranges  |          0.733 |          0.881 |
| rottenapples  |          0.781 |          0.927 |
| rottenbanana  |          0.863 |          0.959 |
| rottenoranges |          0.59  |          0.795 |

## Proposal targets (accuracy >= 90 %, macro-F1 > 0.85)
| model        | view                           |   accuracy (target >= 90%) |   macro-F1 (target > 0.85) | result   |
|:-------------|:-------------------------------|---------------------------:|---------------------------:|:---------|
| baseline_cnn | test (originals only, PRIMARY) |                      0.753 |                      0.746 | FAIL     |
| baseline_cnn | test (all images)              |                      0.774 |                      0.765 | FAIL     |
| mobilenet_v2 | test (originals only, PRIMARY) |                      0.935 |                      0.928 | PASS     |
| mobilenet_v2 | test (all images)              |                      0.918 |                      0.908 | PASS     |

## Fairness of the comparison
|                        | baseline_cnn                            | mobilenet_v2                            |
|:-----------------------|:----------------------------------------|:----------------------------------------|
| split (same CSV files) | see split_manifest.json                 | see split_manifest.json                 |
| seed                   | 42                                      | 42                                      |
| input size             | 224x224                                 | 224x224                                 |
| training augmentation  | True                                    | True                                    |
| class weights          | True                                    | True                                    |
| loss / optimizer       | sparse categorical cross-entropy / Adam | sparse categorical cross-entropy / Adam |
| train / val images     | 8280 / 1782                             | 8280 / 1782                             |
| pretrained (ImageNet)  | True                                    | True                                    |
| epochs run             | 10                                      | 23                                      |
| hardware               | 1 x GPU, global batch 2, off            | 1 x GPU, global batch 2, off            |

## Efficiency
| model        |   params (M) |   file size (MB) |   latency (ms/image, batch 1) |   training time (min) |
|:-------------|-------------:|-----------------:|------------------------------:|----------------------:|
| baseline_cnn |         1.18 |             14.2 |                          50   |                   7.7 |
| mobilenet_v2 |         2.27 |             23   |                         188.5 |                  17.3 |

## Is the difference significant? (exact McNemar test, test originals)
baseline_cnn only correct: 6; mobilenet_v2 only correct: 45; p-value = 0.0000. A p-value above 0.05 means the difference could be chance.

## Final model for the application
**mobilenet_v2** - highest validation macro-F1 (validation macro-F1: baseline_cnn 0.842, mobilenet_v2 0.996).
