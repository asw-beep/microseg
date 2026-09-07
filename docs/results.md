# Detailed results

All numbers are on the BBBC038 held-out **test** split (100 images), using
`split_seed: 42` which every experiment config shares. No image in this split
was seen during training or model selection.

Regenerate everything with:

```bash
python -m microseg.evaluate --config configs/classical_baseline.yaml --backend classical --split test --robustness
python -m microseg.evaluate --config configs/unet_cpu.yaml --backend unet --split test --robustness
python scripts/run_benchmarks.py --config configs/unet_cpu.yaml
python scripts/make_figures.py --config configs/unet_cpu.yaml
```

## Main comparison

Full test split, 100 images:

| Method | Dice | IoU | Precision | Recall | F1 | AP | Matched IoU | Count MAE | s/image |
|---|---|---|---|---|---|---|---|---|---|
| Classical | 0.8646 | 0.7850 | 0.6986 | 0.7387 | 0.7073 | 0.3687 | 0.7753 | 9.13 | 0.177 |
| U-Net | 0.8740 | 0.7990 | 0.8097 | 0.7527 | 0.7753 | 0.4782 | 0.8167 | 5.19 | 0.276 |

External methods, first 50 test images (SAM on 5):

| Method | Dice | IoU | Precision | Recall | F1 | AP | Matched IoU | Count MAE | s/image |
|---|---|---|---|---|---|---|---|---|---|
| Classical | 0.8580 | 0.7799 | 0.6758 | 0.7202 | 0.6877 | 0.3331 | 0.7586 | 7.82 | 0.126 |
| U-Net | 0.8653 | 0.7916 | 0.8039 | 0.7483 | 0.7719 | 0.4591 | 0.7937 | 4.76 | 0.199 |
| Cellpose 3 `nuclei` | 0.8701 | 0.7968 | 0.8706 | 0.8463 | 0.8561 | 0.5300 | 0.7946 | 3.00 | 3.290 |
| SAM ViT-B automatic | 0.6563 | 0.5400 | 0.4440 | 0.6935 | 0.5328 | 0.2602 | 0.8101 | 13.60 | 56.056 |
| SAM ViT-B prompted | 0.8337 | 0.7314 | 0.5650 | 0.6810 | 0.6133 | 0.3338 | 0.8102 | 6.40 | 7.311 |

The headline observation: Dice spans 0.858-0.870 across classical, U-Net and
Cellpose, while AP spans 0.333-0.530. Pixel agreement is nearly identical; object
decomposition is not remotely identical. Reporting Dice alone would make these
three methods look interchangeable.

Note also that SAM has the **highest matched IoU of any method** (0.810) and the
**lowest AP**. When it finds a nucleus it outlines it better than anything else
here; it simply fails to find most of them. That is the clearest single number
supporting the "classical CV for detection, foundation model for boundaries"
hybrid.

## Training run

`configs/unet_cpu.yaml`, CPU only:

| | |
|---|---|
| Parameters | 1,942,323 |
| Epochs run | 24 of 25 (early-stopped, patience 8) |
| Best epoch | 16, by validation instance AP |
| Wall clock | 31.8 minutes |
| Best val Dice / boundary Dice / AP | 0.8663 / 0.6243 / 0.4445 |

Over the last 10 epochs, validation Dice spanned 2.1% of its mean while instance
AP spanned 13.3% -- AP carries several times more signal about which checkpoint to
keep, which is why it is the selection metric. Both peaked at epoch 16 in this
run, so the choice did not change the outcome here.


## Dataset characteristics

Worth stating up front, because it explains most of the difficulty:

| | |
|---|---|
| Images | 670 annotated fields (stage1_train) |
| Split | 470 train / 100 val / 100 test |
| Distinct image sizes | 8, from 256x256 to 1024x1024 |
| Nuclei per field | 1 to 369 (median 27) |
| Modalities | fluorescence, brightfield, H&E |

The modality mix is the reason `preprocess.auto_invert` exists: fluorescence has
bright nuclei on a dark background and H&E the reverse, and without polarity
normalisation no single threshold or set of weights covers both.

## Robustness

Same 25 test images under each corruption. Dice (AP in parentheses):

| Corruption | Classical | U-Net |
|---|---|---|
| none | 0.8152 (0.2945) | 0.8230 (0.4180) |
| exposure x0.5 | 0.7289 (0.2846) | 0.8137 (0.4074) |
| exposure x1.6 | 0.7920 (0.3066) | 0.8193 (0.4251) |
| uneven illumination | 0.7832 (0.2775) | 0.8132 (0.4064) |
| JPEG artifacts | 0.8098 (0.2691) | 0.8051 (0.3737) |
| defocus blur | 0.7899 (0.2464) | 0.7947 (0.3257) |
| Gaussian noise, sigma=0.05 | 0.6606 (0.1734) | 0.7039 (0.2064) |
| heavy noise, sigma=0.12 | 0.5617 (0.0665) | 0.4272 (0.0560) |

Two readings:

**The preprocessing chain works.** Exposure and illumination corruptions cost the
U-Net at most 0.010 Dice, and those are precisely the perturbations the
percentile normalisation and illumination correction are designed to remove. The
classical backend loses 3-9x more on the same corruptions, because a fixed
threshold has less slack.

**The learned model fails harder out of distribution.** Under heavy noise the
U-Net drops to 0.427 Dice while the classical baseline holds 0.562. Thresholding
is a much simpler function and degrades more gracefully when pushed far from the
training distribution. If the deployment target is noisy, that argues for
stronger noise augmentation during training rather than for the baseline.


## Timing

Measured per image on CPU, averaged over the test split.

| Stage | Classical | U-Net |
|---|---|---|
| preprocess | 0.007 s | 0.008 s |
| segment | 0.026 s | 0.119 s |
| analyse | 0.144 s | 0.149 s |
| **total** | **0.177 s** | **0.276 s** |


The result worth acting on: **feature extraction, not segmentation, is the
bottleneck** for the classical backend. GLCM texture over every object dominates,
which is why `analysis.compute_texture` is a config flag — turning it off is the
single biggest speedup available when only counts and morphology are needed.

This was also true of preprocessing before it was profiled: morphological
illumination correction with a 101x101 structuring element cost ~0.26 s/image and
dominated the entire chain. Estimating the background on a downsampled copy (the
background is low-frequency by construction) made it ~39x faster with no
measurable change in segmentation accuracy — Dice moved from 0.8645 to 0.8646.
That equivalence is pinned by a test.

## Notes on the comparison

- Every backend is given the same preprocessed images, the same split and the
  same metrics, so the numbers isolate the segmentation step.
- `ap_mean` is the DSB2018 score: the mean over IoU thresholds 0.50-0.95 of
  `TP / (TP + FP + FN)`. It is not the area under a precision-recall curve; the
  competition named it "average precision" and the name stuck.
- Instance matching is solved with the Hungarian algorithm. Greedy matching
  inflates scores when one large prediction overlaps two ground-truth objects.
- Per-image metrics are averaged with equal weight per field rather than pooled
  over pixels, so a 1024x1024 field does not count 16x a 256x256 one.
