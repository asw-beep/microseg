# Cellpose and SAM: where general models help, and where they don't

Two external comparisons ship with this project. Both are optional — the core
pipeline, the tests and the Docker image run without them.

```bash
pip install -r requirements-benchmarks.txt

# SAM weights are not bundled (375 MB)
mkdir -p weights
curl -L -o weights/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

python scripts/run_benchmarks.py --config configs/unet_cpu.yaml
```

Every method is given the same preprocessed images, the same held-out split and
the same metrics, so the comparison isolates the segmentation step.

## Measured results

First 50 BBBC038 test images (SAM on 5, because it costs ~1 minute per field on
CPU):

| Method | Dice | F1 | AP | Matched IoU | s/image |
|---|---|---|---|---|---|
| Classical | 0.858 | 0.688 | 0.333 | 0.759 | 0.13 |
| U-Net (this project) | 0.865 | 0.772 | 0.459 | 0.794 | 0.20 |
| Cellpose 3 `nuclei` | **0.870** | **0.856** | **0.530** | 0.795 | 3.29 |
| SAM ViT-B, automatic | 0.656 | 0.533 | 0.260 | **0.810** | 56.06 |
| SAM ViT-B, prompted from classical seeds | 0.834 | 0.613 | 0.334 | **0.810** | 7.31 |

## Cellpose

Cellpose is the right reference point for a custom U-Net, and the comparison is
about a trade-off rather than a winner.

**How it differs mechanically.** Cellpose predicts spatial *gradient flows*
pointing toward each cell's centre, then recovers instances by following those
flows to a common attractor. That is a fundamentally different instance
mechanism from the boundary-class watershed used here. It handles crowded and
strongly non-convex cells more gracefully, because it never has to represent a
separating boundary explicitly — it only has to get the flow direction right.

**Where each wins.**

| | Custom U-Net | Cellpose |
|---|---|---|
| Parameters | ~1.9M (CPU config) / 7.8M | much larger |
| Training data needed | a few hundred annotated fields | none (pretrained) |
| Adapting to a new assay | retrain in minutes | fine-tune, or accept the generalist |
| Dense, non-convex cells | boundary class can fail | flow field degrades gracefully |
| Deployment | single file, CPU real-time | heavier dependency stack |

**What the measurement showed.** Cellpose wins on accuracy — AP 0.530 against
0.459 — which is the expected and honest result: it is pretrained on far more
data than BBBC038's 470 training images. It wins by most on **recall** (0.846 vs
0.748), i.e. it misses fewer nuclei, which is where a generalist's broader
training shows. The custom U-Net's compensation is speed: **16x faster on CPU**
(0.20 s vs 3.29 s per field) at ~1.9M parameters, retrainable on a new assay in
half an hour.

The practical reading: Cellpose is what you reach for when you have no
annotations and need a result today. A custom model is what you reach for when
you have a specific assay, some annotations, and a throughput or deployment
constraint — which is the situation most production imaging pipelines are in.

**Version note, and a deployment-relevant finding.** Cellpose 4 replaced the
per-tissue CNNs with a single `cpsam` model — a large transformer with a 1.2 GB
checkpoint. On CPU it **failed to segment 3 images in 30 minutes**, so it is not
usable for CPU inference at all. The benchmark here therefore uses Cellpose 3's
classic `nuclei` model (25 MB, 3.3 s/image on CPU), which is also the model most
published comparisons cite.

`benchmarks/cellpose_compare.py` tries both call signatures and reports which one
it used, so the code works across 2.x-4.x; `requirements-benchmarks.txt` pins
`cellpose<4` so the default install is the CPU-viable one.

## Segment Anything (SAM)

A deliberately small experiment, run to characterise a limitation rather than to
propose SAM as a solution.

**The structural problem.** SAM is class-agnostic: it segments whatever is
there, with no notion of "nucleus". On a field of a few hundred small,
near-identical, touching nuclei this produces three characteristic failures:

1. The automatic mask generator's point grid (32x32 by default) is coarser than
   the nuclei themselves, so many nuclei are never prompted and are simply
   missed.
2. Its ambiguity resolution favours larger, more "object-like" regions, so it
   readily returns a clump of nuclei — or the entire tissue region — as one mask.
3. Returned masks may overlap and are unordered, whereas an instance
   segmentation must be a *partition*. A resolution rule is needed before the
   standard metrics even apply.

Point 3 is easy to get wrong quietly. `masks_to_labels` paints masks
smallest-last, so when SAM returns both a clump and its constituent nuclei, the
more specific masks win the contested pixels. Without a rule like this, any
IoU-matched metric computed on raw SAM output is meaningless.

**Two modes, to separate two questions.** "Can SAM see nuclei at all?" is a
different question from "was SAM told where to look?", so both are measured:

- `sam_automatic_masks` — zero-shot automatic generation. The honest baseline,
  and the one that exhibits failures 1 and 2.
- `sam_prompted_masks` — SAM prompted with one point per nucleus, taken from
  this project's own classical distance-transform seeding stage.

The prompted mode beats automatic by a wide margin, as measured above: Dice
0.656 → 0.834, AP 0.260 → 0.334, and 7.7x faster (5-40 prompts instead of a
1024-point grid). It is a genuinely useful pattern: cheap classical CV is very
good at *finding* nucleus centres and mediocre at outlines; SAM is the reverse.
Composing them plays to both, and needs no training data at all.

**The number that makes the argument.** SAM has the **highest matched IoU of any
method benchmarked** (0.810, above Cellpose's 0.795 and the U-Net's 0.794) and
simultaneously the **lowest AP** (0.260 automatic). Matched IoU only scores
objects it actually found — so when SAM finds a nucleus it traces the outline
better than any purpose-built model here, and it simply fails to find most of
them. Detection and delineation are separable problems, and SAM is strong at
exactly one of them.

**Why this is not the shipped default.** Even prompted, SAM costs seconds per
field on CPU against tens of milliseconds for the U-Net, requires a 375 MB
checkpoint, and its outlines are tuned for natural-image object boundaries
rather than the low-contrast, blurred edges of a fluorescence nucleus. For a
plate scan of thousands of fields, that is the wrong trade.

**SAM 2** adds video/tracking capability, which is genuinely relevant for
live-cell time series — tracking a dividing nucleus across frames is a real
assay need that neither this U-Net nor Cellpose addresses. That is the direction
worth pursuing, rather than SAM as a better static segmenter.
