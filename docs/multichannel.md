# Multi-channel microscopy: design notes

BBBC038 is effectively a single-channel nuclei dataset. Real assays are not.
A high-content screen typically acquires 3-6 channels per field (DAPI for
nuclei, phalloidin for actin, a membrane marker, plus one or two readout
stains), and increasingly 10-40 channels for imaging mass cytometry or cyclic
immunofluorescence.

This pipeline is built channel-aware from the start rather than retrofitted,
because the retrofit is invasive: it touches loading, preprocessing,
segmentation input and feature extraction simultaneously. What follows is what
is implemented, and what would need to change to go further.

## The core abstraction

`microseg.channels.MicroscopyImage` carries an `(H, W, C)` float32 array plus
one `ChannelSpec` per channel:

```python
image = MicroscopyImage(
    data,                                    # (H, W, 4)
    [ChannelSpec("dapi",      "nuclei"),
     ChannelSpec("phalloidin","cytoplasm"),
     ChannelSpec("ecad",      "membrane"),
     ChannelSpec("ki67",      "marker")],
    pixel_size_um=0.325,
)
```

Two ideas do the work:

**Names and roles are separate.** The name identifies the stain; the role says
what the pipeline may do with it. `nuclei` is the channel segmentation runs on,
`marker` is measured but never segmented. This means the pipeline never has to
guess which channel is which, and swapping a stain is a metadata change rather
than a code change.

**Channel selectors are pervasive.** Every preprocessing operator accepts
`channels=`, defaulting to all of them. `apply_per_channel` in
`preprocessing/chain.py` is the single place that knows how to broadcast a
2-D operator over a stack, so every individual operator stays simple and 2-D.

## Why channels are processed independently

Each channel has its own background level, noise characteristics, dynamic range
and bleed-through. Normalising them jointly — the obvious shortcut — is wrong in
a specific and damaging way: a bright, saturated marker channel dominates the
joint percentile, and the dim nuclear channel gets crushed toward zero.

So `preprocess()` runs the full chain per channel, with per-channel statistics.
This is tested directly (`test_channels_are_processed_independently`).

## Segment on one channel, measure on all

This is the workflow that motivates the whole abstraction, and it is what
`SegmentationPipeline.run` implements:

1. Segmentation reads `image.nuclei()` — the channel whose role is `nuclei`.
2. Feature extraction iterates over **every** channel, computing intensity and
   GLCM texture features against the same label image, prefixed by channel name
   (`ki67_mean_intensity`, `ki67_glcm_contrast`, ...).

The result is the table a screen actually wants: one row per nucleus, with
marker intensity per nucleus, ready to gate on. `measure_channels=` narrows this
when only a subset matters.

## Feeding multiple channels to the network

`ModelConfig.in_channels` is configurable and the U-Net is built accordingly, so
a DAPI + membrane stack trains as a 2-channel input with no code change:

```yaml
model:
  in_channels: 2
```

The dataset class in this project stacks only the nuclei channel, because
BBBC038 has nothing else to stack. Extending it means changing which channels
`NucleiDataset.__getitem__` selects — a few lines — and is left undone rather
than written speculatively against a dataset that cannot exercise it.

Worth noting: adding a cytoplasm channel is usually the single biggest accuracy
win available for *cell* (as opposed to nuclei) segmentation, because cell
boundaries are frequently invisible in the nuclear channel.

## What is deliberately not built

Beyond roughly 10 channels the assumptions here start to break, and the honest
answer is that these need different machinery, not a bigger loop:

- **Spectral unmixing.** Overlapping emission spectra mean observed channels are
  linear mixtures of true fluorophores. Correcting this needs a mixing matrix
  from single-stain controls, which is acquisition metadata this pipeline has no
  way to obtain.
- **Channel registration.** Sequential acquisition (cyclic IF, IMC) drifts
  between rounds and needs per-round affine alignment before any per-object
  measurement is meaningful.
- **Z-stacks and time series.** `io_utils.load_image` max-projects a 4-D TIFF
  over the leading axis and says so in `meta["layout"]`. That is the right
  default for counting nuclei and the wrong one for 3-D morphology, which needs
  a 3-D U-Net and anisotropic voxel handling throughout.
- **Very high channel counts.** At 40 channels, per-channel GLCM features
  produce thousands of columns per object. That is a feature-selection problem,
  not an imaging one.

Each of these is a real requirement in some assay. None is faked here.
