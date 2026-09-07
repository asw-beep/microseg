"""Gradio front end for the segmentation and quantification pipeline.

Built around the three questions someone actually asks of a segmentation tool,
in the order they ask them:

1. *Can I trust this result?* The QC verdict sits above everything else. A count
   without a statement of confidence is not a measurement.
2. *What did it find?* Outlines over the input, not filled masks, because a
   filled overlay hides exactly the pixels a reviewer needs to check.
3. *What are the numbers?* The per-nucleus table, downloadable as CSV, since the
   measurements are the actual deliverable.

The preprocessing panel is one tab back rather than hidden: most bad
segmentations are bad preprocessing, and that is invisible in the final overlay.

The U-Net backend appears only when a checkpoint is present, so the app is
useful straight after cloning with the classical backend alone.

    python app/gradio_app.py --checkpoint outputs/unet_cpu/best.pt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

# Make `src/` importable when running this file directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from theme import CSS, empty_state, qc_rail, readout  # noqa: E402

from microseg.channels import ChannelSpec, MicroscopyImage  # noqa: E402
from microseg.config import ExperimentConfig, load_config  # noqa: E402
from microseg.pipeline import SegmentationPipeline  # noqa: E402
from microseg.viz import label_to_rgb, overlay_boundaries  # noqa: E402

CLASSICAL = "Classical (threshold + watershed)"
UNET = "U-Net"

# Columns worth showing on screen; the CSV carries all ~70.
DISPLAY_COLUMNS = [
    "label",
    "area_px",
    "perimeter_crofton_px",
    "circularity",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "integrated_intensity",
    "glcm_contrast",
    "glcm_homogeneity",
]

STATE: dict = {"cfg": ExperimentConfig(), "checkpoint": None}


def _to_image(array: np.ndarray) -> MicroscopyImage:
    """Whatever the upload widget produced -> a single-channel nuclei image."""
    arr = np.asarray(array)

    # Scale to 0..1 *before* collapsing channels, not after.  Collapsing an
    # 8-bit upload with .mean() yields a float array, and MicroscopyImage
    # .from_array only divides by the dtype range for *integer* input -- so the
    # scaling was silently skipped and the field reached the pipeline in 0..255.
    # Every stage downstream assumes 0..1; CLAHE is the one that visibly breaks,
    # and a dim fluorescence field came back with no nuclei at all.
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            # Collapse RGB the way the dataset loader does, so the app and the
            # batch pipeline behave identically on the same file.  The variance
            # test is scale-free, so scaling first does not change which branch
            # is taken.
            variances = arr.reshape(-1, arr.shape[2]).var(axis=0)
            if variances.max() - variances.min() > 0.15 * max(float(variances.max()), 1e-6):
                arr = arr[:, :, int(np.argmax(variances))]
            elif arr.shape[2] == 3:
                # Luminance, matching io_utils._rgb_to_nuclei -- a flat mean is a
                # different image on genuine colour (H&E) input.
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                arr = arr.mean(axis=2)
    return MicroscopyImage.from_array(arr, [ChannelSpec("nuclei", "nuclei")], name="upload")


def _build_pipeline(backend: str, overrides: dict) -> SegmentationPipeline:
    cfg = STATE["cfg"]
    for key, value in overrides.items():
        setattr(cfg.preprocess, key, value)

    if backend == UNET:
        checkpoint = STATE.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            raise gr.Error(
                "No U-Net checkpoint is loaded. Train one with "
                "`python -m microseg.train --config configs/unet_cpu.yaml`, "
                "or switch to the classical backend."
            )
        pipeline = SegmentationPipeline.from_checkpoint(checkpoint, device="cpu")
        # Honour the panel's preprocessing choices over the checkpoint's.
        pipeline.cfg.preprocess = cfg.preprocess
        pipeline.cfg.postprocess = cfg.postprocess
        pipeline.cfg.analysis = cfg.analysis
        return pipeline
    return SegmentationPipeline(cfg, backend="classical")


def analyze(
    image,
    backend: str,
    illumination: str,
    denoise: str,
    contrast: str,
    min_area: int,
    pixel_size_um: float,
    compute_texture: bool,
):
    """Run the pipeline and package everything for display."""
    if image is None:
        raise gr.Error("Upload a microscopy image first.")

    cfg = STATE["cfg"]
    cfg.postprocess.min_object_area = int(min_area)
    cfg.analysis.compute_texture = bool(compute_texture)

    pipeline = _build_pipeline(
        backend,
        {
            "illumination": None if illumination == "off" else illumination,
            "denoise": None if denoise == "off" else denoise,
            "contrast": None if contrast == "off" else contrast,
        },
    )

    micro = _to_image(image)
    if pixel_size_um and pixel_size_um > 0:
        micro.pixel_size_um = float(pixel_size_um)

    result = pipeline.run(micro, collect_stages=True)

    overlay = overlay_boundaries(result.preprocessed.nuclei(), result.instances, width=2)
    instances_rgb = label_to_rgb(result.instances)
    stages = [(np.clip(v, 0, 1), k.replace("_", " ")) for k, v in result.stages.items()]

    df = result.features
    if df is not None and not df.empty:
        table = df[[c for c in DISPLAY_COLUMNS if c in df.columns]].round(3)
        # A unique directory per run. A fixed path in the shared temp dir would
        # let concurrent users overwrite each other's results.
        csv_path = Path(tempfile.mkdtemp(prefix="microseg-")) / "nuclei_features.csv"
        df.to_csv(csv_path, index=False)
        csv_out = str(csv_path)
    else:
        table = pd.DataFrame({"note": ["No objects were detected."]})
        csv_out = None

    return (
        qc_rail(result.qc, result.n_objects),
        _readout_for(result),
        overlay,
        instances_rgb,
        stages,
        table,
        csv_out,
        gr.update(visible=False),
    )


def _readout_for(result) -> str:
    """The measurement counters: what was found, and how long it took."""
    df = result.features
    cells = [("Nuclei", str(result.n_objects), "")]

    if df is not None and not df.empty:
        if "area_um2" in df:
            cells.append(("Median area", f"{df['area_um2'].median():.0f}", "um2"))
        else:
            cells.append(("Median area", f"{df['area_px'].median():.0f}", "px"))
        cells.append(("Median circularity", f"{df['circularity'].median():.2f}", ""))
        cells.append(("Median solidity", f"{df['solidity'].median():.2f}", ""))
        if "touches_border" in df:
            cells.append(("Clipped by edge", str(int(df["touches_border"].sum())), ""))

    cells.append(("Runtime", f"{result.timings['total_s']:.2f}", "s"))
    return readout(cells)


def build_interface() -> gr.Blocks:
    backends = [CLASSICAL]
    if STATE.get("checkpoint") and Path(STATE["checkpoint"]).exists():
        backends.insert(0, UNET)

    with gr.Blocks(title="Nuclei analysis") as demo:
        gr.HTML(
            '<header class="ms-masthead">'
            "<h1>Nuclei analysis</h1>"
            "<p>Segment individual nuclei in a microscopy field and measure the shape, "
            "intensity and chromatin texture of each one.</p>"
            '<span class="ms-source">Validated on BBBC038</span>'
            "</header>"
        )

        with gr.Row():
            # ------------------------------------------------------- controls
            with gr.Column(scale=3, min_width=280):
                image_input = gr.Image(label="Field of view", type="numpy", height=260)
                backend_input = gr.Radio(
                    backends, value=backends[0], label="Segmentation method"
                )
                run_button = gr.Button("Analyze", variant="primary", elem_id="ms-run")

                with gr.Accordion("Preprocessing", open=False):
                    illumination_input = gr.Dropdown(
                        ["morphological", "polynomial", "gaussian", "off"],
                        value="morphological",
                        label="Illumination correction",
                        info="Flattens uneven background so one threshold works across the field.",
                    )
                    denoise_input = gr.Dropdown(
                        ["median", "gaussian", "bilateral", "nlm", "tv", "off"],
                        value="median",
                        label="Denoising",
                        info="Median keeps edges sharp. Non-local means preserves texture but is slow.",
                    )
                    contrast_input = gr.Dropdown(
                        ["clahe", "gamma", "stretch", "off"],
                        value="clahe",
                        label="Contrast enhancement",
                        info="Lifts dim nuclei to match brighter ones before segmentation.",
                    )

                with gr.Accordion("Objects and measurements", open=False):
                    min_area_input = gr.Slider(
                        1, 200, value=15, step=1,
                        label="Smallest object to keep (px)",
                        info="Anything smaller counts as debris.",
                    )
                    pixel_size_input = gr.Number(
                        value=0,
                        label="Pixel size (um)",
                        info="Set this to report areas in um2 as well as pixels.",
                    )
                    texture_input = gr.Checkbox(
                        value=True,
                        label="Measure chromatin texture (GLCM)",
                        info="The slowest stage. Turn it off for counts and shape only.",
                    )

            # -------------------------------------------------------- results
            with gr.Column(scale=7):
                qc_output = gr.HTML(qc_rail(None, 0))
                readout_output = gr.HTML()

                with gr.Tab("Nuclei"):
                    overlay_output = gr.Image(label="Detected outlines", height=440)
                with gr.Tab("Instances"):
                    instances_output = gr.Image(label="One colour per nucleus", height=440)
                with gr.Tab("Preprocessing"):
                    stages_output = gr.Gallery(
                        label="Each stage of the chain", columns=3, height=440
                    )
                with gr.Tab("Measurements"):
                    table_output = gr.Dataframe(label="One row per nucleus", wrap=True)
                    csv_output = gr.File(label="Download all measurements (CSV)")

                empty_output = gr.HTML(empty_state())

        run_button.click(
            analyze,
            inputs=[
                image_input,
                backend_input,
                illumination_input,
                denoise_input,
                contrast_input,
                min_area_input,
                pixel_size_input,
                texture_input,
            ],
            outputs=[
                qc_output,
                readout_output,
                overlay_output,
                instances_output,
                stages_output,
                table_output,
                csv_output,
                empty_output,
            ],
        )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the nuclei analysis app")
    parser.add_argument("--config", default="configs/unet_cpu.yaml")
    parser.add_argument("--checkpoint", default="outputs/unet_cpu/best.pt")
    # Localhost by default. Containers pass --host 0.0.0.0 explicitly, so
    # exposing the app on every interface is always a deliberate act.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    if Path(args.config).exists():
        STATE["cfg"] = load_config(args.config)
    if Path(args.checkpoint).exists():
        STATE["checkpoint"] = args.checkpoint
        print(f"U-Net backend enabled from {args.checkpoint}")
    else:
        print(f"No checkpoint at {args.checkpoint}; classical backend only.")

    build_interface().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=CSS,
        theme=gr.themes.Base(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
