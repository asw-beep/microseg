"""Visual design for the analysis app.

Design read: a scientific instrument UI for bench biologists and imaging
scientists.  Density is high, motion is near-zero, and the layout is aligned
rather than expressive.  It is a measurement tool, not a landing page.

Three decisions carry the design:

**The interface is monochrome; colour means something.**  The only saturated
colour on screen belongs to the data itself (false-colour instance maps, red
contour traces) and to the QC verdict.  A tinted interface would compete with
the output it frames, and in an image-analysis tool the image must win.  The
three status colours are taken from real fluorophore emission wavelengths, which
happen to run in the right order for a severity scale: FITC 519 nm green
(passed), TRITC 576 nm amber (needs review), Cy5 670 nm red (nothing found).
:class:`~microseg.channels.ChannelSpec` already carries ``wavelength_nm``.

**The dark ground is functional.**  Fiji, napari and CellProfiler are all dark
for the same reason: a bright interface wrapped around a dark fluorescence field
destroys your ability to judge faint signal by eye.

**Numbers get a face designed for numbers.**  Geist sets the interface, IBM Plex
Mono sets every measurement with tabular figures so digits align in a column and
values can be compared down a table without re-reading them.

Corner radius is 0 everywhere and there is exactly one uppercase tracked label on
the page, both on purpose: instruments are square, and a label above every panel
is decoration pretending to be structure.
"""

from __future__ import annotations

# Neutral scale.  Cool-tinted rather than pure grey so it sits under the blue
# cast that microscopy software conventionally uses, and never pure black.
VOID = "#080A0D"  # page ground
PANEL = "#101318"  # raised surface
PANEL_2 = "#161A21"  # inputs
RULE = "#232830"  # hairlines
TEXT = "#E6E9ED"
MUTED = "#828C99"

# Status only.  Emission wavelengths of the three standard filter sets.
FITC = "#3DDC84"  # 519 nm -- passed
TRITC = "#FFAE3B"  # 576 nm -- needs review
CY5 = "#FF5470"  # 670 nm -- nothing found

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --void: {VOID};
  --panel: {PANEL};
  --panel-2: {PANEL_2};
  --rule: {RULE};
  --text: {TEXT};
  --muted: {MUTED};
  --pass: {FITC};
  --review: {TRITC};
  --fail: {CY5};
  --sans: 'Geist', ui-sans-serif, system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', monospace;
}}

.gradio-container {{
  background: var(--void) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  max-width: 1560px !important;
}}

/* ---------------------------------------------------------------- masthead */
.ms-masthead {{
  display: flex;
  align-items: baseline;
  gap: 20px;
  flex-wrap: wrap;
  padding: 20px 2px 16px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 18px;
}}
.ms-masthead h1 {{
  font-family: var(--sans);
  font-weight: 600;
  font-size: 19px;
  letter-spacing: -0.015em;
  color: var(--text);
  margin: 0;
}}
.ms-masthead p {{
  color: var(--muted);
  font-size: 13.5px;
  line-height: 1.5;
  margin: 0;
  max-width: 60ch;
}}
.ms-source {{
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
}}

/* --------------------------------------------- QC verdict (the one signature) */
.ms-qc {{
  border: 1px solid var(--rule);
  border-left-width: 3px;
  border-left-color: var(--rule);
  background: var(--panel);
  padding: 15px 17px;
  margin-bottom: 14px;
}}
.ms-qc[data-state="pass"] {{ border-left-color: var(--pass); }}
.ms-qc[data-state="review"] {{ border-left-color: var(--review); }}
.ms-qc[data-state="fail"] {{ border-left-color: var(--fail); }}

.ms-qc-head {{
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}}
.ms-qc-verdict {{
  font-family: var(--sans);
  font-weight: 600;
  font-size: 14px;
  letter-spacing: -0.005em;
  color: var(--muted);
}}
.ms-qc[data-state="pass"] .ms-qc-verdict {{ color: var(--pass); }}
.ms-qc[data-state="review"] .ms-qc-verdict {{ color: var(--review); }}
.ms-qc[data-state="fail"] .ms-qc-verdict {{ color: var(--fail); }}

/* Segmented gauge: reads as an instrument level meter rather than a progress
   bar, which is right because this is a measured quantity, not a task. */
.ms-gauge {{ display: flex; gap: 2px; align-items: center; }}
.ms-tick {{ width: 4px; height: 14px; background: #1D222A; }}
.ms-qc[data-state="pass"] .ms-tick.on {{ background: var(--pass); }}
.ms-qc[data-state="review"] .ms-tick.on {{ background: var(--review); }}
.ms-qc[data-state="fail"] .ms-tick.on {{ background: var(--fail); }}

.ms-qc-score {{
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-size: 12.5px;
  color: var(--text);
}}
.ms-qc-score i {{ font-style: normal; color: var(--muted); margin-right: 7px; }}
.ms-qc-note {{
  margin: 10px 0 0;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.55;
  max-width: 78ch;
}}
.ms-qc-flags {{
  list-style: none;
  margin: 12px 0 0;
  padding: 11px 0 0;
  border-top: 1px solid var(--rule);
}}
.ms-qc-flags li {{
  font-family: var(--mono);
  font-size: 11.5px;
  line-height: 1.8;
  color: #C6CEDA;
  padding-left: 17px;
  position: relative;
}}
.ms-qc-flags li::before {{
  content: "\\203A";
  position: absolute;
  left: 2px;
  color: var(--review);
}}

/* ----------------------------------------------------------------- readout */
/* Plain layout with hairline dividers. At this density a grid of bordered
   cards would add six containers and no information. */
.ms-readout {{
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  border-top: 1px solid var(--rule);
  border-bottom: 1px solid var(--rule);
  margin: 0 0 16px;
  padding: 0;
}}
.ms-readout > div {{
  flex: 1 1 132px;
  padding: 13px 18px 13px 0;
  margin-right: 18px;
  border-right: 1px solid var(--rule);
}}
.ms-readout > div:last-child {{ border-right: 0; margin-right: 0; }}
.ms-readout dt {{
  font-family: var(--sans);
  font-weight: 500;
  font-size: 11.5px;
  color: var(--muted);
  margin: 0 0 6px;
}}
.ms-readout dd {{
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-weight: 500;
  font-size: 24px;
  line-height: 1;
  color: var(--text);
  margin: 0;
}}
.ms-readout dd small {{
  font-size: 11.5px;
  color: var(--muted);
  margin-left: 4px;
  font-weight: 400;
}}

/* ------------------------------------------------------------- empty state */
.ms-empty {{
  border: 1px dashed var(--rule);
  padding: 52px 32px;
  text-align: center;
}}
.ms-empty h2 {{
  font-family: var(--sans);
  font-weight: 600;
  font-size: 16px;
  letter-spacing: -0.01em;
  color: var(--text);
  margin: 0 0 10px;
}}
.ms-empty p {{
  color: var(--muted);
  font-size: 13.5px;
  line-height: 1.65;
  margin: 0 auto;
  max-width: 52ch;
}}

/* -------------------------------------------------------- gradio overrides */
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap {{
  background: var(--panel) !important;
  border-color: var(--rule) !important;
  border-radius: 0 !important;
}}
.gradio-container label span {{
  color: var(--muted) !important;
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 12px !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
}}
.gradio-container input,
.gradio-container select,
.gradio-container textarea {{
  background: var(--panel-2) !important;
  color: var(--text) !important;
  border-color: var(--rule) !important;
  border-radius: 0 !important;
  font-family: var(--sans) !important;
}}
/* Gradio sets `appearance: none` on radios and checkboxes and draws the
   selected state entirely with background-color + background-image.  The
   `background` *shorthand* therefore erases the only thing that marks a choice
   as chosen, and with no native control underneath the control renders
   identically whether or not it is selected -- the segmentation method looked
   unpickable.  So paint the states explicitly instead of blanking them. */
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {{
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 14px !important;
  height: 14px !important;
  background-color: var(--panel-2) !important;
  background-image: none !important;
  border: 1px solid var(--muted) !important;
  cursor: pointer !important;
}}
.gradio-container input[type="radio"] {{
  border-radius: 50% !important;
}}
.gradio-container input[type="radio"]:checked {{
  background-color: var(--text) !important;
  border-color: var(--text) !important;
  /* Inset ring in the input ground turns the filled disc into a radio dot. */
  box-shadow: inset 0 0 0 3px var(--panel-2) !important;
}}
.gradio-container input[type="checkbox"]:checked {{
  background-color: var(--text) !important;
  border-color: var(--text) !important;
  box-shadow: inset 0 0 0 2px var(--panel-2) !important;
}}
.gradio-container input[type="radio"]:hover,
.gradio-container input[type="checkbox"]:hover {{
  border-color: var(--text) !important;
}}
/* The whole option, label included, should read as clickable. */
.gradio-container label:has(input[type="radio"]),
.gradio-container label:has(input[type="checkbox"]) {{
  cursor: pointer !important;
}}
.gradio-container label:has(input[type="radio"]:checked) span,
.gradio-container label:has(input[type="checkbox"]:checked) span {{
  color: var(--text) !important;
}}
.gradio-container input:focus-visible,
.gradio-container select:focus-visible,
.gradio-container button:focus-visible {{
  outline: 2px solid var(--text) !important;
  outline-offset: 1px !important;
}}
.gradio-container .tab-nav button {{
  font-family: var(--sans) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  color: var(--muted) !important;
  border-radius: 0 !important;
}}
.gradio-container .tab-nav button.selected {{
  color: var(--text) !important;
  border-bottom-color: var(--text) !important;
}}
#ms-run {{
  background: var(--text) !important;
  border: 1px solid var(--text) !important;
  color: var(--void) !important;
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  border-radius: 0 !important;
  white-space: nowrap !important;
}}
#ms-run:hover {{ background: #FFFFFF !important; }}
#ms-run:active {{ transform: translateY(1px); }}
.gradio-container table {{
  font-family: var(--mono) !important;
  font-variant-numeric: tabular-nums !important;
  font-size: 11.5px !important;
}}

@media (max-width: 760px) {{
  .ms-readout > div {{ flex: 1 1 100%; border-right: 0; margin-right: 0; }}
  .ms-source {{ margin-left: 0; }}
}}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
"""


def qc_rail(report, n_objects: int) -> str:
    """Render the QC verdict.

    The most prominent element on the page, deliberately: a nucleus count is
    only useful next to a statement of whether to believe it.
    """
    if report is None:
        return (
            '<div class="ms-qc">'
            '<div class="ms-qc-head"><span class="ms-qc-verdict">No result yet</span></div>'
            '<p class="ms-qc-note">Quality checks run automatically after each analysis.</p>'
            "</div>"
        )

    confidence = float(report.confidence)
    if report.passed:
        state = "pass"
        verdict = "Passed checks"
        note = f"{n_objects} nuclei measured. Every plausibility check cleared."
    elif n_objects == 0:
        state = "fail"
        verdict = "No nuclei found"
        note = (
            "Nothing was detected. Check the focus and exposure, and confirm this "
            "is the nuclear channel."
        )
    else:
        state = "review"
        verdict = "Needs review"
        note = (
            f"Found {n_objects} objects, but the checks below did not clear. "
            "Treat these measurements as provisional."
        )

    filled = max(0, min(20, round(confidence * 20)))
    ticks = "".join(
        '<span class="ms-tick on"></span>' if i < filled else '<span class="ms-tick"></span>'
        for i in range(20)
    )
    flags = "".join(f"<li>{_escape(f)}</li>" for f in report.flags)

    return (
        f'<div class="ms-qc" data-state="{state}">'
        '<div class="ms-qc-head">'
        f'<span class="ms-qc-verdict">{verdict}</span>'
        f'<span class="ms-gauge" role="img" aria-label="confidence {confidence:.2f} of 1">{ticks}</span>'
        f'<span class="ms-qc-score"><i>confidence</i>{confidence:.2f}</span>'
        "</div>"
        f'<p class="ms-qc-note">{note}</p>'
        + (f'<ul class="ms-qc-flags">{flags}</ul>' if flags else "")
        + "</div>"
    )


def readout(cells: list[tuple[str, str, str]]) -> str:
    """Render the measurement counters. ``cells`` is (label, value, unit)."""
    parts: list[str] = []
    for label, value, unit in cells:
        suffix = f"<small>{_escape(unit)}</small>" if unit else ""
        parts.append(
            f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}{suffix}</dd></div>"
        )
    return '<dl class="ms-readout">' + "".join(parts) + "</dl>"


def empty_state() -> str:
    return (
        '<div class="ms-empty">'
        "<h2>Drop in a field of view</h2>"
        "<p>The pipeline flattens illumination, separates each nucleus, measures its "
        "shape, intensity and chromatin texture, then checks whether the result is "
        "trustworthy before handing it back.</p>"
        "</div>"
    )


def _escape(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
