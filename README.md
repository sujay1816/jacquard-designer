# Jacquard Designer

A browser-based tool that converts saree/textile design images into **1-bit BMP files** for driving a jacquard weaving loom. Upload a design, set your loom parameters, assign colours to shuttles, choose weave patterns, and download ready-to-use BMP files — one per shuttle.

It also includes a full **BMP editor**, a **border studio**, and a **tracing guide** for cleaning up photographed designs.

<!-- SCREENSHOT: a wide hero shot of the Generator with a detected design works best here -->
<!-- ![Jacquard Designer](docs/hero.png) -->

---

## What it does

Upload a cropped design image (butta motif, running lines, full repeats, etc.), set your loom's pin and card count, assign the detected colours to shuttles, pick a weave per shuttle, and download a ZIP of loom-ready BMPs.

**BMP pixel convention**

- Black (0) = thread **UP** (visible on fabric)
- White (255) = thread **DOWN** (hidden)

**Shuttle types**

| Shuttle      | Purpose                                                                   |
| ------------ | ------------------------------------------------------------------------- |
| Zari         | Gold thread — satin or solid fill                                         |
| Meena 1      | First colour thread                                                       |
| Meena 2      | Second colour thread                                                      |
| Rani (auto)  | Plain-weave base — auto-generated, suppressed wherever another shuttle fires |

---

## Tools / pages

| Page                   | Route        | What it's for                                                                 |
| ---------------------- | ------------ | ----------------------------------------------------------------------------- |
| **Generator**          | `/`          | Main flow: upload → detect colours → assign shuttles → generate BMPs          |
| **Butta Studio**       | `/butta`     | Single-motif reduction: gap-preserving downscale to loom resolution           |
| **BMP Editor**         | `/edit`      | Pixel-level editing of a generated/loaded BMP                                 |
| **Trace Guide**        | `/trace`     | Turns a messy fabric photo into clean tracing references                      |
| **Border Studio**      | `/border`    | High-detail pipeline for thin border / running-line designs                   |
| **Border ID**          | `/border-id` | Border identification and generation                                          |

<!-- SCREENSHOT: a 2x2 grid or a couple of stacked shots of the Generator + BMP Editor -->
<!-- ![Generator](docs/generator.png) -->
<!-- ![BMP Editor](docs/editor.png) -->

### BMP Editor highlights

- Drawing: pencil, eraser, flood fill, line, rectangle, and a **satin/weave brush**
- **Per-region weave fill** — click a single petal/motif to texture only that area
- **Smart fill** and **fill interiors** — fill enclosed shapes with a chosen weave
- Morphology via the engine: dilate, erode, clean noise, close gaps, open, remove isolated
- Transforms: invert, flip H/V, rotate 90/180/270, invert region
- Weave patterns: satin (multiple end-counts), twill, herringbone, basket, honeycomb, diamond, crepe, rib, and more
- Selection with copy / cut / paste / nudge, 50-level undo/redo, zoom & pan
- Exports loom-ready 1-bit BMPs that are byte-identical to the generator's output

---

## Smart detection (optional)

The Generator has a **Smart detect** mode that measurably improves conversion
accuracy on photographed designs. It is off by default; the legacy path is
untouched.

The legacy pipeline resizes the photo to pins x cards and *then* clusters, so a
2px vine in a 3000px photo is already a grey smear before clustering sees it.
Smart mode clusters at source resolution and pools the **label map** down by
area coverage, with a rescue pass that reinstates thin features majority-vote
would drop. It also clusters in CIELAB with luminance down-weighted, which
makes it robust to uneven lighting and to fabric weave texture.

Measured on a synthetic design degraded with a lighting gradient, weave
texture, and JPEG q72 (`python tools/bench_detect.py`):

| | pixel accuracy | thin-line recall | zari IoU | meena IoU |
| --------- | -------------- | ---------------- | -------- | --------- |
| legacy    | 95.8%          | 93.0%            | 77.3%    | 33.4%     |
| smart     | 99.9%          | 98.2%            | 99.4%    | 97.8%     |

Cost: about 3s for a 3000px source vs 0.3s for legacy. No new dependencies, no
network access, and output is deterministic — the same image and settings
always produce the same BMPs.

**Runs entirely on your machine.** No design image is ever uploaded anywhere.

---

## Design assistant (optional, requires an API key)

The Generator can take instructions in plain language — "make the gold satin
finer", "put the green on meena 2", "480 pins" — and translate them into
settings changes.

**The model never touches pixels.** It emits a structured settings patch, which
is validated against loom and shuttle limits before anything runs. Every edit is
executed by the same deterministic code as the manual controls, so output
remains loom-safe and reproducible. The assistant will refuse a request that
cannot be woven — for example, assigning a third thread on a two-shuttle loom —
and explain the real options instead.

### Enabling it

The assistant needs an Anthropic API key, supplied by you. **No key ships with
the app.** Without one the panel is hidden and everything else works normally.

Either set an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

or create `config.json` next to `app.py`:

```json
{ "anthropic_api_key": "sk-ant-..." }
```

`config.json` is gitignored — do not commit it.

The default model is `claude-sonnet-5`. Override it with the
`JQ_ASSISTANT_MODEL` environment variable, or a `"model"` key in
`config.json`. Model IDs are pinned to fixed snapshots, so the assistant's
behaviour will not drift under a design already in production.

**No design image is sent to the API.** The assistant receives only numeric
settings (shuttle count, pins, cards, detected colour count) and your typed
message — never the uploaded photo, the label map, or any BMP data. It uses
`urllib` from the standard library, so there is nothing extra to install.

Note this is the **only** feature that uses the network. Uploads, colour
detection, and BMP generation all stay on your machine.

---

## Installation

Works on Windows, macOS, and Linux.

**1. Install Python 3.9+**
Download from [python.org](https://python.org), or `brew install python3` on macOS.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run**

```bash
python run.py
```

The app opens automatically at **http://localhost:5000**.

---

## Usage

1. **Upload** your design image (JPEG, PNG, BMP, TIFF, WebP; HEIC/HEIF with the optional dependency).
2. **Set Pins** (loom width) and **Cards** (height — auto-computed if left blank).
3. **Choose shuttle count** (1–4).
4. **Detect Colours** — KMeans clusters the image into dominant colours.
5. **Drag colours** into shuttle zones (Zari, Meena 1, Meena 2, Background).
6. **Set the weave** per shuttle (satin end-count, with optional flip).
7. **Generate BMP Files** — download the ZIP, or open the **BMP Editor** to fine-tune first.

---

## Key technical notes

- **Smart fill** — thin design elements (vertical run &lt; n) get a solid fill while thicker fills get satin, so running lines stay crisp and butta bodies stay textured.
- **Phase-corrected Rani** — plain-weave phase is tracked per column and resynced at design boundaries, eliminating mis-picks (weft floats) in multi-shuttle mode.
- **Pixel-perfect label map** — colour assignments from the detect step are carried straight through to generation (no second KMeans run, no boundary drift).
- **Noise removal** — isolated single-pixel KMeans artefacts are stripped before masks are built.
- **Outline / edging** — motif boundaries are extracted with a DISK structuring
  element, so the ring width stays uniform around curves. A square element
  over-erodes on the diagonals: at `stroke_thickness=5` it produced a ring
  spanning 7px of radius instead of 5.
- **Continuous outlines at any reed** — outline masks are pooled to loom
  resolution by area coverage rather than LANCZOS-plus-threshold. Interpolating
  a 1px ring drops most of it below the 50% threshold: a single closed loop
  measured 16 fragments at reed 80 and 42 at reed 60. Coverage pooling keeps it
  as one closed loop, with no dilation of solid fills.
- **Hand-written 1-bit BMP writer** — emits a correct BITMAPINFOHEADER, bottom-up rows, and 4-byte row padding (padded with white) so output is loom-ready and consistent across the generator and editor.

---

## Project structure

```
jacquard-designer/
├── run.py               # App launcher (opens the browser)
├── app.py               # Flask backend — all page and API routes
├── bmp_engine.py        # Core BMP generation: colour detect, fill patterns, writer
├── border_engine.py     # High-detail border generation pipeline
├── border_id_engine.py  # Border identification / generation
├── butta_engine.py      # Butta motif reduction (gap-preserving downscale)
├── enhanced_engine.py   # Image preprocessing helpers (lighting, suggestions)
├── loom_utils.py        # Physical-size conversion + weave-ability warnings
├── templates/
│   ├── index.html       # Generator (main UI)
│   ├── edit.html        # BMP Editor
│   ├── trace.html       # Trace Guide
│   ├── butta.html       # Butta Studio
│   ├── border.html      # Border Studio
│   └── border_id.html   # Border Identification
├── requirements.txt
└── README.md
```

---

## Requirements

```
flask>=2.3.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
scikit-image>=0.21.0
# Optional — for HEIC/HEIF uploads (iPhone photos):
# pillow-heif>=0.15.0
```

---

## Notes

- Runs locally; no data leaves your machine.
- Designed for desktop/laptop screens.
