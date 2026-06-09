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
├── enhanced_engine.py   # Image preprocessing helpers (lighting, suggestions)
├── templates/
│   ├── index.html       # Generator (main UI)
│   ├── edit.html        # BMP Editor
│   ├── trace.html       # Trace Guide
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
