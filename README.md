# Jacquard Designer

A web-based tool that converts saree/textile design images into **1-bit BMP files** for driving a jacquard weaving loom.

## What it does

Upload a cropped design image (Butta motif, running lines, etc.), set your loom's pin and card count, assign colours to shuttles, choose a satin weave type, and download ready-to-use BMP files — one per shuttle.

**BMP pixel convention:**
- Black (0) = thread UP (visible on fabric)
- White (255) = thread DOWN (hidden)

## Shuttle types

| Shuttle | Purpose |
|---------|---------|
| Zari | Gold thread — satin or solid fill |
| Meena 1 | First colour thread |
| Meena 2 | Second colour thread |
| Rani (auto) | Plain weave base — auto-generated, suppressed where other shuttles fire |

## Installation

### Windows / Mac / Linux

**1. Install Python 3.9+**
- Download from [python.org](https://python.org) or use `brew install python3` on Mac.

**2. Install dependencies**
```bash
pip install flask pillow numpy scikit-learn scipy scikit-image
```

> **macOS note:** To upload HEIC/HEIF photos taken on iPhone, also run:
> `pip install pillow-heif`

**3. Run**
```bash
python run.py
```

The app opens automatically at **http://localhost:5000**

## Usage

1. **Upload** your design image (JPEG, PNG, BMP, TIFF, WebP; HEIC with pillow-heif installed)
2. **Set Pins** (loom width) and Cards (height — auto-computed if left blank)
3. **Choose shuttle count** (1–4)
4. **Detect Colours** — KMeans clusters the image into dominant colours
5. **Drag colours** into shuttle zones (Zari, Meena 1, Meena 2, Background)
6. **Set satin type** per shuttle (4/5/6/7/8/16-end, with optional flip)
7. **Generate BMP Files** — download the ZIP

## Key technical notes

- **Smart fill**: thin design elements (vertical run < n) get solid fill; thick fills get satin. This keeps running lines crisp and Butta bodies textured.
- **Phase-corrected Rani**: plain weave phase is tracked per column and resynced at design boundaries — eliminates mis-picks (weft floats) in multi-shuttle mode.
- **Pixel-perfect label map**: colour assignments from the detect step are carried through to BMP generation — no second KMeans run, no boundary drift.
- **Noise removal**: isolated 1-pixel KMeans artefacts are stripped before generating masks.

## Requirements

```
flask>=2.3.0
pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

## Project structure

```
jacquard_design-2/
├── run.py              # App launcher
├── app.py              # Flask backend (API routes)
├── bmp_engine.py       # Core BMP generation logic
├── templates/
│   └── index.html      # Single-page UI
├── requirements.txt
└── README.md
```
