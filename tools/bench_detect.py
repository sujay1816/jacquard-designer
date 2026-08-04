"""
Benchmark: legacy detect_colors vs vision_engine.detect_colors_smart.

Builds a synthetic saree design where the ground truth is known exactly, then
degrades it the way a real phone photo of fabric is degraded:
  * lighting gradient across the cloth
  * fabric weave texture (high-frequency noise)
  * JPEG compression artefacts

Ground truth at loom resolution is derived from the CLEAN high-res design by
area coverage, which is what a perfect converter would produce.

Run:  python tools/bench_detect.py
"""
import io
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmp_engine import detect_colors
from vision_engine import detect_colors_smart

HI = 1800           # source photo edge, px
PINS = CARDS = 300  # loom target

BG = (72, 14, 42)        # dark maroon ground
ZARI = (206, 172, 78)    # gold thread
MEENA = (26, 118, 92)    # green accent


def build_design():
    """Clean, full-resolution design. Returns (image, class_map) at HI res."""
    img = Image.new('RGB', (HI, HI), BG)
    d = ImageDraw.Draw(img)

    # Large butta body (thick region — should always survive)
    d.ellipse((520, 460, 1280, 1340), fill=ZARI)
    d.ellipse((700, 660, 1100, 1140), fill=BG)
    d.ellipse((790, 760, 1010, 1040), fill=MEENA)

    # Thin vines — THE critical test. 3px strokes at 1800px is ~0.5 loom cells.
    for i in range(7):
        y = 150 + i * 30
        d.arc((180, y, 1620, y + 900), start=200, end=340, fill=ZARI, width=3)
    # Thin vertical running lines
    for x in (240, 260, 1540, 1560):
        d.line((x, 100, x, 1700), fill=ZARI, width=3)
    # Fine meena outline ring around the butta
    d.ellipse((500, 440, 1300, 1360), outline=MEENA, width=4)

    a = np.asarray(img)
    cls = np.zeros((HI, HI), dtype=np.int32)
    cls[np.all(a == ZARI, axis=2)] = 1
    cls[np.all(a == MEENA, axis=2)] = 2
    return img, cls


def ground_truth(cls):
    """Pool the clean class map to loom resolution by area coverage."""
    k = HI // PINS
    cov = np.stack([
        (cls == c).astype(np.float32).reshape(CARDS, k, PINS, k).sum(axis=(1, 3)) / (k * k)
        for c in range(3)
    ])
    gt = cov.argmax(axis=0)
    # Thin features that cover a meaningful slice of a cell belong to the
    # design, not the ground — that is what a correct converter must keep.
    for c in (2, 1):
        gt[(gt == 0) & (cov[c] >= 0.22)] = c
    return gt


def degrade(img):
    """Simulate a phone photo of fabric: lighting, weave texture, JPEG."""
    a = np.asarray(img, dtype=np.float32)

    # Diagonal lighting gradient, 55% to 135% brightness
    yy, xx = np.mgrid[0:HI, 0:HI]
    field = 0.55 + 0.80 * ((xx + yy) / (2.0 * HI))
    a *= field[:, :, None]

    # Fabric weave texture
    rng = np.random.default_rng(7)
    weave = 7.0 * np.sin(xx / 1.7) * np.sin(yy / 1.7)
    a += weave[:, :, None] + rng.normal(0, 4.0, a.shape)

    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format='JPEG', quality=72)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def match_classes(pred, gt, n=3):
    """
    Optimally map predicted cluster indices onto ground-truth classes.

    Detection is unsupervised so index order is arbitrary. Greedy matching lets
    two predictions claim the same class and strands a third, so use Hungarian
    assignment on the overlap matrix for a one-to-one mapping.
    """
    from scipy.optimize import linear_sum_assignment
    overlap = np.zeros((n, n), dtype=np.int64)
    for p in range(n):
        pm = (pred == p)
        for g in range(n):
            overlap[p, g] = np.logical_and(pm, gt == g).sum()
    rows, cols = linear_sum_assignment(-overlap)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    return np.vectorize(lambda v: mapping.get(int(v), 0))(pred)


def score(name, mapped, gt, thin_mask):
    rows = []
    for c, label in ((1, 'zari'), (2, 'meena')):
        tp = np.logical_and(mapped == c, gt == c).sum()
        fp = np.logical_and(mapped == c, gt != c).sum()
        fn = np.logical_and(mapped != c, gt == c).sum()
        iou = tp / max(tp + fp + fn, 1)
        rec = tp / max(tp + fn, 1)
        rows.append((label, iou, rec))
    thin_total = thin_mask.sum()
    thin_kept = np.logical_and(thin_mask, mapped > 0).sum()
    thin_rec = thin_kept / max(thin_total, 1)
    overall = (mapped == gt).mean()
    print(f"  {name:<10}  pixel acc {overall*100:5.1f}%   "
          f"thin-line recall {thin_rec*100:5.1f}%   " +
          "   ".join(f"{l} IoU {i*100:4.1f}%" for l, i, _ in rows))
    return overall, thin_rec


def main():
    clean, cls = build_design()
    gt = ground_truth(cls)
    photo = degrade(clean)

    # Thin features = design cells whose 3x3 neighbourhood is mostly background
    from scipy.ndimage import uniform_filter
    design = (gt > 0).astype(np.float32)
    thin_mask = (design > 0) & (uniform_filter(design, size=5) < 0.45)

    print(f"\nCanvas {PINS}x{CARDS} from a {HI}x{HI} photo "
          f"(lighting gradient + weave texture + JPEG q72)")
    print(f"Ground truth: {int((gt>0).sum()):,} design cells, "
          f"of which {int(thin_mask.sum()):,} are thin features\n")

    small = photo.resize((PINS, CARDS), Image.LANCZOS)
    _, _, lm_old, _ = detect_colors(small, 3)
    score('legacy', match_classes(lm_old.astype(int), gt), gt, thin_mask)

    _, _, lm_new, _ = detect_colors_smart(photo, 3, PINS, CARDS)
    score('smart', match_classes(lm_new.astype(int), gt), gt, thin_mask)

    print()
    for flag in ('flatten_light', 'superpixels', 'thin_rescue'):
        _, _, lm, _ = detect_colors_smart(photo, 3, PINS, CARDS, **{flag: False})
        score(f'no {flag[:9]}', match_classes(lm.astype(int), gt), gt, thin_mask)
    print()


if __name__ == '__main__':
    main()
