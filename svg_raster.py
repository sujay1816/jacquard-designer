"""
A rasteriser for the SVG this project generates. No native libraries.

Why this exists: cairosvg needs Cairo, a C library pip cannot install. On
Windows that means downloading a GTK runtime; on Apple Silicon it means brew
plus a DYLD path variable. Both are real installation work, and both fail in a
way that looks like a Python problem while pip insists everything is fine.

The dependency was never justified. cairosvg is a general SVG renderer, and this
project does not have general SVG — it has SVG that motif_library wrote a few
milliseconds earlier. Surveyed across every motif and every layout, the complete
feature set is:

    elements    svg, g, rect, circle, line, polyline, path
    path data   M, C, Q, Z
    transforms  translate(), scale()
    paint       black, white, #000000, none — no gradients, no opacity
    attributes  fill, stroke, stroke-width, stroke-linecap, stroke-linejoin

That is a small, closed, self-imposed grammar. Supporting it needs a few hundred
lines against PIL, which is already a hard dependency. Supporting it via Cairo
needs a C toolchain on every machine the product ships to.

Quality note: everything is drawn at SS× scale and downsampled with LANCZOS.
Anti-aliasing matters more here than in ordinary graphics, because the output is
immediately thresholded onto a thread grid — a jagged stroke edge becomes a
visibly ragged line of lifts, and no later stage can recover it.

cairosvg is still used when it is available and working; this is the fallback.
Their outputs are compared in tools/test_svg_raster.py so the two do not drift.
"""
import math
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

SS = 4                      # supersample factor
_NUM = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')
_CMD = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])')
_TF = re.compile(r'(translate|scale|rotate|matrix)\s*\(([^)]*)\)')

_NAMED = {'black': (0, 0, 0), 'white': (255, 255, 255),
          'none': None, 'transparent': None}


# ── Geometry ────────────────────────────────────────────────────────────────

class Matrix:
    """2D affine transform as (a, b, c, d, e, f), applied as SVG does."""

    __slots__ = ('a', 'b', 'c', 'd', 'e', 'f')

    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    def mul(self, o):
        """self then o applied to a point == self.mul(o) applied to it."""
        return Matrix(self.a * o.a + self.c * o.b,
                      self.b * o.a + self.d * o.b,
                      self.a * o.c + self.c * o.d,
                      self.b * o.c + self.d * o.d,
                      self.a * o.e + self.c * o.f + self.e,
                      self.b * o.e + self.d * o.f + self.f)

    def apply(self, x, y):
        return (self.a * x + self.c * y + self.e,
                self.b * x + self.d * y + self.f)

    def scale_factor(self):
        """
        Mean linear scale, for turning stroke-width into device pixels.

        The geometric mean of the two axis scales is what SVG's own
        non-scaling-stroke maths uses, and it is right for the uniform
        translate/scale transforms this project emits.
        """
        return math.sqrt(abs(self.a * self.d - self.b * self.c)) or 1.0


def parse_transform(text):
    m = Matrix()
    for fn, args in _TF.findall(text or ''):
        v = [float(x) for x in _NUM.findall(args)]
        if fn == 'translate':
            t = Matrix(1, 0, 0, 1, v[0] if v else 0, v[1] if len(v) > 1 else 0)
        elif fn == 'scale':
            sx = v[0] if v else 1
            t = Matrix(sx, 0, 0, v[1] if len(v) > 1 else sx, 0, 0)
        elif fn == 'rotate':
            a = math.radians(v[0] if v else 0)
            cos, sin = math.cos(a), math.sin(a)
            t = Matrix(cos, sin, -sin, cos, 0, 0)
            if len(v) >= 3:                       # rotate about a point
                cx, cy = v[1], v[2]
                t = (Matrix(1, 0, 0, 1, cx, cy).mul(t)
                     .mul(Matrix(1, 0, 0, 1, -cx, -cy)))
        elif fn == 'matrix' and len(v) >= 6:
            t = Matrix(*v[:6])
        else:
            continue
        m = m.mul(t)
    return m


def _bezier3(p0, p1, p2, p3, n):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * u * p0[0] + 3 * u * u * t * p1[0]
                    + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3 * u * u * t * p1[1]
                    + 3 * u * t * t * p2[1] + t * t * t * p3[1]))
    return out


def _bezier2(p0, p1, p2, n):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return out


def _steps(a, b, scale):
    """
    How finely to flatten a curve.

    Proportional to on-screen size: a curve spanning three pixels needs three
    segments, one spanning three hundred needs far more. A fixed count either
    wastes work on small motifs or visibly facets large ones — and faceting
    survives into the thread grid as a straight run where a curve belongs.
    """
    d = math.hypot(b[0] - a[0], b[1] - a[1]) * scale
    return max(4, min(96, int(d / 3) + 4))


def parse_path(d, m):
    """Flatten path data to a list of device-space polylines and closed flags."""
    tokens = _CMD.split(d or '')
    subpaths, pts = [], []
    cur = start = (0.0, 0.0)
    scale = m.scale_factor()
    i, cmd = 1, None

    def flush(closed):
        if len(pts) > 1:
            subpaths.append(([m.apply(*p) for p in pts], closed))
        del pts[:]

    while i < len(tokens):
        cmd = tokens[i]
        nums = [float(x) for x in _NUM.findall(tokens[i + 1] if i + 1 < len(tokens) else '')]
        i += 2
        rel = cmd.islower()
        up = cmd.upper()
        j = 0

        if up == 'Z':
            if pts:
                pts.append(start)
                flush(True)
            cur = start
            continue

        while True:
            if up == 'M':
                if j + 1 >= len(nums):
                    break
                p = (nums[j] + (cur[0] if rel else 0), nums[j + 1] + (cur[1] if rel else 0))
                if pts:
                    flush(False)
                cur = start = p
                pts.append(p)
                j += 2
                up = 'L'                     # extra pairs after M are lineto
            elif up == 'L':
                if j + 1 >= len(nums):
                    break
                cur = (nums[j] + (cur[0] if rel else 0), nums[j + 1] + (cur[1] if rel else 0))
                pts.append(cur)
                j += 2
            elif up == 'H':
                if j >= len(nums):
                    break
                cur = (nums[j] + (cur[0] if rel else 0), cur[1])
                pts.append(cur)
                j += 1
            elif up == 'V':
                if j >= len(nums):
                    break
                cur = (cur[0], nums[j] + (cur[1] if rel else 0))
                pts.append(cur)
                j += 1
            elif up == 'C':
                if j + 5 >= len(nums):
                    break
                ox, oy = (cur if rel else (0, 0))
                p1 = (nums[j] + ox, nums[j + 1] + oy)
                p2 = (nums[j + 2] + ox, nums[j + 3] + oy)
                p3 = (nums[j + 4] + ox, nums[j + 5] + oy)
                pts.extend(_bezier3(cur, p1, p2, p3, _steps(cur, p3, scale)))
                cur = p3
                j += 6
            elif up == 'Q':
                if j + 3 >= len(nums):
                    break
                ox, oy = (cur if rel else (0, 0))
                p1 = (nums[j] + ox, nums[j + 1] + oy)
                p2 = (nums[j + 2] + ox, nums[j + 3] + oy)
                pts.extend(_bezier2(cur, p1, p2, _steps(cur, p2, scale)))
                cur = p2
                j += 4
            else:
                break
    if pts:
        flush(False)
    return subpaths


# ── Painting ────────────────────────────────────────────────────────────────

def _colour(v, inherited):
    if v is None:
        return inherited
    v = v.strip().lower()
    if v in _NAMED:
        return _NAMED[v]
    if v.startswith('#'):
        h = v[1:]
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        if len(h) == 6:
            return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
    if v == 'currentcolor':
        return inherited
    return inherited


def _tag(el):
    return el.tag.rsplit('}', 1)[-1]


def _draw_shape(dr, pts, closed, fill, stroke, width, cap):
    if fill is not None and closed and len(pts) > 2:
        dr.polygon(pts, fill=fill)
    if stroke is not None and len(pts) > 1:
        w = max(1, int(round(width)))
        # joint='curve' rounds the corners between segments. Without it a
        # flattened bezier shows a notch at every joint, which thresholds into
        # a visible nick in the thread run.
        dr.line(pts, fill=stroke, width=w, joint='curve')
        if w > 2 and cap != 'butt':
            r = w / 2.0
            for p in (pts[0], pts[-1]):
                dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=stroke)


def _render_el(el, dr, m, style):
    tag = _tag(el)
    fill = _colour(el.get('fill'), style['fill'])
    stroke = _colour(el.get('stroke'), style['stroke'])
    try:
        sw = float(el.get('stroke-width', style['stroke_width']))
    except (TypeError, ValueError):
        sw = style['stroke_width']
    cap = el.get('stroke-linecap', style['cap'])
    local = m.mul(parse_transform(el.get('transform')))
    dev_w = sw * local.scale_factor()
    st = {'fill': fill, 'stroke': stroke, 'stroke_width': sw, 'cap': cap}

    if tag == 'g' or tag == 'svg':
        for child in el:
            _render_el(child, dr, local, st)
        return

    if tag == 'rect':
        x, y = float(el.get('x', 0)), float(el.get('y', 0))
        w, h = float(el.get('width', 0)), float(el.get('height', 0))
        pts = [local.apply(x, y), local.apply(x + w, y),
               local.apply(x + w, y + h), local.apply(x, y + h)]
        _draw_shape(dr, pts + [pts[0]], True, fill, stroke, dev_w, cap)

    elif tag == 'circle':
        cx, cy, r = (float(el.get('cx', 0)), float(el.get('cy', 0)),
                     float(el.get('r', 0)))
        # Transformed as a polygon rather than a bounding box, so a circle
        # inside a non-uniform scale comes out the ellipse it should be.
        n = max(12, min(160, int(r * local.scale_factor() * 1.6) + 12))
        pts = [local.apply(cx + r * math.cos(2 * math.pi * k / n),
                           cy + r * math.sin(2 * math.pi * k / n))
               for k in range(n + 1)]
        _draw_shape(dr, pts, True, fill, stroke, dev_w, cap)

    elif tag == 'ellipse':
        cx, cy = float(el.get('cx', 0)), float(el.get('cy', 0))
        rx, ry = float(el.get('rx', 0)), float(el.get('ry', 0))
        n = max(12, min(160, int(max(rx, ry) * local.scale_factor() * 1.6) + 12))
        pts = [local.apply(cx + rx * math.cos(2 * math.pi * k / n),
                           cy + ry * math.sin(2 * math.pi * k / n))
               for k in range(n + 1)]
        _draw_shape(dr, pts, True, fill, stroke, dev_w, cap)

    elif tag == 'line':
        pts = [local.apply(float(el.get('x1', 0)), float(el.get('y1', 0))),
               local.apply(float(el.get('x2', 0)), float(el.get('y2', 0)))]
        _draw_shape(dr, pts, False, None, stroke or fill, dev_w, cap)

    elif tag in ('polyline', 'polygon'):
        v = [float(x) for x in _NUM.findall(el.get('points', ''))]
        pts = [local.apply(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)]
        closed = tag == 'polygon'
        if closed and pts:
            pts.append(pts[0])
        _draw_shape(dr, pts, closed, fill if closed else None,
                    stroke or (fill if not closed else None), dev_w, cap)

    elif tag == 'path':
        for pts, closed in parse_path(el.get('d'), local):
            _draw_shape(dr, pts, closed, fill, stroke, dev_w, cap)


def render(svg: str, width: int, height: int, background='white'):
    """Rasterise to a PIL RGB image of exactly (width, height)."""
    root = ET.fromstring(svg)
    vb = [float(x) for x in _NUM.findall(root.get('viewBox', ''))]
    if len(vb) == 4:
        vx, vy, vw, vh = vb
    else:
        vx = vy = 0.0
        vw = float(_NUM.findall(root.get('width', '1000'))[0])
        vh = float(_NUM.findall(root.get('height', '1000'))[0])
    vw = vw or 1.0
    vh = vh or 1.0

    W, H = max(1, int(width) * SS), max(1, int(height) * SS)
    img = Image.new('RGB', (W, H), background or 'white')
    dr = ImageDraw.Draw(img)

    base = Matrix(W / vw, 0, 0, H / vh, -vx * W / vw, -vy * H / vh)
    style = {'fill': (0, 0, 0), 'stroke': None, 'stroke_width': 1.0, 'cap': 'butt'}
    for child in root:
        _render_el(child, dr, base.mul(parse_transform(root.get('transform'))), style)

    return img.resize((max(1, int(width)), max(1, int(height))), Image.LANCZOS)


def available():
    """Always true — the point of this module is that it cannot be unavailable."""
    return True
