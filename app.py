"""
Jacquard Designer App — Flask Backend
"""

from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import io, os, zipfile, base64
from bmp_engine import (detect_colors, generate_bmps, verify_bmp, enhance_image,
                        assess_image_quality, extract_outline,
                        generate_fill_pattern, FILL_PATTERNS)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB upload cap

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif'}


def _json_error(msg: str, status: int = 400):
    """Return a JSON error response (never HTML)."""
    return jsonify({'success': False, 'error': msg}), status


@app.errorhandler(413)
def too_large(_e):
    """Override Flask's default HTML 413 page with JSON so the frontend can parse it."""
    return _json_error('File too large. Maximum upload size is 50 MB.', 413)


@app.route('/')
def index():
    from flask import make_response
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma']        = 'no-cache'
    resp.headers['Expires']       = '0'
    return resp


@app.route('/trace')
def trace_page_redirect():
    from flask import make_response
    resp = make_response(render_template('trace.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma']        = 'no-cache'
    return resp


@app.route('/edit')
def edit_page_redirect():
    from flask import make_response
    resp = make_response(render_template('edit.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma']        = 'no-cache'
    return resp


@app.route('/api/fill-patterns', methods=['GET'])
def api_fill_patterns():
    """Return list of available fill patterns for the UI dropdown."""
    return jsonify({'patterns': [
        {'id': k, 'label': v} for k, v in FILL_PATTERNS.items()
    ]})


@app.route('/api/detect-colors', methods=['POST'])
def api_detect_colors():
    """
    Upload image, detect N dominant colours, return swatches + preview.

    Form fields:
        image    : image file
        n_colors : int  — number of colours to detect
        pins     : int  — loom width in threads
        cards    : int  — loom height in cards (optional; auto-computed from aspect ratio)
    """
    try:
        # ── Input validation ─────────────────────────────────────────────────
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')

        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _json_error(
                f'Unsupported file type "{ext}". '
                f'Please upload a JPEG, PNG, BMP, TIFF, or WebP image. '
                f'For HEIC/HEIF (iPhone photos) install pillow-heif: pip install pillow-heif'
            )

        try:
            pins = int(request.form.get('pins', 360))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        if pins < 10:
            return _json_error('Pins must be at least 10.')

        try:
            n_colors = int(request.form.get('n_colors', 4))
        except (ValueError, TypeError):
            return _json_error('n_colors must be a whole number.')
        if n_colors < 1 or n_colors > 16:
            return _json_error('Number of colours must be between 1 and 16.')

        cards_raw = request.form.get('cards', '').strip()
        cards = None
        if cards_raw:
            try:
                cards = int(cards_raw)
                if cards < 10:
                    return _json_error('Cards must be at least 10.')
            except ValueError:
                return _json_error('Cards must be a whole number.')

        # ── Open image ───────────────────────────────────────────────────────
        try:
            img = Image.open(file.stream)
            img = ImageOps.exif_transpose(img).convert('RGB')
        except UnidentifiedImageError:
            return _json_error(
                'Could not read the uploaded file as an image. '
                'HEIC/HEIF files (iPhone photos) require the pillow-heif package — '
                'run: pip install pillow-heif. For all other files, '
                'please check the file is not corrupted.'
            )

        orig_w, orig_h = img.size
        if cards is None:
            cards = max(10, int(pins * orig_h / orig_w))

        # ── Optional image enhancement ───────────────────────────────────────
        if request.form.get('enhance', 'false').lower() == 'true':
            img = enhance_image(img)

        # ── Detect colours ───────────────────────────────────────────────────
        resized = img.resize((pins, cards), Image.LANCZOS)
        colors, counts, label_map, genuine_flags = detect_colors(resized, n_colors)

        total_pixels = pins * cards
        color_data = [
            {
                'index':      i,
                'rgb':        [int(x) for x in color],
                'hex':        '#{:02x}{:02x}{:02x}'.format(*[int(x) for x in color]),
                'percentage': round(100 * count / total_pixels, 1),
                'count':      count,
                'is_genuine': bool(genuine_flags[i]) if i < len(genuine_flags) else True,
            }
            for i, (color, count) in enumerate(zip(colors, counts))
        ]

        # ── Build colour-map preview ─────────────────────────────────────────
        preview_arr = np.zeros((cards, pins, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            preview_arr[label_map == i] = color
        preview_img = Image.fromarray(preview_arr)

        def _to_b64(pil_img, fmt='PNG'):
            buf = io.BytesIO()
            pil_img.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode()

        # ── Encode label_map as lossless PNG ─────────────────────────────────
        # Carried through to /api/generate so BMP generation uses the exact same
        # pixel assignments the user saw in the preview — no second KMeans run.
        label_img = Image.fromarray(label_map.astype(np.uint8), mode='L')

        # Full-resolution source for supersample (fine detail mode)
        # Store original before resizing so supersample can detect at 4× target
        # Cap full_image to 800px max -- prevents huge base64 payloads
        full_img_send = img
        if max(img.width, img.height) > 800:
            scale = 800 / max(img.width, img.height)
            full_img_send = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf_full = io.BytesIO()
        full_img_send.save(buf_full, format='JPEG', quality=85)
        full_image_b64 = base64.b64encode(buf_full.getvalue()).decode()

        return jsonify({
            'success':        True,
            'colors':         color_data,
            'preview_image':  _to_b64(preview_img),
            'original_image': _to_b64(resized),
            'full_image':     full_image_b64,
            'label_map':      _to_b64(label_img),
            'pins':           pins,
            'cards':          cards,
        })

    except Exception as e:
        import traceback
        return _json_error(f'Unexpected error: {e}')


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    Generate BMP files from a previously detected design.

    JSON body:
        image_b64         : base64 PNG of the resized source image
        label_map         : base64 PNG of the colour-index label map
        pins              : int
        cards             : int
        shuttle_count     : int  (1-4)
        design_name       : str
        color_assignments : {color_index_str: shuttle_name}
        satin_settings    : {shuttle_name: {n: int, flip: bool}}
    """
    try:
        if not request.is_json:
            return _json_error('Request must be JSON.')

        data = request.get_json(silent=True)
        if data is None:
            return _json_error('Invalid or empty JSON body.')

        # ── Validate required fields ─────────────────────────────────────────
        for field in ('image_b64', 'pins', 'cards', 'shuttle_count', 'color_assignments'):
            if field not in data:
                return _json_error(f'Missing required field: {field}')

        try:
            pins          = int(data['pins'])
            cards         = int(data['cards'])
            shuttle_count = int(data['shuttle_count'])
        except (ValueError, TypeError) as e:
            return _json_error(f'Invalid numeric field: {e}')

        if pins < 10:
            return _json_error('Pins must be at least 10.')
        if cards < 10:
            return _json_error('Cards must be at least 10.')
        if shuttle_count not in (1, 2, 3, 4):
            return _json_error('Shuttle count must be 1, 2, 3, or 4.')

        # ── Decode image ─────────────────────────────────────────────────────
        try:
            img_bytes = base64.b64decode(data['image_b64'])
            img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception:
            return _json_error('Could not decode image_b64.')

        # Decode full-resolution source image for supersample (fine detail mode).
        # Falls back to the resized image if not provided (older frontend).
        full_img = img   # default: same as resized
        if data.get('full_image'):
            try:
                full_img = Image.open(
                    io.BytesIO(base64.b64decode(data['full_image']))
                ).convert('RGB')
            except Exception:
                full_img = img   # silent fallback

        # ── Sanitise design name ─────────────────────────────────────────────
        design_name = str(data.get('design_name', 'design')).strip() or 'design'
        design_name = ''.join(c for c in design_name if c.isalnum() or c in '_- ')
        design_name = design_name.replace(' ', '_') or 'design'

        # ── Color assignments ─────────────────────────────────────────────────
        try:
            color_assignments = {int(k): str(v)
                                 for k, v in data['color_assignments'].items()}
        except (ValueError, TypeError) as e:
            return _json_error(f'Invalid color_assignments: {e}')

        # ── Satin settings ────────────────────────────────────────────────────
        raw_satin    = data.get('satin_settings', {})
        satin_settings = {}
        valid_n      = {4, 5, 6, 7, 8, 16}
        for k, v in raw_satin.items():
            try:
                n = int(v.get('n', 8))
            except (ValueError, TypeError):
                return _json_error(f'Satin n for "{k}" must be a whole number.')
            if n not in valid_n:
                return _json_error(f'Satin n for "{k}" must be one of {sorted(valid_n)}.')
            min_h = int(v.get('min_height', 35))
            if min_h < 1:   min_h = 1
            if min_h > 999: min_h = 999
            pattern = str(v.get('pattern', 'satin')).lower().strip()
            if pattern not in FILL_PATTERNS:
                pattern = 'satin'
            satin_settings[str(k)] = {
                'n': n, 'flip': bool(v.get('flip', False)),
                'min_height': min_h, 'pattern': pattern,
                'weave_off': bool(v.get('weave_off', False)),
            }

        # ── Decode label_map ──────────────────────────────────────────────────
        label_map = None
        if data.get('label_map'):
            try:
                lm_bytes  = base64.b64decode(data['label_map'])
                lm_img    = Image.open(io.BytesIO(lm_bytes)).convert('L')
                label_map = np.array(lm_img)
            except Exception:
                label_map = None   # fall back to re-running KMeans

        # ── Generate ──────────────────────────────────────────────────────────
        # Emboss: 1-shuttle only — split outline into rani
        emboss      = bool(data.get('emboss', False)) and shuttle_count == 1
        supersample = bool(data.get('supersample', False))

        bmp_files = generate_bmps(
            image=full_img if supersample else img,
            pins=pins,
            cards=cards,
            shuttle_count=shuttle_count,
            color_assignments=color_assignments,
            satin_settings=satin_settings,
            design_name=design_name,
            label_map=label_map,
            emboss=emboss,
            supersample=supersample,
        )

        # ── Verify ────────────────────────────────────────────────────────────
        verification = {fname: verify_bmp(bdata)
                        for fname, bdata in bmp_files.items()}

        # ── ZIP ───────────────────────────────────────────────────────────────
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname, bdata in bmp_files.items():
                zf.writestr(fname, bdata)
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()

        # ── Thumbnail previews (for display cards) ───────────────────────────
        previews = {}
        bmp_b64  = {}   # full-res BMP bytes for the editor
        for fname, bdata in bmp_files.items():
            # Thumbnail: display card preview only (scaled)
            thumb = Image.open(io.BytesIO(bdata)).convert('RGB')
            thumb.thumbnail((300, 300), Image.NEAREST)
            buf = io.BytesIO()
            thumb.save(buf, format='PNG')
            previews[fname] = base64.b64encode(buf.getvalue()).decode()
            # Full-res PNG for editor: convert 1-bit BMP → 8-bit grey PNG
            # (1-bit BMP cannot be decoded by <img> in Safari/some Chrome versions)
            editor_img = Image.open(io.BytesIO(bdata)).convert('L')
            editor_buf = io.BytesIO()
            editor_img.save(editor_buf, format='PNG')
            bmp_b64[fname] = base64.b64encode(editor_buf.getvalue()).decode()

        return jsonify({
            'success':      True,
            'zip_b64':      zip_b64,
            'zip_filename': f'{design_name}_jacquard.zip',
            'files':        list(bmp_files.keys()),
            'verification': verification,
            'previews':     previews,
            'bmp_b64':      bmp_b64,
        })

    except ValueError as e:
        # Raised by generate_bmps for label_map shape mismatch
        return _json_error(str(e))
    except Exception as e:
        import traceback
        return _json_error(f'Generation failed: {e}')


@app.route('/api/assess-quality', methods=['POST'])
def assess_quality():
    """Assess uploaded image quality and return diagnostics + suggestions."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        ext = os.path.splitext(file.filename or '')[1].lower()
        if not file or ext not in ALLOWED_EXTENSIONS:
            return jsonify({'success': False, 'error': 'Invalid image file'})

        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert('RGB')
        quality = assess_image_quality(img)
        return jsonify({'success': True, **quality})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# trace route merged above


@app.route('/api/trace-guide', methods=['POST'])
def api_trace_guide():
    """
    Pre-process a complex fabric image into tracing reference images.
    Returns:
      - faded:       grid faded, design visible (best for manual tracing)
      - highlighted: colour-coded layer map
      - cleaned:     auto-cleaned black-on-white (ready to upload to main app)
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image provided')

        file = request.files['image']
        if not file:
            return _json_error('Empty file')

        raw = file.read()
        if len(raw) > 50 * 1024 * 1024:
            return _json_error('File too large (max 50 MB)')

        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert('RGB')

        # Work at a sensible processing resolution
        MAX_PROC = 1200
        if max(img.width, img.height) > MAX_PROC:
            scale = MAX_PROC / max(img.width, img.height)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

        W, H = img.size
        arr  = np.array(img)
        r_ch = arr[:,:,0].astype(float)
        g_ch = arr[:,:,1].astype(float)
        b_ch = arr[:,:,2].astype(float)

        # HSV for saturation / value / hue analysis
        from PIL import ImageEnhance
        hsv_arr = np.array(img.convert('HSV'))
        hue_f   = hsv_arr[:,:,0].astype(float)
        sat_f   = hsv_arr[:,:,1].astype(float)
        val_f   = hsv_arr[:,:,2].astype(float)

        # ── Detect dominant background colour ─────────────────────────────
        from sklearn.cluster import KMeans
        flat = arr.reshape(-1, 3).astype(np.float32)
        km   = KMeans(n_clusters=2, random_state=42, n_init=10)
        lbs  = km.fit_predict(flat)
        cnts = np.bincount(lbs, minlength=2)
        bg_col = km.cluster_centers_[np.argmax(cnts)]

        # Background mask (pixels close to bg colour)
        diff = np.sqrt(((arr.astype(float) - bg_col) ** 2).sum(axis=2))
        is_bg = diff < 60

        # ── OUTPUT 1: Grid-faded (best for tracing) ───────────────────────
        from scipy.ndimage import median_filter
        faded = arr.copy().astype(float)
        # Detect dominant grid hue (same hue family as bg) and fade it
        bg_hue = float(bg_col[0])  # crude proxy; use red-channel dominance
        r_dom   = r_ch - np.maximum(g_ch, b_ch)
        is_grid = (r_dom > 30) & (val_f > 100)   # reddish/tinted pixels = grid/bg
        # Fade grid toward white
        faded[is_grid] = faded[is_grid] * 0.25 + 255 * 0.75
        faded = np.clip(faded, 0, 255).astype(np.uint8)
        # Boost contrast on non-grid areas
        faded_pil = Image.fromarray(faded)
        faded_pil = ImageEnhance.Contrast(faded_pil).enhance(2.0)
        faded_pil = ImageEnhance.Sharpness(faded_pil).enhance(2.5)
        faded_out = np.array(faded_pil)

        # ── OUTPUT 2: Colour-coded layer highlight ────────────────────────
        highlight = np.full((H, W, 3), 200, dtype=np.uint8)
        # Layers: white body, lavender/wing, gold/detail, everything else = grey
        is_white  = (sat_f < 45)  & (val_f > 200)
        is_cream  = (sat_f < 65)  & (val_f > 165) & ~is_white
        is_lav    = (hue_f >= 165) & (hue_f <= 225) & (sat_f >= 30) & (val_f > 140)
        is_gold   = (hue_f >= 18)  & (hue_f <= 42)  & (sat_f > 70)  & (val_f > 130)
        is_bg_vis = is_bg
        highlight[is_bg_vis]  = [60,  60,  60]
        highlight[is_white]   = [240, 240, 240]
        highlight[is_cream]   = [210, 200, 180]
        highlight[is_lav]     = [130,  90, 200]
        highlight[is_gold]    = [220, 150,  20]

        # ── OUTPUT 3: Auto-cleaned (ready to upload to main app) ──────────
        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label

        # Blur to kill periodic grid
        grid_period = max(3, int(9 * W / 1200))
        blur_r      = max(3, grid_period + 2)
        blurred     = np.stack([median_filter(arr[:,:,c], size=blur_r)
                                for c in range(3)], axis=2)
        hsv_b       = np.array(Image.fromarray(blurred).convert('HSV'))
        sat_b       = hsv_b[:,:,1].astype(float)
        val_b       = hsv_b[:,:,2].astype(float)

        # Detect design in blurred image
        m = (sat_b < 90) & (val_b > 152)
        m = binary_opening(m, structure=np.ones((3, 3)))

        # Keep large connected regions only
        min_region = max(100, int(1500 * (W / 1200) ** 2))
        lbl, n_lbl = label(m)
        sizes = np.bincount(lbl.ravel())[1:]
        clean = np.zeros((H, W), dtype=bool)
        for i, s in enumerate(sizes):
            if s >= min_region:
                clean |= (lbl == i + 1)

        close_px = max(5, int(15 * W / 1200))
        clean    = binary_closing(clean, structure=np.ones((close_px, close_px)))
        clean    = binary_fill_holes(clean)

        cleaned_out = np.full((H, W, 3), 255, dtype=np.uint8)
        cleaned_out[clean] = [0, 0, 0]

        # ── Encode all three outputs as base64 PNG ────────────────────────
        def _to_b64(arr_img):
            buf = io.BytesIO()
            Image.fromarray(arr_img.astype(np.uint8)).save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()

        n_design_regions = int((sizes >= min_region).sum()) if len(sizes) else 0
        design_pct = round(100 * clean.sum() / (H * W), 1)

        return jsonify({
            'success':   True,
            'width':     W,
            'height':    H,
            'faded':     _to_b64(faded_out),
            'highlighted': _to_b64(highlight),
            'cleaned':   _to_b64(cleaned_out),
            'stats': {
                'design_regions': n_design_regions,
                'design_pct':     design_pct,
                'bg_colour':      f'#{int(bg_col[0]):02x}{int(bg_col[1]):02x}{int(bg_col[2]):02x}',
            }
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# edit route merged above


@app.route('/api/bmp-process', methods=['POST'])
def api_bmp_process():
    """
    Apply a server-side morphological operation to a 1-bit BMP.
    Operations: dilate, erode, clean_noise, invert, remove_isolated
    Accepts: image (file) + op (string) + params (JSON)
    Returns: processed image as PNG base64 + stats
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image provided')

        file   = request.files['image']
        op     = request.form.get('op', 'clean_noise')
        import json as _json
        params = _json.loads(request.form.get('params', '{}'))

        raw = file.read()
        buf = io.BytesIO(raw); buf.seek(0)
        img = Image.open(buf).convert('L')
        arr = np.array(img)

        # Binarise: anything < 128 = UP (0), >= 128 = DOWN (255)
        mask = arr < 128   # True = design pixel (UP/black)
        H, W = mask.shape

        from scipy.ndimage import (binary_dilation, binary_erosion,
                                   binary_opening, binary_closing, label)

        result_mask = mask.copy()

        if op == 'dilate':
            r = max(1, int(params.get('radius', 1)))
            struct = np.ones((r*2+1, r*2+1), dtype=bool)
            result_mask = binary_dilation(mask, structure=struct)

        elif op == 'erode':
            r = max(1, int(params.get('radius', 1)))
            struct = np.ones((r*2+1, r*2+1), dtype=bool)
            result_mask = binary_erosion(mask, structure=struct)

        elif op == 'clean_noise':
            min_size = max(1, int(params.get('min_size', 5)))
            lbl, n = label(mask)
            sizes  = np.bincount(lbl.ravel())[1:]
            result_mask = np.zeros_like(mask)
            for i, s in enumerate(sizes):
                if s >= min_size:
                    result_mask |= (lbl == i + 1)

        elif op == 'invert':
            result_mask = ~mask

        elif op == 'remove_isolated':
            # Remove single UP pixels (all 4 neighbours are DOWN)
            result_mask = mask.copy()
            has_up_nb = (
                np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0) |
                np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
            )
            result_mask[mask & ~has_up_nb] = False

        elif op == 'close_gaps':
            r = max(1, int(params.get('radius', 2)))
            struct = np.ones((r*2+1, r*2+1), dtype=bool)
            result_mask = binary_closing(mask, structure=struct)

        elif op == 'open':
            r = max(1, int(params.get('radius', 1)))
            struct = np.ones((r*2+1, r*2+1), dtype=bool)
            result_mask = binary_opening(mask, structure=struct)

        elif op == 'flip_h':
            result_mask = np.fliplr(mask)

        elif op == 'flip_v':
            result_mask = np.flipud(mask)

        elif op == 'rotate_90':
            result_mask = np.rot90(mask, k=1)

        elif op == 'rotate_180':
            result_mask = np.rot90(mask, k=2)

        elif op == 'rotate_270':
            result_mask = np.rot90(mask, k=3)

        elif op == 'fill_pattern':
            # Apply a weave fill pattern inside the design (UP) pixels
            from bmp_engine import generate_fill_pattern
            pat     = params.get('pattern', 'satin')
            n_val   = max(4, min(16, int(params.get('n', 8))))
            flip    = bool(params.get('flip', False))
            min_h   = max(1, int(params.get('min_height', 1)))
            fill    = generate_fill_pattern(pat, n_val, W, H, flip=flip)
            # fill: 0=UP, 1=DOWN  |  mask: True=design pixel
            # Apply: where mask=True, use fill pattern; where mask=False, keep DOWN
            result_mask = mask & (fill == 0)

        else:
            return _json_error(f'Unknown operation: {op}')

        # Build output BMP (1-bit: black=UP, white=DOWN)
        out = np.where(result_mask, np.uint8(0), np.uint8(255))
        out_img = Image.fromarray(out, mode='L')

        # Stats directly (no need to call verify_bmp)
        non_binary = int(((out != 0) & (out != 255)).sum())
        up_px   = int((out == 0).sum())
        down_px = int((out == 255).sum())

        # Encode as PNG
        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Also encode as BMP for download
        bmp_buf = io.BytesIO()
        out_img.save(bmp_buf, format='BMP')
        bmp_b64 = base64.b64encode(bmp_buf.getvalue()).decode()

        return jsonify({
            'success':    True,
            'image_b64':  b64,
            'bmp_b64':    bmp_b64,
            'width':      int(out.shape[1]),
            'height':     int(out.shape[0]),
            'up_pixels':  up_px,
            'down_pixels':down_px,
            'up_pct':     round(100 * up_px / max(up_px + down_px, 1), 2),
            'non_binary': non_binary,
            'is_clean':   non_binary == 0,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


if __name__ == '__main__':
    # Prevent joblib/OpenMP from spawning parallel workers.
    # Required on macOS (avoids 10-30s KMeans hang) and Windows alike.
    import os as _os
    _os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
    _os.environ.setdefault('OMP_NUM_THREADS',    '1')
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)
