"""
Jacquard Designer App — Flask Backend
"""

from flask import Flask, request, jsonify, render_template, session
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import io, os, zipfile, base64
from bmp_engine import (detect_colors, generate_bmps, verify_bmp, enhance_image,
                        assess_image_quality, extract_outline,
                        generate_fill_pattern, FILL_PATTERNS, write_1bit_bmp)
from border_engine import generate_border_bmps, detect_border
from border_id_engine import generate_border_id_bmps
from enhanced_engine import preprocess_fabric_image, analyze_border_image
import butta_engine

app = Flask(__name__)
app.secret_key = os.environ.get('JQ_SECRET_KEY', 'jq-designer-2024')
_bmp_store = {}  # token → {bmp_b64, filename, preview}
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


@app.route('/api/store-bmp', methods=['POST'])
def api_store_bmp():
    import uuid as _uuid
    try:
        data  = request.get_json()
        bmp_b64 = data.get('bmp_b64', '')
        try:
            if not bmp_b64 or not base64.b64decode(bmp_b64, validate=True):
                return _json_error('Invalid or empty BMP data.')
        except Exception:
            return _json_error('BMP data is not valid base64.')
        token = str(_uuid.uuid4())
        _bmp_store[token] = {
            'bmp_b64':  bmp_b64,
            'filename': data.get('filename', 'design.bmp'),
            'preview':  data.get('preview', ''),
        }
        if len(_bmp_store) > 30:
            del _bmp_store[next(iter(_bmp_store))]
        return jsonify({'success': True, 'token': token})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/edit')
def edit_page_redirect():
    import uuid as _uuid
    token    = request.args.get('token', '')
    entry    = _bmp_store.pop(token, {}) if token else {}
    bmp_b64  = entry.get('bmp_b64', '')
    filename = entry.get('filename', '')
    preview  = entry.get('preview', '')
    from flask import make_response
    resp = make_response(render_template('edit.html',
                         editor_bmp_b64=bmp_b64,
                         editor_filename=filename,
                         editor_preview=preview))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
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
            img = preprocess_fabric_image(img)   # lighting normalisation + texture suppression
            img = enhance_image(img)             # sharpen / contrast (existing step)

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
        hollow_weave_settings = data.get('hollow_weave_settings', None)
        outline_white   = data.get('outline_white',   None)
        invert_output   = data.get('invert_output',   None)
        bg_texture      = data.get('bg_texture',      None)
        # Stroke mode (default True for 2/3/4 shuttle): thin design to 1px outline ring
        stroke_mode = bool(data.get('stroke_mode', True))

        # New enhancement parameters
        reed              = max(1,  min(200, int(data.get('reed', 80))))
        stroke_thickness  = max(1,  min(5,   int(data.get('stroke_thickness', 1))))
        rani_weave        = str(data.get('rani_weave', 'plain'))
        if rani_weave not in ('plain', 'twill', 'matt'):
            rani_weave = 'plain'
        curvilinear_satin = bool(data.get('curvilinear_satin', False))

        # Auto-preprocessing: apply JPEG deblock + enhance when artifacts detected.
        # Runs a quick quality check on the image and auto-enhances if needed.
        try:
            from bmp_engine import assess_image_quality
            _q = assess_image_quality(img)
            if _q.get('jpeg_artifacts', 0) > 15 or _q.get('noise_level', 0) > 20:
                img = preprocess_fabric_image(img)
                img = enhance_image(img)
        except Exception:
            pass

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
            hollow_weave_settings=hollow_weave_settings,
            outline_white=outline_white,
            invert_output=invert_output,
            bg_texture=bg_texture,
            stroke_mode=stroke_mode,
            reed=reed,
            stroke_thickness=stroke_thickness,
            rani_weave=rani_weave,
            curvilinear_satin=curvilinear_satin,
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
            # Raw BMP bytes for editor — parseBmpBytes in JS handles 1-bit BMP
            # directly without any browser <img> rendering dependency
            bmp_b64[fname] = base64.b64encode(bdata).decode()

        return jsonify({
            'success':      True,
            'zip_b64':      zip_b64,
            'zip_filename': f'{design_name}_jacquard.zip',
            'files':        list(bmp_files.keys()),
            'verification': verification,
            'previews':     previews,
            'bmp_b64':      bmp_b64,
            'warnings':     _design_warnings(bmp_files, pins, cards),
            'composite_b64': _composite_render(bmp_files, pins, cards),
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
        from PIL import ImageEnhance

        # Read bg_type before computing faded reference so we can tune it
        bg_type = request.form.get('bg_type', 'auto')
        if bg_type == 'auto':
            bg_is_light = bool(float(bg_col.mean()) > 128)
        elif bg_type == 'light':
            bg_is_light = True
        else:
            bg_is_light = False  # 'dark'

        # Fade pixels that are similar to the detected background colour.
        # Works for ANY background colour (purple, grey, white, red, …).
        # The old reddish-channel heuristic (r_dom > 30) only worked for
        # warm/reddish backgrounds and did nothing for purple, grey or white.
        bg_fade_mask = diff < 80          # slightly wider than is_bg
        faded = arr.copy().astype(float)
        if bg_is_light:
            # Light background (white paper, cream etc.) — push bg pixels to
            # pure white more aggressively so any texture/grain disappears
            faded[bg_fade_mask] = faded[bg_fade_mask] * 0.05 + 255 * 0.95
        else:
            # Dark background (coloured cloth, fabric photos) — bleach the
            # dark bg toward white so the lighter design pops out clearly
            faded[bg_fade_mask] = faded[bg_fade_mask] * 0.15 + 255 * 0.85
        faded = np.clip(faded, 0, 255).astype(np.uint8)
        # Boost contrast and sharpness on the non-faded design pixels
        faded_pil = Image.fromarray(faded)
        faded_pil = ImageEnhance.Contrast(faded_pil).enhance(2.5)
        faded_pil = ImageEnhance.Sharpness(faded_pil).enhance(2.5)
        faded_out = np.array(faded_pil)

        # ── OUTPUT 2: Colour-coded layer highlight ────────────────────────
        # (Computed dynamically below with KMeans — no hardcoded bird colours)
        highlight = np.full((H, W, 3), [60, 60, 60], dtype=np.uint8)  # default: all bg-grey

        # ── OUTPUT 3: Auto-cleaned (ready to upload to main app) ──────────
        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label

        # User-adjustable strength (1 = conservative, 5 = aggressive capture)
        try:
            strength = max(1, min(5, int(request.form.get('strength', 3))))
        except (ValueError, TypeError):
            strength = 3

        # Blur to kill periodic grid
        grid_period = max(3, int(9 * W / 1200))
        blur_r      = max(3, grid_period + 2)
        blurred     = np.stack([median_filter(arr[:,:,c], size=blur_r)
                                for c in range(3)], axis=2)

        # ── Universal design detection: pixels far from background colour ──
        # Works for both dark-on-light (outlines, sketches) and
        # light-on-dark (coloured motifs on fabric photos).
        bg_dist_blurred = np.sqrt(
            ((blurred.astype(float) - bg_col) ** 2).sum(axis=2))
        # Lower strength → higher threshold (stricter) → fewer pixels kept
        dist_threshold = max(15, int(85 - strength * 12))   # 73,61,49,37,25
        m = bg_dist_blurred > dist_threshold
        m = binary_opening(m, structure=np.ones((3, 3)))

        # Keep large connected regions only — vectorised (np.isin) instead of
        # a Python for-loop that creates a full boolean array per region.
        min_region = max(50, int(1200 * (W / 1200) ** 2))
        lbl, n_lbl = label(m)
        sizes = np.bincount(lbl.ravel())[1:]
        if len(sizes):
            large_labels = np.where(sizes >= min_region)[0] + 1  # labels start at 1
            clean = np.isin(lbl, large_labels)
        else:
            clean = np.zeros((H, W), dtype=bool)

        close_px = max(5, int(15 * W / 1200))
        clean    = binary_closing(clean, structure=np.ones((close_px, close_px)))
        clean    = binary_fill_holes(clean)

        cleaned_out = np.full((H, W, 3), 255, dtype=np.uint8)
        cleaned_out[clean] = [0, 0, 0]

        # ── Dynamic Layer Map: KMeans on actual design colours ─────────────
        # n_lbl counts *spatial regions*, not distinct colours — wrong proxy.
        # Use a fixed sensible cluster count (4 covers most real designs).
        from sklearn.cluster import KMeans as _KM
        design_mask = ~is_bg
        design_pix  = arr[design_mask].astype(np.float32)
        n_layers    = 4          # sensible default: covers most design colour ranges
        legend_colors = []
        if len(design_pix) >= n_layers:
            km2  = _KM(n_clusters=n_layers, n_init=4, random_state=42,
                       max_iter=80).fit(design_pix)
            ctrs = km2.cluster_centers_.astype(np.uint8)
            # Sort by brightness (lightest first) for a consistent legend
            order = np.argsort(-ctrs.mean(axis=1))
            ctrs  = ctrs[order]

            # ── Vectorised nearest-centre assignment ─────────────────────
            # Replace the old pixel-by-pixel Python loop (O(N) in Python)
            # with a fully vectorised numpy operation.  For a 1200×1200
            # image with 50% design pixels this is ~300× faster.
            dm_idx   = np.where(design_mask.ravel())[0]      # (N,)
            diff2    = ((design_pix[:, None, :] -             # (N,1,3)
                         ctrs[None, :, :].astype(np.float32)  # (1,K,3)
                        ) ** 2).sum(axis=2)                   # (N,K)
            lbls_d   = diff2.argmin(axis=1)                   # (N,)

            # Write results back into the highlight array in one vectorised step
            highlight_flat       = highlight.reshape(-1, 3)
            highlight_flat[dm_idx] = ctrs[lbls_d]
            # Re-stamp detected bg pixels as dark grey (some may have been
            # overwritten if they sat inside the design_mask boundary)
            highlight[is_bg] = [60, 60, 60]

            legend_colors = [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}'
                             for c in ctrs]

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
            'bg_is_light':  bg_is_light,
            'legend_colors': legend_colors,
            'stats': {
                'design_regions': n_design_regions,
                'design_pct':     design_pct,
                'bg_colour':      f'#{int(bg_col[0]):02x}{int(bg_col[1]):02x}{int(bg_col[2]):02x}',
            }
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e)})


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
        try:
            params = _json.loads(request.form.get('params', '{}'))
        except (ValueError, TypeError):
            return _json_error('Invalid params: must be valid JSON.')
        if not isinstance(params, dict):
            return _json_error('Invalid params: must be a JSON object.')

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

        elif op == 'fill_rani_weave':
            # Rani Weave Fill: fill all white (DOWN) pixels with 1/1 plain weave
            # while preserving every existing black (UP) pixel unchanged.
            # Used on rani BMPs to add the plain weave background after generation.
            Y_pw, X_pw = np.mgrid[0:H, 0:W]
            plain_weave_pixels = (Y_pw + X_pw) % 2 == 0   # True = would fire in plain weave
            # Keep existing black + add plain weave to white areas
            result_mask = mask | plain_weave_pixels

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

        # Also encode as a TRUE 1-bit BMP (loom format) via the engine's writer.
        # write_1bit_bmp expects 0 = black/UP, 1 = white/DOWN.
        bmp_bytes = write_1bit_bmp(np.where(result_mask, np.uint8(0), np.uint8(1)))
        bmp_b64 = base64.b64encode(bmp_bytes).decode()

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
        return jsonify({'success': False, 'error': str(e)})


@app.route('/border')
def border_page():
    """Dedicated page for border / running-line designs (high-detail mode)."""
    from flask import make_response
    resp = make_response(render_template('border.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma']        = 'no-cache'
    resp.headers['Expires']       = '0'
    return resp


@app.route('/api/border-detect', methods=['POST'])
def api_border_detect():
    """
    Ink-first colour detection for border designs.

    Separates ink from paper with an adaptive threshold, then classifies only
    the ink pixels into true thread colours — avoiding the muddy "blend halo"
    cluster a flat KMeans produces on fine line-art.

    Form fields: image, pins, cards (optional), n_colors (total incl. paper),
                 ink_sensitivity (optional, default 1.0)
    Response shape matches /api/detect-colors plus ink_centers.
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')
        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _json_error(f'Unsupported file type "{ext}".')

        try:
            pins = int(request.form.get('pins', 400))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        if pins < 10:
            return _json_error('Pins must be at least 10.')

        try:
            n_colors = int(request.form.get('n_colors', 3))
        except (ValueError, TypeError):
            return _json_error('n_colors must be a whole number.')
        n_colors = max(2, min(8, n_colors))
        max_ink = max(1, n_colors - 1)

        try:
            ink_sensitivity = float(request.form.get('ink_sensitivity', 1.0))
        except (ValueError, TypeError):
            ink_sensitivity = 1.0
        ink_sensitivity = max(0.25, min(3.0, ink_sensitivity))

        cards_raw = request.form.get('cards', '').strip()
        cards = None
        if cards_raw:
            try:
                cards = int(cards_raw)
                if cards < 10:
                    return _json_error('Cards must be at least 10.')
            except ValueError:
                return _json_error('Cards must be a whole number.')

        try:
            img = Image.open(file.stream)
            img = ImageOps.exif_transpose(img).convert('RGB')
        except UnidentifiedImageError:
            return _json_error('Could not read the uploaded file as an image.')

        orig_w, orig_h = img.size
        if cards is None:
            cards = max(10, int(pins * orig_h / orig_w))

        det = detect_border(img, pins, cards,
                            max_ink_colors=max_ink, ink_sensitivity=ink_sensitivity)

        def _to_b64(pil_img, fmt='PNG'):
            buf = io.BytesIO(); pil_img.save(buf, format=fmt)
            return base64.b64encode(buf.getvalue()).decode()

        resized   = img.resize((pins, cards), Image.LANCZOS)
        preview_img = Image.fromarray(det['preview'])
        label_img   = Image.fromarray(det['label_map'].astype(np.uint8), mode='L')

        full_img_send = img
        if max(img.width, img.height) > 800:
            sc = 800 / max(img.width, img.height)
            full_img_send = img.resize((int(img.width * sc), int(img.height * sc)),
                                       Image.LANCZOS)
        buf_full = io.BytesIO(); full_img_send.save(buf_full, format='JPEG', quality=85)

        return jsonify({
            'success':        True,
            'colors':         det['colors'],
            'ink_centers':    det['ink_centers'],
            'preview_image':  _to_b64(preview_img),
            'original_image': _to_b64(resized),
            'full_image':     base64.b64encode(buf_full.getvalue()).decode(),
            'label_map':      _to_b64(label_img),
            'pins':           pins,
            'cards':          cards,
        })

    except Exception as e:
        import traceback
        return _json_error(f'Border detection failed: {e}')


@app.route('/api/border-generate', methods=['POST'])
def api_border_generate():
    """
    Generate BMP files for a border design using the high-detail border engine.

    Same JSON contract as /api/generate, plus:
        palette          : [[r,g,b], ...]  colours by index (from detect step)
        detail_retention : float 0-1       lower = preserve more fine detail
        hi_detail        : bool            use high-res detection (default true)

    Response shape matches /api/generate (zip_b64, files, verification,
    previews, bmp_b64) so the frontend can reuse the same rendering code.
    """
    try:
        if not request.is_json:
            return _json_error('Request must be JSON.')
        data = request.get_json(silent=True)
        if data is None:
            return _json_error('Invalid or empty JSON body.')

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

        # Prefer the full-resolution source for high-res detection; fall back
        # to the resized image if the frontend didn't send it.
        src_img = None
        if data.get('full_image'):
            try:
                src_img = Image.open(
                    io.BytesIO(base64.b64decode(data['full_image']))).convert('RGB')
            except Exception:
                src_img = None
        if src_img is None:
            try:
                src_img = Image.open(
                    io.BytesIO(base64.b64decode(data['image_b64']))).convert('RGB')
            except Exception:
                return _json_error('Could not decode image data.')

        # Sanitise design name (same rules as /api/generate)
        design_name = str(data.get('design_name', 'border')).strip() or 'border'
        design_name = ''.join(c for c in design_name if c.isalnum() or c in '_- ')
        design_name = design_name.replace(' ', '_') or 'border'

        try:
            color_assignments = {int(k): str(v)
                                 for k, v in data['color_assignments'].items()}
        except (ValueError, TypeError) as e:
            return _json_error(f'Invalid color_assignments: {e}')

        # Satin settings (same validation as /api/generate)
        raw_satin = data.get('satin_settings', {})
        satin_settings = {}
        valid_n = {4, 5, 6, 7, 8, 16}
        for k, v in raw_satin.items():
            try:
                n = int(v.get('n', 8))
            except (ValueError, TypeError):
                return _json_error(f'Satin n for "{k}" must be a whole number.')
            if n not in valid_n:
                return _json_error(f'Satin n for "{k}" must be one of {sorted(valid_n)}.')
            min_h = int(v.get('min_height', 9999))
            if min_h < 1:   min_h = 1
            if min_h > 99999: min_h = 99999
            pattern = str(v.get('pattern', 'satin')).lower().strip()
            if pattern not in FILL_PATTERNS:
                pattern = 'satin'
            satin_settings[str(k)] = {
                'n': n, 'flip': bool(v.get('flip', False)),
                'min_height': min_h, 'pattern': pattern,
                'weave_off': bool(v.get('weave_off', False)),
            }

        # Label map (target resolution) — optional fallback / shape reference
        label_map = None
        if data.get('label_map'):
            try:
                lm_img = Image.open(
                    io.BytesIO(base64.b64decode(data['label_map']))).convert('L')
                label_map = np.array(lm_img)
            except Exception:
                label_map = None

        # Palette (RGB per colour index) — makes hi-res assignment robust
        palette_rgb = data.get('palette') or None

        # Detail retention (pool threshold). Clamp to a sane range.
        try:
            detail_retention = float(data.get('detail_retention', 0.18))
        except (ValueError, TypeError):
            detail_retention = 0.18
        detail_retention = max(0.0, min(0.9, detail_retention))

        hi_detail = bool(data.get('hi_detail', True))
        auto_detail = bool(data.get('auto_detail', False))

        try:
            ink_sensitivity = float(data.get('ink_sensitivity', 1.0))
        except (ValueError, TypeError):
            ink_sensitivity = 1.0
        ink_sensitivity = max(0.25, min(3.0, ink_sensitivity))

        bmp_files = generate_border_bmps(
            image=src_img,
            pins=pins,
            cards=cards,
            shuttle_count=shuttle_count,
            color_assignments=color_assignments,
            satin_settings=satin_settings,
            design_name=design_name,
            label_map=label_map,
            palette_rgb=palette_rgb,
            detail_retention=detail_retention,
            ink_sensitivity=ink_sensitivity,
            hi_detail=hi_detail,
            auto_detail=auto_detail,
        )

        verification = {fname: verify_bmp(bdata)
                        for fname, bdata in bmp_files.items()}

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname, bdata in bmp_files.items():
                zf.writestr(fname, bdata)
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()

        previews = {}
        bmp_b64  = {}
        for fname, bdata in bmp_files.items():
            thumb = Image.open(io.BytesIO(bdata)).convert('RGB')
            thumb.thumbnail((300, 300), Image.NEAREST)
            buf = io.BytesIO()
            thumb.save(buf, format='PNG')
            previews[fname] = base64.b64encode(buf.getvalue()).decode()
            bmp_b64[fname]  = base64.b64encode(bdata).decode()

        return jsonify({
            'success':      True,
            'zip_b64':      zip_b64,
            'zip_filename': f'{design_name}_jacquard.zip',
            'files':        list(bmp_files.keys()),
            'verification': verification,
            'previews':     previews,
            'bmp_b64':      bmp_b64,
            'warnings':     _design_warnings(bmp_files, pins, cards),
        })

    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        import traceback
        return _json_error(f'Border generation failed: {e}')




# ─────────────────────────────────────────────────────────────────────────────
# BORDER IDENTIFICATION  (/border-id) — enhanced fine-detail generation
# Detection reuses /api/border-detect (unchanged).
# Generation uses border_id_engine: adaptive scale, dual-threshold pooling,
# pre-pool closing.
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/border-id')
def border_id_page():
    """Border ID is now combined into /border — redirect for backwards compatibility."""
    from flask import redirect
    return redirect('/border', code=301)


@app.route('/api/border-id-generate', methods=['POST'])
def api_border_id_generate():
    """
    Enhanced BMP generation for border identification.

    Accepts same JSON as /api/border-generate, plus:
      noise_min_size : int 1–6   minimum feature size at output resolution

    Response shape matches /api/border-generate exactly.
    """
    try:
        if not request.is_json:
            return _json_error('Request must be JSON.')
        data = request.get_json(silent=True)
        if data is None:
            return _json_error('Invalid or empty JSON body.')

        for field in ('image_b64', 'pins', 'cards', 'shuttle_count', 'color_assignments'):
            if field not in data:
                return _json_error(f'Missing required field: {field}')

        try:
            pins          = int(data['pins'])
            cards         = int(data['cards'])
            shuttle_count = int(data['shuttle_count'])
        except (ValueError, TypeError) as e:
            return _json_error(f'Invalid numeric field: {e}')

        if pins  < 10: return _json_error('Pins must be at least 10.')
        if cards < 10: return _json_error('Cards must be at least 10.')
        if shuttle_count not in (1, 2, 3, 4):
            return _json_error('Shuttle count must be 1, 2, 3, or 4.')

        src_img = None
        if data.get('full_image'):
            try:
                src_img = Image.open(
                    io.BytesIO(base64.b64decode(data['full_image']))).convert('RGB')
            except Exception:
                src_img = None
        if src_img is None:
            try:
                src_img = Image.open(
                    io.BytesIO(base64.b64decode(data['image_b64']))).convert('RGB')
            except Exception:
                return _json_error('Could not decode image data.')

        design_name = str(data.get('design_name', 'border_id')).strip() or 'border_id'
        design_name = ''.join(c for c in design_name if c.isalnum() or c in '_- ')
        design_name = design_name.replace(' ', '_') or 'border_id'

        try:
            color_assignments = {int(k): str(v)
                                 for k, v in data['color_assignments'].items()}
        except (ValueError, TypeError) as e:
            return _json_error(f'Invalid color_assignments: {e}')

        raw_satin      = data.get('satin_settings', {})
        satin_settings = {}
        for k, v in raw_satin.items():
            try:
                n = int(v.get('n', 8))
            except (ValueError, TypeError):
                return _json_error(f'Satin n for "{k}" must be a whole number.')
            if n not in FILL_PATTERNS and n not in {4,5,6,7,8,16}:
                n = 8
            min_h   = max(1, min(99999, int(v.get('min_height', 9999))))
            pattern = str(v.get('pattern', 'satin')).lower().strip()
            if pattern not in FILL_PATTERNS:
                pattern = 'satin'
            satin_settings[str(k)] = {
                'n': n, 'flip': bool(v.get('flip', False)),
                'min_height': min_h, 'pattern': pattern,
                'weave_off': bool(v.get('weave_off', False)),
            }

        label_map = None
        if data.get('label_map'):
            try:
                lm_img    = Image.open(
                    io.BytesIO(base64.b64decode(data['label_map']))).convert('L')
                label_map = np.array(lm_img)
            except Exception:
                label_map = None

        palette_rgb = data.get('palette') or None

        try:    detail_retention = max(0.0, min(0.9, float(data.get('detail_retention', 0.12))))
        except (ValueError, TypeError): detail_retention = 0.12

        try:    ink_sensitivity = max(0.25, min(3.0, float(data.get('ink_sensitivity', 1.0))))
        except (ValueError, TypeError): ink_sensitivity = 1.0

        try:    noise_min_size = max(1, min(8, int(data.get('noise_min_size', 1))))
        except (ValueError, TypeError): noise_min_size = 1

        hi_detail = bool(data.get('hi_detail', True))
        auto_detail = bool(data.get('auto_detail', False))

        bmp_files = generate_border_id_bmps(
            image=src_img, pins=pins, cards=cards,
            shuttle_count=shuttle_count,
            color_assignments=color_assignments,
            satin_settings=satin_settings,
            design_name=design_name,
            label_map=label_map, palette_rgb=palette_rgb,
            detail_retention=detail_retention,
            ink_sensitivity=ink_sensitivity,
            noise_min_size=noise_min_size,
            hi_detail=hi_detail,
            auto_detail=auto_detail,
        )

        verification = {f: verify_bmp(b) for f, b in bmp_files.items()}

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname, bdata in bmp_files.items():
                zf.writestr(fname, bdata)
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()

        previews = {}; bmp_b64 = {}
        for fname, bdata in bmp_files.items():
            thumb = Image.open(io.BytesIO(bdata)).convert('RGB')
            thumb.thumbnail((300, 300), Image.NEAREST)
            buf = io.BytesIO(); thumb.save(buf, format='PNG')
            previews[fname] = base64.b64encode(buf.getvalue()).decode()
            bmp_b64[fname]  = base64.b64encode(bdata).decode()

        return jsonify({
            'success': True, 'zip_b64': zip_b64,
            'zip_filename': f'{design_name}_jacquard.zip',
            'files': list(bmp_files.keys()),
            'verification': verification,
            'previews': previews, 'bmp_b64': bmp_b64,
            'warnings': _design_warnings(bmp_files, pins, cards),
        })

    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        import traceback
        return _json_error(f'Border identification failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# BORDER SUGGEST  — smart slider recommendations from image analysis
# Lightweight: no KMeans, no detection. Just reads the image and returns values.
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/border-suggest', methods=['POST'])
def api_border_suggest():
    """
    Analyze a border image and return suggested slider values.

    Form field: image (file)

    Response:
        pins            : int
        ink_sensitivity : float
        detail_retention: float
        noise_min_size  : int
        reasons         : {key: str}
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')
        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')

        try:
            img = Image.open(file.stream)
            img = ImageOps.exif_transpose(img).convert('RGB')
        except Exception:
            return _json_error('Could not read the uploaded image.')

        result = analyze_border_image(img)
        return jsonify({'success': True, **result})

    except Exception as e:
        return _json_error(f'Analysis failed: {e}')


# ══════════════════════════════════════════════════════════════════════════
# BUTTA STUDIO — reduce a dense motif to a small pin width (gap-preserving)
# ══════════════════════════════════════════════════════════════════════════
@app.route('/butta')
def butta_page():
    from flask import make_response
    resp = make_response(render_template('butta.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


def _open_upload(file):
    img = Image.open(file.stream)
    return ImageOps.exif_transpose(img).convert('RGB')


_SHUTTLE_NAMES = ('rani', 'meena4', 'meena3', 'meena2', 'meena1', 'zari')


def _shuttle_of(filename):
    """
    Identify a file's shuttle by its name SUFFIX ('{design}_{shuttle}.bmp'),
    not a loose substring — so a design named e.g. 'rani_floral' doesn't make
    every file look like the rani ground.
    """
    base = str(filename).lower().rsplit('.', 1)[0]
    tok = base.rsplit('_', 1)[-1]
    return tok if tok in _SHUTTLE_NAMES else None


def _design_warnings(bmp_files, pins, cards):
    """
    Loom warnings for a generated BMP set. Unions ink (black = thread up) across
    the motif shuttles — excluding the plain-weave 'rani' ground, whose regular
    weave would otherwise look like thousands of isolated single pixels.
    """
    try:
        from loom_utils import loom_warnings, count_long_floats
        MAX_FLOAT = 12
        mask = None
        float_count, longest = 0, 0
        for fn, by in bmp_files.items():
            if _shuttle_of(fn) == 'rani':
                continue
            a = np.array(Image.open(io.BytesIO(by)).convert('L')) < 128
            mask = a if mask is None else (mask | a if mask.shape == a.shape else mask)
            fc, fl = count_long_floats(a, MAX_FLOAT)
            float_count += fc
            longest = max(longest, fl)
        warns = loom_warnings(mask, pins, cards)
        if float_count:
            warns.append(
                f"{float_count} thread float{'s' if float_count != 1 else ''} longer than "
                f"{MAX_FLOAT} (longest {longest}) — long floats can snag or sag; "
                f"raise satin binding or pin count.")
        return warns
    except Exception:
        return []


def _composite_render(bmp_files, pins, cards):
    """
    Woven-fabric simulation: layer the actual generated shuttle BMPs in their
    thread colours onto a ground, so the user sees what the cloth will look
    like instead of separate 1-bit files. black = thread up for that shuttle.
    Colours match the UI's SHUTTLE_COLORS palette. Returns base64 PNG or None.
    """
    try:
        GROUND = (74, 64, 48)
        PAL = {
            'rani':   (200, 120, 144),
            'meena4': (150, 110, 170),
            'meena3': (120, 180, 130),
            'meena2': (184, 144, 200),
            'meena1': (138, 180, 216),
            'zari':   (200, 160, 64),
        }
        canvas = np.full((cards, pins, 3), GROUND, dtype=np.uint8)
        # bottom-to-top: ground weave first, motif shuttles over it, zari on top
        for key in ['rani', 'meena4', 'meena3', 'meena2', 'meena1', 'zari']:
            col = PAL[key]
            for fn, by in bmp_files.items():
                if _shuttle_of(fn) == key:
                    a = np.array(Image.open(io.BytesIO(by)).convert('L'))
                    if a.shape != (cards, pins):
                        continue
                    canvas[a < 128] = col
        img = Image.fromarray(canvas, 'RGB')
        scale = max(1, min(6, 720 // max(1, pins)))
        img = img.resize((pins * scale, cards * scale), Image.NEAREST)
        buf = io.BytesIO(); img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


@app.route('/api/butta-preview', methods=['POST'])
def api_butta_preview():
    """Fast preview of the gap-preserving reduction for the live slider."""
    try:
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')
        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')
        try:
            img = _open_upload(file)
        except Exception:
            return _json_error('Could not read the uploaded image.')

        try:
            target_pins = int(request.form.get('pins', 200))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        target_pins = max(40, min(600, target_pins))

        try:
            detail = float(request.form.get('detail', 0.0))
        except (ValueError, TypeError):
            detail = 0.0
        detail = max(-1.0, min(1.0, detail))

        autocrop = request.form.get('autocrop', 'true').lower() == 'true'

        cards_raw = (request.form.get('cards', '') or '').strip()
        try:
            target_cards = max(8, min(2000, int(cards_raw))) if cards_raw else None
        except (ValueError, TypeError):
            target_cards = None

        try:
            n_colors = max(1, min(6, int(request.form.get('colors', 1))))
        except (ValueError, TypeError):
            n_colors = 1
        thin_rescue = request.form.get('thin_rescue', 'false').lower() == 'true'

        if n_colors > 1:
            label_map, colors, _assign, info = butta_engine.reduce_butta_multi(
                img, target_pins, target_cards=target_cards, n_colors=n_colors,
                detail=detail, autocrop=autocrop)
            scale = max(1, min(6, 900 // max(1, info['target_w'])))
            prev = butta_engine.labelmap_to_preview_png(label_map, colors, scale=scale)
        else:
            mask, info = butta_engine.reduce_butta(
                img, target_pins, target_cards=target_cards, detail=detail,
                autocrop=autocrop, thin_rescue=thin_rescue)
            scale = max(1, min(6, 900 // max(1, info['target_w'])))
            prev = butta_engine.mask_to_preview_png(mask, scale=scale)

        buf = io.BytesIO(); prev.save(buf, format='PNG')
        return jsonify({
            'success': True,
            'preview_b64': base64.b64encode(buf.getvalue()).decode(),
            'info': info,
        })
    except Exception as e:
        return _json_error(f'Preview failed: {e}')


@app.route('/api/butta-generate', methods=['POST'])
def api_butta_generate():
    """
    Final output. output_mode:
      'quick' -> single clean 1-bit BMP of the motif.
      'full'  -> hand the reduced label_map to generate_bmps (zari + rani).
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')
        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')
        try:
            img = _open_upload(file)
        except Exception:
            return _json_error('Could not read the uploaded image.')

        try:
            target_pins = max(40, min(600, int(request.form.get('pins', 200))))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        try:
            detail = max(-1.0, min(1.0, float(request.form.get('detail', 0.0))))
        except (ValueError, TypeError):
            detail = 0.0
        autocrop = request.form.get('autocrop', 'true').lower() == 'true'
        output_mode = request.form.get('output_mode', 'quick').lower()
        design_name = (request.form.get('design_name', '') or '').strip() or 'butta'

        cards_raw = (request.form.get('cards', '') or '').strip()
        try:
            target_cards = max(8, min(2000, int(cards_raw))) if cards_raw else None
        except (ValueError, TypeError):
            target_cards = None

        try:
            n_colors = max(1, min(6, int(request.form.get('colors', 1))))
        except (ValueError, TypeError):
            n_colors = 1
        thin_rescue = request.form.get('thin_rescue', 'false').lower() == 'true'

        files = []
        if n_colors > 1:
            # Colour butta -> always a full multi-shuttle set.
            label_map, colors, assignments, info = butta_engine.reduce_butta_multi(
                img, target_pins, target_cards=target_cards, n_colors=n_colors,
                detail=detail, autocrop=autocrop)
            satin = {assignments[i]: {'n': 8, 'flip': False, 'min_height': 35,
                                      'pattern': 'satin', 'weave_off': True}
                     for i in assignments if i != 0}
            results = generate_bmps(
                image=img, pins=info['target_w'], cards=info['target_h'],
                shuttle_count=len(colors), color_assignments=assignments,
                satin_settings=satin, design_name=design_name,
                label_map=label_map, rani_weave='plain',
                stroke_mode=False, supersample=False)
            for fn, by in results.items():
                files.append({'filename': fn, 'bmp_b64': base64.b64encode(by).decode()})
            return jsonify({'success': True, 'files': files, 'info': info})

        mask, info = butta_engine.reduce_butta(
            img, target_pins, target_cards=target_cards, detail=detail,
            autocrop=autocrop, thin_rescue=thin_rescue)

        if output_mode == 'full':
            label_map, colors, assignments = butta_engine.mask_to_label_map(mask)
            satin = {'zari': {'n': 8, 'flip': False, 'min_height': 35,
                              'pattern': 'satin', 'weave_off': True}}
            results = generate_bmps(
                image=img, pins=info['target_w'], cards=info['target_h'],
                shuttle_count=2, color_assignments=assignments,
                satin_settings=satin, design_name=design_name,
                label_map=label_map, outline_white={'zari': True},
                rani_weave='plain', stroke_mode=False, supersample=False)
            for fn, by in results.items():
                files.append({'filename': fn,
                              'bmp_b64': base64.b64encode(by).decode()})
        else:
            by = butta_engine.mask_to_bmp_bytes(mask)
            files.append({'filename': f'{design_name}.bmp',
                          'bmp_b64': base64.b64encode(by).decode()})

        return jsonify({'success': True, 'files': files, 'info': info})
    except Exception as e:
        return _json_error(f'Generation failed: {e}')


@app.route('/api/butta-repeat-generate', methods=['POST'])
def api_butta_repeat_generate():
    """
    Tile the reduced motif into a single 1-bit BMP at THREAD resolution so the
    whole step-and-repeat can be opened in the BMP editor. Mono only — the editor
    edits one 1-bit surface. Mirrors butta-generate's reduce params, plus:
        across, down : tile counts
        layout       : straight | half | brick
        gap          : threads between tiles
    Returns {success, file:{filename, bmp_b64}, info}.
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image file uploaded.')
        file = request.files['image']
        if not file.filename:
            return _json_error('No file selected.')
        try:
            img = _open_upload(file)
        except Exception:
            return _json_error('Could not read the uploaded image.')

        try:
            target_pins = max(40, min(600, int(request.form.get('pins', 200))))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        try:
            detail = max(-1.0, min(1.0, float(request.form.get('detail', 0.0))))
        except (ValueError, TypeError):
            detail = 0.0
        autocrop = request.form.get('autocrop', 'true').lower() == 'true'
        thin_rescue = request.form.get('thin_rescue', 'false').lower() == 'true'
        design_name = (request.form.get('design_name', '') or '').strip() or 'butta'
        cards_raw = (request.form.get('cards', '') or '').strip()
        try:
            target_cards = max(8, min(2000, int(cards_raw))) if cards_raw else None
        except (ValueError, TypeError):
            target_cards = None

        # Repeat params (same clamps as the UI controls).
        try:
            across = max(1, min(10, int(request.form.get('across', 3))))
            down   = max(1, min(10, int(request.form.get('down', 2))))
            gap    = max(0, min(200, int(request.form.get('gap', 0))))
        except (ValueError, TypeError):
            return _json_error('Repeat across/down/gap must be whole numbers.')
        layout = request.form.get('layout', 'straight').lower()
        if layout not in ('straight', 'half', 'brick'):
            layout = 'straight'

        mask, info = butta_engine.reduce_butta(
            img, target_pins, target_cards=target_cards, detail=detail,
            autocrop=autocrop, thin_rescue=thin_rescue)
        h, w = mask.shape

        step_x = w + gap
        step_y = h + gap
        cw = across * w + (across - 1) * gap
        ch = down * h + (down - 1) * gap

        # Guard against an editor canvas too large to be usable.
        if cw * ch > 16_000_000 or cw > 6000 or ch > 6000:
            return _json_error('Repeat is too large for the editor — reduce tiles, gap, or pins.')

        big = np.zeros((ch, cw), dtype=bool)

        def _place(x, y):
            # OR the motif mask onto the canvas at (x, y), clipping partial tiles
            # at the edges so offset (brick/half-drop) rows wrap seamlessly.
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(cw, x + w), min(ch, y + h)
            if x1 <= x0 or y1 <= y0:
                return
            big[y0:y1, x0:x1] |= mask[y0 - y:y1 - y, x0 - x:x1 - x]

        if layout == 'brick':
            for r in range(down):
                off = (step_x // 2) if (r % 2) else 0
                for c in range(-1, across + 1):
                    _place(off + c * step_x, r * step_y)
        elif layout == 'half':
            for c in range(across):
                off = (step_y // 2) if (c % 2) else 0
                for r in range(-1, down + 1):
                    _place(c * step_x, off + r * step_y)
        else:
            for r in range(down):
                for c in range(across):
                    _place(c * step_x, r * step_y)

        by = butta_engine.mask_to_bmp_bytes(big)
        return jsonify({
            'success': True,
            'file': {'filename': f'{design_name}_repeat_{across}x{down}.bmp',
                     'bmp_b64': base64.b64encode(by).decode()},
            'info': {'tiles': f'{across}×{down}', 'layout': layout,
                     'width': cw, 'height': ch, 'motif': f'{w}×{h}'},
        })
    except Exception as e:
        return _json_error(f'Repeat build failed: {e}')


@app.route('/api/butta-batch', methods=['POST'])
def api_butta_batch():
    """
    Batch-reduce several butta images with one shared set of settings and return
    a single ZIP. Form fields: images[] (multiple), pins, detail, colors,
    thin_rescue, autocrop, output_mode, plus optional cards.
    """
    try:
        files = request.files.getlist('images')
        files = [f for f in files if f and f.filename]
        if not files:
            return _json_error('No images uploaded.')

        try:
            target_pins = max(40, min(600, int(request.form.get('pins', 200))))
        except (ValueError, TypeError):
            return _json_error('Pins must be a whole number.')
        try:
            detail = max(-1.0, min(1.0, float(request.form.get('detail', 0.0))))
        except (ValueError, TypeError):
            detail = 0.0
        autocrop = request.form.get('autocrop', 'true').lower() == 'true'
        thin_rescue = request.form.get('thin_rescue', 'false').lower() == 'true'
        output_mode = request.form.get('output_mode', 'quick').lower()
        try:
            n_colors = max(1, min(6, int(request.form.get('colors', 1))))
        except (ValueError, TypeError):
            n_colors = 1
        cards_raw = (request.form.get('cards', '') or '').strip()
        try:
            target_cards = max(8, min(2000, int(cards_raw))) if cards_raw else None
        except (ValueError, TypeError):
            target_cards = None

        results, errors = {}, []
        for f in files:
            base = os.path.splitext(os.path.basename(f.filename))[0] or 'butta'
            base = base.strip() or 'butta'
            try:
                img = _open_upload(f)
            except Exception:
                errors.append(f"{f.filename}: could not read image")
                continue
            try:
                if n_colors > 1:
                    label_map, colors, assignments, info = butta_engine.reduce_butta_multi(
                        img, target_pins, target_cards=target_cards,
                        n_colors=n_colors, detail=detail, autocrop=autocrop)
                    satin = {assignments[i]: {'n': 8, 'flip': False, 'min_height': 35,
                                              'pattern': 'satin', 'weave_off': True}
                             for i in assignments if i != 0}
                    out = generate_bmps(
                        image=img, pins=info['target_w'], cards=info['target_h'],
                        shuttle_count=len(colors), color_assignments=assignments,
                        satin_settings=satin, design_name=base, label_map=label_map,
                        rani_weave='plain', stroke_mode=False, supersample=False)
                    for fn, by in out.items():
                        results[fn] = by
                elif output_mode == 'full':
                    mask, info = butta_engine.reduce_butta(
                        img, target_pins, target_cards=target_cards, detail=detail,
                        autocrop=autocrop, thin_rescue=thin_rescue)
                    label_map, _c, assignments = butta_engine.mask_to_label_map(mask)
                    satin = {'zari': {'n': 8, 'flip': False, 'min_height': 35,
                                      'pattern': 'satin', 'weave_off': True}}
                    out = generate_bmps(
                        image=img, pins=info['target_w'], cards=info['target_h'],
                        shuttle_count=2, color_assignments=assignments,
                        satin_settings=satin, design_name=base, label_map=label_map,
                        outline_white={'zari': True}, rani_weave='plain',
                        stroke_mode=False, supersample=False)
                    for fn, by in out.items():
                        results[fn] = by
                else:
                    mask, info = butta_engine.reduce_butta(
                        img, target_pins, target_cards=target_cards, detail=detail,
                        autocrop=autocrop, thin_rescue=thin_rescue)
                    results[f'{base}.bmp'] = butta_engine.mask_to_bmp_bytes(mask)
            except Exception as e:
                errors.append(f"{f.filename}: {e}")

        if not results:
            return _json_error('No files could be processed. ' + '; '.join(errors[:3]))

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn, by in results.items():
                zf.writestr(fn, by)
        zip_b64 = base64.b64encode(zip_buf.getvalue()).decode()
        return jsonify({
            'success': True,
            'zip_b64': zip_b64,
            'count': len(files),
            'file_count': len(results),
            'errors': errors,
        })
    except Exception as e:
        return _json_error(f'Batch failed: {e}')


@app.route('/api/generate-preview', methods=['POST'])
def api_generate_preview():
    """
    Fast 'loom grid' preview for the generator: shows the design downscaled to
    the loom thread resolution (pins x cards), so the user can see how detail
    holds up before running the full Detect/Generate pipeline. Lightweight — a
    LANCZOS downscale + nearest-neighbour upscale for visible threads.
    """
    try:
        data = request.get_json(silent=True) or {}
        b64 = data.get('image_b64')
        if not b64:
            return _json_error('No image provided.')
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            img = ImageOps.exif_transpose(img).convert('RGB')
        except Exception:
            return _json_error('Could not read the image.')
        try:
            pins = max(10, min(2000, int(data.get('pins', 240))))
        except (ValueError, TypeError):
            pins = 240
        cards_raw = data.get('cards')
        try:
            cards = max(10, min(2000, int(cards_raw))) if cards_raw else \
                max(1, round(img.height * pins / max(1, img.width)))
        except (ValueError, TypeError):
            cards = max(1, round(img.height * pins / max(1, img.width)))

        small = img.resize((pins, cards), Image.LANCZOS)
        scale = max(1, min(6, 720 // max(1, pins)))
        prev = small.resize((pins * scale, cards * scale), Image.NEAREST)
        buf = io.BytesIO(); prev.save(buf, format='PNG')
        return jsonify({
            'success': True,
            'preview_b64': base64.b64encode(buf.getvalue()).decode(),
            'pins': pins, 'cards': cards,
        })
    except Exception as e:
        return _json_error(f'Preview failed: {e}')


if __name__ == '__main__':
    # Prevent joblib/OpenMP from spawning parallel workers.
    # Required on macOS (avoids 10-30s KMeans hang) and Windows alike.
    import os as _os
    _os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
    _os.environ.setdefault('OMP_NUM_THREADS',    '1')
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)
