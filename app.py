"""
Jacquard Designer App — Flask Backend
"""

from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import numpy as np
import io, os, zipfile, base64
from bmp_engine import detect_colors, generate_bmps, verify_bmp, enhance_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB upload cap

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _json_error(msg: str, status: int = 400):
    """Return a JSON error response (never HTML)."""
    return jsonify({'success': False, 'error': msg}), status


@app.errorhandler(413)
def too_large(_e):
    """Override Flask's default HTML 413 page with JSON so the frontend can parse it."""
    return _json_error('File too large. Maximum upload size is 50 MB.', 413)


@app.route('/')
def index():
    return render_template('index.html')


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
                f'Please upload a JPEG, PNG, BMP, TIFF, or WebP image.'
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
            img = Image.open(file.stream).convert('RGB')
        except UnidentifiedImageError:
            return _json_error(
                'Could not read the uploaded file as an image. '
                'Please check the file is not corrupted.'
            )

        orig_w, orig_h = img.size
        if cards is None:
            cards = max(10, int(pins * orig_h / orig_w))

        # ── Optional image enhancement ───────────────────────────────────────
        if request.form.get('enhance', 'false').lower() == 'true':
            img = enhance_image(img)

        # ── Detect colours ───────────────────────────────────────────────────
        resized = img.resize((pins, cards), Image.LANCZOS)
        colors, counts, label_map = detect_colors(resized, n_colors)

        total_pixels = pins * cards
        color_data = [
            {
                'index':      i,
                'rgb':        [int(x) for x in color],
                'hex':        '#{:02x}{:02x}{:02x}'.format(*[int(x) for x in color]),
                'percentage': round(100 * count / total_pixels, 1),
                'count':      count,
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

        return jsonify({
            'success':        True,
            'colors':         color_data,
            'preview_image':  _to_b64(preview_img),
            'original_image': _to_b64(resized),
            'label_map':      _to_b64(label_img),
            'pins':           pins,
            'cards':          cards,
        })

    except Exception as e:
        import traceback
        return _json_error(f'Unexpected error: {e}'), 500


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
            satin_settings[str(k)] = {'n': n, 'flip': bool(v.get('flip', False))}

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
        bmp_files = generate_bmps(
            image=img,
            pins=pins,
            cards=cards,
            shuttle_count=shuttle_count,
            color_assignments=color_assignments,
            satin_settings=satin_settings,
            design_name=design_name,
            label_map=label_map,
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

        # ── Thumbnail previews ────────────────────────────────────────────────
        previews = {}
        for fname, bdata in bmp_files.items():
            thumb = Image.open(io.BytesIO(bdata)).convert('RGB')
            thumb.thumbnail((300, 300), Image.NEAREST)
            buf = io.BytesIO()
            thumb.save(buf, format='PNG')
            previews[fname] = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'success':      True,
            'zip_b64':      zip_b64,
            'zip_filename': f'{design_name}_jacquard.zip',
            'files':        list(bmp_files.keys()),
            'verification': verification,
            'previews':     previews,
        })

    except ValueError as e:
        # Raised by generate_bmps for label_map shape mismatch
        return _json_error(str(e))
    except Exception as e:
        import traceback
        return _json_error(f'Generation failed: {e}'), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)
