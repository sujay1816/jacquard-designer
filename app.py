"""
Jacquard Designer App — Flask Backend
"""

from flask import (Flask, request, jsonify, render_template, send_file,
                   Response, stream_with_context)
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import io, os, re, zipfile, base64
from bmp_engine import (detect_colors, generate_bmps, verify_bmp, enhance_image,
                        assess_image_quality,
                        generate_fill_pattern, FILL_PATTERNS, write_1bit_bmp)
from border_engine import generate_border_bmps, detect_border
from border_id_engine import generate_border_id_bmps
from enhanced_engine import preprocess_fabric_image, analyze_border_image
from vision_engine import detect_colors_smart
import assistant_engine
import butta_engine

app = Flask(__name__)
app.secret_key = os.environ.get('JQ_SECRET_KEY', 'jq-designer-2024')
_bmp_store = {}  # token → {bmp_b64, filename, preview}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB upload cap

# Build identifier derived from the actual content of the templates and the
# core modules.
#
# This was previously a hand-written string, and it went stale immediately: it
# still read "2026.08.04-nav-unified" three navigation changes later, so the
# one signal meant to distinguish a fresh deployment from an old one reported
# the same value for both. A hash cannot be forgotten — change any file below
# and the build id changes with it.
def _compute_build():
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    targets = []
    tdir = os.path.join(here, 'templates')
    if os.path.isdir(tdir):
        targets += [os.path.join(tdir, f) for f in sorted(os.listdir(tdir))
                    if f.endswith('.html')]
    targets += [os.path.join(here, f) for f in (
        'app.py', 'bmp_engine.py', 'vision_engine.py', 'butta_engine.py',
        'fidelity.py', 'auto_convert.py', 'agent_engine.py', 'loom_utils.py')]
    for path in targets:
        try:
            with open(path, 'rb') as fh:
                h.update(os.path.basename(path).encode())
                h.update(fh.read())
        except OSError:
            h.update(b'missing:' + os.path.basename(path).encode())
    return h.hexdigest()[:12]


NAV_BUILD = _compute_build()

# Pages the app is expected to serve. Used by the startup self-check and by
# /api/build, so a partial file copy shows up immediately instead of as a
# confusing 500 later.
EXPECTED_TEMPLATES = ('_nav.html', '404.html', 'index.html', 'edit.html',
                      'trace.html', 'border.html', 'butta.html', 'agent.html')


def missing_templates():
    """Template files that should be present but are not."""
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    return [t for t in EXPECTED_TEMPLATES
            if not os.path.exists(os.path.join(tdir, t))]


@app.context_processor
def _inject_nav_build():
    return {'nav_build': NAV_BUILD}


@app.route('/api/build', methods=['GET'])
def api_build():
    """
    Report the running build so a stale or partial copy is easy to identify.

    nav_build is a hash of the templates and core modules, so it changes
    whenever any of them do.
    """
    missing = missing_templates()
    return jsonify({
        'success': True,
        'nav_build': NAV_BUILD,
        'pages': ['/', '/butta', '/border', '/edit', '/agent', '/trace'],
        'missing_templates': missing,
        'complete': not missing,
    })


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif'}


MAX_PINS  = 2640          # ends across the warp — matches loom_utils
MAX_CARDS = 6000          # picks / cards
MAX_CELLS = 6_000_000     # pins*cards ceiling: guards against OOM


def _bounds_error(pins, cards=None):
    """
    Reject dimensions that would exhaust memory. Returns an error response, or
    None when the values are usable.

    Only a lower bound was previously enforced, so pins=50000 x cards=50000
    (2.5 billion cells) allocated until the OS killed the process, taking the
    user's whole session with it. A local single-process server has no
    supervisor to restart it, so a typo with one extra zero was unrecoverable.
    """
    if pins > MAX_PINS:
        return _json_error(
            f'{pins} pins exceeds the maximum of {MAX_PINS}.')
    if cards is not None:
        if cards > MAX_CARDS:
            return _json_error(
                f'{cards} cards exceeds the maximum of {MAX_CARDS}.')
        if pins * cards > MAX_CELLS:
            return _json_error(
                f'{pins} x {cards} is {pins * cards:,} cells, above the '
                f'{MAX_CELLS:,} limit. Reduce pins or cards.')
    return None


_MODULE_ERR = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _json_error(msg: str, status: int = 400):
    """Return a JSON error response (never HTML)."""
    return jsonify({'success': False, 'error': _dep_message(msg)}), status


def _dep_message(msg):
    """Rewrite a missing-module error as an install instruction."""
    hit = _MODULE_ERR.search(str(msg))
    if not hit:
        return str(msg)
    mod = hit.group(1).split('.')[0]
    try:
        import deps
        entry = next((r for r in deps.REQUIRED + deps.OPTIONAL if r[0] == mod), None)
    except Exception:
        entry = None
    pip_name, purpose = (entry[1], entry[2]) if entry else (mod, 'this feature')
    prefix = str(msg).split('No module named')[0].strip().rstrip(':').strip()
    lead = f'{prefix}: ' if prefix else ''
    return (f'{lead}a required package is not installed. '
            f'{mod} is needed for {purpose}. '
            f'Run:  pip install "{pip_name}"  '
            f'(or: pip install -r requirements.txt), then restart the app.')


@app.errorhandler(ImportError)
def _import_error(e):
    """Catch a missing package that no route wrapped in a try block."""
    return jsonify({'success': False, 'error': _dep_message(e)}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """
    Whether this install is actually complete.

    Pages call it on load so a missing package is reported before the weaver
    does ten minutes of work and loses it at the last step.
    """
    try:
        import deps
        result = deps.check()
        return jsonify({'success': True, 'ok': result['ok'],
                        'missing': result['missing'],
                        'missing_optional': result['missing_optional'],
                        'install_command': (deps.install_command(result['missing'])
                                            if result['missing'] else None),
                        'templates_missing': missing_templates()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(_e):
    """
    Friendly 404. API paths get JSON so the frontend can parse the failure;
    page paths get the app shell with navigation, so a stale bookmark (notably
    the retired /border-id) is a signpost rather than a dead end.
    """
    if request.path.startswith('/api/'):
        return _json_error(f'No such endpoint: {request.path}', 404)
    return render_template('404.html', requested_path=request.path), 404


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
        _e = _bounds_error(pins)
        if _e: return _e

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
                _e = _bounds_error(pins, cards)
                if _e: return _e
            except ValueError:
                return _json_error('Cards must be a whole number.')

        # ── Open image ───────────────────────────────────────────────────────
        try:
            img = Image.open(file.stream)
            img = _open_upload(img)
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
        # Smart mode clusters at source resolution and pools the LABEL MAP down
        # by area coverage. The legacy path resizes the photo first, which
        # destroys thin features before clustering ever sees them.
        smart = request.form.get('smart_detect', 'false').lower() == 'true'
        # 'resized' is the loom-resolution preview returned to the frontend as
        # original_image, so it must exist on BOTH paths. Smart mode does not
        # need it for detection (that is the point — it clusters at source
        # resolution) but the response still does.
        resized = img.resize((pins, cards), Image.LANCZOS)
        if smart:
            colors, counts, label_map, genuine_flags = detect_colors_smart(
                img, n_colors, pins, cards)
        else:
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
        return _json_error(f'Unexpected error: {e}')


@app.route('/api/check-trace', methods=['POST'])
def api_check_trace():
    """
    Score a hand-traced design against the original it was traced from.

    Closes the loop for beginners: the Tracing Guide explains how to trace, but
    until now nothing told them whether the result would actually weave. The
    common failure is invisible on screen — strokes drawn a little heavy look
    bolder, and only reveal themselves at the loom when every gap has closed
    and the motif comes out solid.

    Takes both files so no server-side state is needed; the browser still holds
    the original from the analyse step.
    """
    try:
        if 'original' not in request.files or 'traced' not in request.files:
            return _json_error('Upload both the original image and your traced file.')

        def _load(key):
            raw = request.files[key].read()
            if not raw:
                raise ValueError(f'{key} file is empty.')
            if len(raw) > 50 * 1024 * 1024:
                raise ValueError(f'{key} file is too large (max 50 MB).')
            return ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))

        original = _load('original').convert('RGB')
        traced = _load('traced')

        from fidelity import trace_feedback
        report = trace_feedback(original, traced)
        return jsonify({'success': True, 'report': report})
    except UnidentifiedImageError:
        return _json_error('One of the files is not a readable image.')
    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        return _json_error(f'Could not check the trace: {e}')


@app.route('/api/assistant-status', methods=['GET'])
def api_assistant_status():
    """
    Report whether a model backend is configured, so the UI can hide the panel.

    Asks the provider layer, not for an Anthropic key specifically: a mill
    running a local Llama has no key at all, and gating on one hid a working
    assistant behind a notice telling them to go and get one.
    """
    try:
        import llm
        return jsonify({'success': True, 'available': llm.is_available(),
                        'backend': llm.describe(),
                        'vision': bool(llm.provider().supports_vision)})
    except Exception:
        return jsonify({'success': True, 'available': False,
                        'backend': 'unavailable', 'vision': False})


def _label_map_b64(label_map):
    """
    Encode a label map as a base64 PNG for the frontend to hand back at
    generate time. Values are class indices, not brightness, so PNG (lossless)
    is required — a lossy format would silently reassign pixels to the wrong
    shuttle.
    """
    buf = io.BytesIO()
    Image.fromarray(np.asarray(label_map).astype(np.uint8), 'L').save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


@app.route('/agent')
def agent_page():
    """Conversational conversion. Degrades to an explanatory notice with no key."""
    from flask import make_response
    resp = make_response(render_template('agent.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp



def _open_upload(src):
    """
    Open an uploaded image as RGB, flattening transparency onto white.

    Accepts raw bytes, a Werkzeug upload, or an already-open Image, because the
    upload routes differ in what they have to hand and two separate loaders is
    how one of them ends up without the fixes the other got.

    Transparent PNGs are the normal export of every design tool, and PIL's
    convert('RGB') on one DISCARDS the alpha and keeps whatever RGB sat beneath
    it — (0, 0, 0) in almost every exporter. The whole canvas then goes black,
    the motif disappears into it, and the conversion returns a blank BMP while
    reporting a mere 'warn'. Transparent means no ink, which means bare cloth,
    which means white.

    EXIF rotation is applied too, so a design photographed in portrait is not
    converted sideways.
    """
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src.stream)
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        img = Image.alpha_composite(Image.new('RGBA', img.size, (255, 255, 255, 255)), img)
    return img.convert('RGB')


@app.route('/api/agent/start', methods=['POST'])
def api_agent_start():
    """Upload a design and open a conversation about converting it."""
    try:
        if not agent_engine_available():
            return _json_error(
                'No model backend configured. Set ANTHROPIC_API_KEY, or set '
                '"llm_provider" in config.json to use a local model.')
        if 'image' not in request.files:
            return _json_error('No image uploaded.')
        f = request.files['image']
        raw = f.read()
        if not raw:
            return _json_error('Uploaded file is empty.')
        if len(raw) > 50 * 1024 * 1024:
            return _json_error('Image is too large (max 50 MB).')
        img = _open_upload(raw)

        import agent_engine
        token = agent_engine.new_session(img, f.filename or 'design')
        session = agent_engine.get_session(token)
        opening = str(request.form.get('opening', '')).strip()[:400] or (
            'I have uploaded a design. Please look at it and tell me what you '
            'see, then ask what I need.')
        result = agent_engine.converse(session, opening)
        return jsonify({'success': result['ok'], 'token': token,
                        'reply': result['reply'],
                        'tools_used': result['tools_used'],
                        'has_files': result['has_files']})
    except UnidentifiedImageError:
        return _json_error('That file is not a readable image.')
    except Exception as e:
        return _json_error(f'Could not start: {e}')


@app.route('/api/agent/message', methods=['POST'])
def api_agent_message():
    """Continue a conversation."""
    try:
        data = request.get_json() or {}
        message = str(data.get('message', '')).strip()
        if not message:
            return _json_error('No message provided.')
        if len(message) > 2000:
            return _json_error('Message too long.')

        import agent_engine
        session = agent_engine.get_session(str(data.get('token', '')))
        if not session:
            return _json_error('That conversation has expired. Upload the design again.')

        result = agent_engine.converse(session, message)
        return jsonify({'success': result['ok'], 'reply': result['reply'],
                        'tools_used': result['tools_used'],
                        'has_files': result['has_files']})
    except Exception as e:
        return _json_error(f'Assistant failed: {e}')


@app.route('/api/agent/blank', methods=['POST'])
def api_agent_blank():
    """
    Open a conversation with nothing uploaded, to design from scratch.

    Previously the page faked this by posting an 8x8 white PNG, which meant a
    generate-from-scratch job masqueraded as a conversion and the weaver was
    shown a white square as their "design". A design session has no source
    image, and saying so plainly is what lets the UI show the generated panel
    instead.
    """
    try:
        if not agent_engine_available():
            return _json_error('No model backend configured.')
        import agent_engine
        token = agent_engine.new_session(None, 'new-design')
        return jsonify({'success': True, 'token': token})
    except Exception as e:
        return _json_error(f'Could not start: {e}')


@app.route('/api/agent/state', methods=['GET'])
def api_agent_state():
    """
    Everything the design panel needs, in one call.

    The page could show the design but nothing about it — the verdict, the
    drift, the finished size, what was saved, which files exist all lived only
    inside the agent's prose. A weaver reading "it is a warn" in a sentence
    cannot glance at it later, and cannot see it change as they work.
    """
    try:
        import agent_engine
        import canvas_ops as co

        session = agent_engine.get_session(request.args.get('token', ''))
        if not session:
            return _json_error('That conversation has expired.')

        # Use the same accessor the tools use, not the raw key: `working` is
        # materialised lazily from the conversion, so reading the key directly
        # showed no canvas at all until some tool happened to touch it first.
        lm = agent_engine._working(session)
        conv = session.get('conversion') or {}
        rep = ((conv.get('best') or {}).get('report')) or {}
        spec = session.get('spec') or {}
        reed = agent_engine._reed_of(session)

        out = {'success': True,
               'has_design': lm is not None or session.get('image') is not None,
               'has_source': bool(session.get('source_is_upload')),
               'verdict': str(conv.get('verdict', '')).lower() or None,
               'thread_drift_pct': rep.get('ink_drift_pct'),
               'design_gaps': rep.get('output_white_regions'),
               'reed': reed,
               'can_undo': bool(session.get('undo')),
               'checkpoints': [{'name': k, 'design': v['summary'],
                                'pins': v['pins'], 'cards': v['cards']}
                               for k, v in (session.get('checkpoints') or {}).items()],
               'files': [{'name': n, 'bytes': len(d)}
                         for n, d in (session.get('files') or {}).items()],
               'plan': session.get('plan'),
               'reference_rebased': bool(session.get('reference_rebased')),
               }

        if lm is not None:
            import numpy as np
            arr = np.asarray(lm)
            out['canvas'] = co.stats(arr)
            out['width_in'] = round(arr.shape[1] / reed, 2)
            out['length_in'] = round(arr.shape[0] / reed, 2)
        if spec:
            out['description'] = spec.get('body_motif')
        return jsonify(out)
    except Exception as e:
        return _json_error(f'Could not read the session: {e}')


@app.route('/api/agent/file', methods=['GET'])
def api_agent_file():
    """
    One generated BMP on its own.

    A zip is the right thing to hand a loom operator, but it is the wrong thing
    for checking a single shuttle before committing — and looking at one file
    should not mean downloading and unpacking all of them.
    """
    try:
        import agent_engine
        session = agent_engine.get_session(request.args.get('token', ''))
        if not session:
            return _json_error('That conversation has expired.')
        name = request.args.get('name', '')
        data = (session.get('files') or {}).get(name)
        if not data:
            return _json_error('No such file in this conversation.')
        return send_file(io.BytesIO(data), mimetype='image/bmp',
                         as_attachment=True, download_name=name)
    except Exception as e:
        return _json_error(f'Download failed: {e}')


def agent_engine_unpack(blob):
    """Undo entries are stored packed; the preview needs them as arrays."""
    import agent_engine
    return agent_engine._unpack(blob)


@app.route('/api/agent/preview', methods=['GET'])
def api_agent_preview():
    """
    The design as it currently stands, as a PNG.

    The single largest gap in the assistant page: it is a design tool in which
    the weaver could not see the design. The agent would build a panel, score
    it, refine it twice, and the weaver read prose about all of it.
    """
    try:
        import agent_engine
        session = agent_engine.get_session(request.args.get('token', ''))
        if not session:
            return _json_error('That conversation has expired.')
        import numpy as np
        which = request.args.get('which', 'design')
        img = session.get('image')
        working = session.get('working')

        if which == 'source':
            # The artwork as uploaded, for comparing against the conversion.
            if img is None:
                return _json_error('There is no source image in this conversation.')
        elif which == 'previous':
            # The state before the last edit, so a change can be seen rather
            # than only described. Rendered from the undo stack, which holds
            # packed label maps.
            undo = session.get('undo') or []
            if not undo:
                return _json_error('There is no earlier version to show.')
            arr = agent_engine_unpack(undo[-1])
            img = Image.fromarray(((arr == 0) * 255).astype('uint8'), 'L')
        else:
            if working is not None:
                arr = np.asarray(working)
                img = Image.fromarray(((arr == 0) * 255).astype('uint8'), 'L')
            elif img is None:
                return _json_error('Nothing has been designed yet.')

        buf = io.BytesIO()
        preview = img.convert('L')
        if max(preview.size) > 2400:
            scale = 2400 / max(preview.size)
            preview = preview.resize((max(1, int(preview.size[0] * scale)),
                                      max(1, int(preview.size[1] * scale))))
        preview.save(buf, format='PNG')
        buf.seek(0)
        resp = send_file(buf, mimetype='image/png')
        # The design changes every turn, so a cached preview would show the
        # weaver the previous design and look like the refinement did nothing.
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        return _json_error(f'Preview failed: {e}')


@app.route('/api/agent/stream', methods=['GET'])
def api_agent_stream():
    """
    Run one turn, streaming progress as server-sent events.

    A full agentic turn runs a dozen tools over a minute or more. Returning one
    JSON blob at the end means a static spinner for that whole time, which is
    indistinguishable from a hang — and it hides the work, which is the part
    worth seeing.

    GET rather than POST because EventSource only issues GETs. The message
    rides in the query string, which caps it at a couple of thousand
    characters — fine for a chat line.
    """
    import json as _json
    import queue
    import threading

    import agent_engine

    token = request.args.get('token', '')
    message = str(request.args.get('message', '')).strip()[:2000]
    session = agent_engine.get_session(token)

    def run():
        if not session:
            yield f'data: {_json.dumps({"type": "error", "error": "That conversation has expired. Start again."})}\n\n'
            return
        if not message:
            yield f'data: {_json.dumps({"type": "error", "error": "No message provided."})}\n\n'
            return

        events = queue.Queue()
        result = {}

        def worker():
            try:
                result['out'] = agent_engine.converse(
                    session, message, on_event=events.put)
            except Exception as e:
                result['out'] = {'ok': False, 'reply': f'Assistant failed: {e}',
                                 'tools_used': [], 'has_files': False}
            finally:
                events.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            try:
                ev = events.get(timeout=90)
            except queue.Empty:
                yield f'data: {_json.dumps({"type": "error", "error": "The assistant took too long."})}\n\n'
                return
            if ev is None:
                break
            # 'reply' is emitted mid-loop but the turn is not finished until
            # the worker returns, so it is skipped here and sent as 'done'.
            if ev.get('type') != 'reply':
                yield f'data: {_json.dumps(ev)}\n\n'

        out = result.get('out') or {}
        yield 'data: ' + _json.dumps({
            'type': 'done', 'ok': out.get('ok', False),
            'reply': out.get('reply', ''),
            'tools_used': out.get('tools_used', []),
            'has_files': out.get('has_files', False),
            'plan': session.get('plan') if session else None,
            'has_design': bool(session and session.get('image') is not None),
        }) + '\n\n'

    return Response(stream_with_context(run()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/agent/download', methods=['GET'])
def api_agent_download():
    """Download the BMPs the agent generated for this conversation."""
    try:
        import agent_engine
        session = agent_engine.get_session(request.args.get('token', ''))
        if not session:
            return _json_error('That conversation has expired.')
        payload, name = agent_engine.files_zip(session)
        if not payload:
            return _json_error('No files have been generated yet.')
        return send_file(io.BytesIO(payload), mimetype='application/zip',
                         as_attachment=True, download_name=name)
    except Exception as e:
        return _json_error(f'Download failed: {e}')


def agent_engine_available():
    """
    True when a model backend is configured, so the UI can hide the panel.

    Asks the provider layer rather than checking for an Anthropic key: a mill
    running a local Llama has no key at all, and gating on one would hide a
    working assistant.
    """
    try:
        import llm
        return llm.is_available()
    except Exception:
        return False


@app.route('/api/auto-convert', methods=['POST'])
def api_auto_convert():
    """
    Convert with self-checking: generate, score against the source, retry.

    The app's failure mode has always been silence — a clean, valid BMP whose
    design had been ruined, with nothing to flag it. This tries the settings
    that have historically been the right answer, scores each against the
    uploaded image, and returns the winner together with what it cost and what
    the alternatives were.

    Returns a PROPOSAL, not a file. Pin count and shuttle mapping are craft and
    hardware decisions; the weaver confirms before anything is generated.
    """
    try:
        if 'image' not in request.files:
            return _json_error('No image uploaded.')
        raw = request.files['image'].read()
        if not raw:
            return _json_error('Uploaded file is empty.')
        img = _open_upload(raw)

        pins = request.form.get('pins')
        pins = int(pins) if pins and str(pins).strip().isdigit() else None
        if pins is not None:
            if pins < 10:
                return _json_error('Pins must be at least 10.')
            _e = _bounds_error(pins)
            if _e:
                return _e

        n_colors = max(2, min(8, int(request.form.get('n_colors', 2) or 2)))

        from auto_convert import auto_convert
        result = auto_convert(img, pins=pins, n_colors=n_colors)
        best = result.get('best')
        if not best:
            return _json_error(result.get('summary', 'Conversion failed.'))

        return jsonify({
            'success': True,
            'verdict': result['verdict'],
            'summary': result['summary'],
            'advice': result['advice'],
            'attempts': result['attempts'],
            'pins': best['pins'],
            'cards': best['cards'],
            'settings': best['settings'],
            'report': best['report'],
            'label_map': _label_map_b64(best['label_map']),
            'alternatives': result['alternatives'],
        })
    except UnidentifiedImageError:
        return _json_error('That file is not a readable image.')
    except Exception as e:
        return _json_error(f'Auto-convert failed: {e}')


@app.route('/api/analyze-design', methods=['POST'])
def api_analyze_design():
    """
    Ask the model how the detected colours should map onto shuttles.

    This is the one step in the pipeline that is a judgement call rather than a
    measurement: clustering reports visual tones, but artwork routinely renders
    a single zari thread as cream in the highlights and deep ochre in shadow.
    Nothing in the pixels distinguishes 'two tones of one thread' from 'two
    threads'.

    The proposal is validated against the shuttle budget before it is returned,
    and grouping is all-or-nothing, so a partially-wrong reading is never
    silently applied.
    """
    try:
        data = request.get_json() or {}
        image_b64 = str(data.get('image_b64', '') or '')
        if ',' in image_b64[:64]:
            image_b64 = image_b64.split(',', 1)[1]      # strip data: prefix
        if not image_b64:
            return _json_error('No image provided.')

        colors = data.get('colors') or []
        counts = data.get('counts') or []
        shuttle_count = _coerce_shuttles(data.get('shuttle_count', 2))
        if not colors:
            return _json_error('Run colour detection first.')

        result = assistant_engine.analyze_design(
            image_b64, colors, counts, shuttle_count,
            media_type=str(data.get('media_type', 'image/png')))
        return jsonify({'success': result['ok'], **result})
    except Exception as e:
        return _json_error(f'Analysis failed: {e}')


def _coerce_shuttles(v):
    try:
        return max(1, min(4, int(v)))
    except (TypeError, ValueError):
        return 2


@app.route('/api/assistant', methods=['POST'])
def api_assistant():
    """
    One conversational turn against the Generator settings.

    The model proposes a settings patch; assistant_engine validates it against
    loom and shuttle limits before it is returned. The patch is applied by the
    frontend only after the weaver confirms it, so nothing changes silently.
    """
    try:
        data = request.get_json() or {}
        message = str(data.get('message', '')).strip()
        if not message:
            return _json_error('No message provided.')
        if len(message) > 2000:
            return _json_error('Message too long.')

        state = data.get('state') or {}
        history = data.get('history') or []
        if not isinstance(state, dict) or not isinstance(history, list):
            return _json_error('Malformed assistant state.')

        result = assistant_engine.ask(message, state, history[-10:])
        return jsonify({'success': result['ok'], **result})
    except Exception as e:
        return _json_error(f'Assistant failed: {e}')


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
        _e = _bounds_error(pins)
        if _e: return _e
        if cards < 10:
            return _json_error('Cards must be at least 10.')
        _e = _bounds_error(pins, cards)
        if _e: return _e
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

        # Auto-preprocessing: denoise ONLY when the image is genuinely noisy.
        #
        # This previously triggered on assess_image_quality's whole-frame noise
        # and JPEG scores (> 20 / > 15). Those metrics saturate on dense
        # artwork — both a real saree border and a clean line-art butta scored
        # 86-100 — so enhancement ran on every detailed design. Measured effect
        # on a real border: ink components rose from 249 to 1159 and background
        # regions from 997 to 1485. It was shattering the designs it claimed to
        # clean.
        #
        # estimate_noise measures variation only in low-gradient areas, which
        # separates sensor noise from design detail, so this now fires on
        # genuinely noisy photographs and leaves clean artwork alone.
        try:
            from enhanced_engine import estimate_noise
            _n = estimate_noise(img)
            if _n.get('recommend_enhance'):
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
            'fidelity':     _fidelity_of(label_map, img),
            'composite_b64': _composite_render(bmp_files, pins, cards),
        })

    except ValueError as e:
        # Raised by generate_bmps for label_map shape mismatch
        return _json_error(str(e))
    except Exception as e:
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
        img = _open_upload(img)
        quality = assess_image_quality(img)
        try:
            from enhanced_engine import estimate_noise
            _noise = estimate_noise(img)
            quality['noise_level'] = _noise['noise_level']
            quality['noise_sigma'] = _noise['noise_sigma']
            quality['recommend_enhance'] = _noise['recommend_enhance']
            # Replace the blanket "high noise" advice with the measured verdict.
            quality['suggestions'] = [
                t for t in (quality.get('suggestions') or [])
                if 'noise' not in t.lower() and 'artifact' not in t.lower()
            ]
            if _noise['recommend_enhance']:
                quality['suggestions'].append(_noise['reason'])
        except Exception:
            pass
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
        img = _open_upload(img)

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

        # Pin advice computed from THIS image rather than a fixed number. The
        # page previously told everyone to use 960 pins, which is wrong for
        # most sources: what a design can carry depends on how wide its
        # thinnest stroke is in the original file, not on a default. Measured
        # against the ORIGINAL upload, not the downscaled working copy.
        pin_advice = None
        try:
            from loom_utils import source_resolution_check
            orig = Image.open(io.BytesIO(raw))
            orig = ImageOps.exif_transpose(orig).convert('RGB')
            best = None
            for cand in (240, 360, 480, 600, 720, 858, 960, 1200):
                chk = source_resolution_check(orig, cand)
                if chk.get('ok'):
                    best = cand
                    break
            probe = source_resolution_check(orig, best or 480)
            pin_advice = {
                'source_width': orig.size[0],
                'stroke_px': probe.get('stroke_px'),
                'recommended_pins': best or probe.get('recommended_min_pins'),
                'achievable': bool(best),
                'note': (
                    f"Your finest strokes are about {probe.get('stroke_px')}px wide in a "
                    f"{orig.size[0]}px image. "
                    + (f"At {best} pins each stroke lands on two or more threads, "
                       f"so detail survives."
                       if best else
                       "No standard pin count keeps two threads per stroke — rescan "
                       "larger, or accept lighter linework.")),
            }
        except Exception:
            pass

        return jsonify({
            'success':   True,
            'pin_advice': pin_advice,
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
            pat     = params.get('pattern', 'satin')
            n_val   = max(4, min(16, int(params.get('n', 8))))
            flip    = bool(params.get('flip', False))
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
        _e = _bounds_error(pins)
        if _e: return _e

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
                _e = _bounds_error(pins, cards)
                if _e: return _e
            except ValueError:
                return _json_error('Cards must be a whole number.')

        try:
            img = Image.open(file.stream)
            img = _open_upload(img)
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
        _e = _bounds_error(pins)
        if _e: return _e
        if cards < 10:
            return _json_error('Cards must be at least 10.')
        _e = _bounds_error(pins, cards)
        if _e: return _e
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
        return _json_error(f'Border generation failed: {e}')




# ─────────────────────────────────────────────────────────────────────────────
# BORDER IDENTIFICATION  (/border-id) — enhanced fine-detail generation
# Detection reuses /api/border-detect (unchanged).
# Generation uses border_id_engine: adaptive scale, dual-threshold pooling,
# pre-pool closing.
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: the /border-id PAGE was removed; this endpoint is retained because
# Border Studio's fine-detail mode posts here (see templates/border.html).
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
        _e = _bounds_error(pins, cards)
        if _e: return _e
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
            img = _open_upload(img)
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


def _fidelity_of(label_map, source_image):
    """
    Compare the reduced DESIGN against the uploaded source.

    Deliberately uses the label map, not the generated BMPs. The BMPs carry
    weave fill, and a satin float pattern punches thousands of tiny holes in
    the ink by design — measuring those counts the weave rather than the
    artwork. On a real border this read 21,367 background areas against the
    source's 377 and wrongly reported total failure, when the design geometry
    was intact and only the satin texture differed.

    The design mask is every non-background label, which is exactly the
    geometry that reduction can damage.
    """
    try:
        from fidelity import fidelity_report
        lm = np.asarray(label_map)
        if lm.ndim != 2 or lm.size == 0:
            return None
        return fidelity_report(source_image, lm > 0)
    except Exception:
        return None


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
            img = _open_upload(img)
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
