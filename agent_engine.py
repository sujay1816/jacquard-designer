"""
Conversational conversion agent.

Takes an uploaded design, asks what the job needs, converts it, explains what
the conversion cost, and hands back loom-ready BMPs.

The division of labour is the point:

  * The MODEL decides what to ask, which tool to call, and how to explain the
    result in language a weaver can act on.
  * DETERMINISTIC CODE does every pixel operation, every measurement, and
    every validation. The model never sees or produces pixel data — it sees
    numbers and names, and calls tools by name.

That boundary is what makes this safe to run unattended. A model that
misunderstands a design produces a bad sentence, not a bad BMP: the settings
it proposes are range-checked, the conversion is scored against the source
before anything is offered for download, and the weaver confirms.

Reproducibility: the conversion itself is deterministic, so the same image and
the same confirmed pin count always yield the same files. The model's
contribution is the conversation around that, which is not part of the output.
"""
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from assistant_engine import API_URL, API_VERSION, API_TIMEOUT, _key, model_id

MAX_TOOL_ROUNDS = 6          # tool calls per user turn before we stop
MAX_HISTORY = 24             # messages retained per session
SESSION_TTL = 3600           # seconds
MAX_SESSIONS = 40

# Session state lives here: the uploaded image, the last conversion, and the
# generated files. Kept server-side so the model never handles image data.
_sessions = {}


# ── Session handling ────────────────────────────────────────────────────────

def _prune():
    now = time.time()
    for k in [k for k, v in _sessions.items() if now - v['created'] > SESSION_TTL]:
        _sessions.pop(k, None)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)), None)


def new_session(image, filename='design'):
    """Register an uploaded image and return its session token."""
    import uuid
    _prune()
    token = str(uuid.uuid4())
    _sessions[token] = {
        'created': time.time(),
        'image': image,
        'filename': os.path.splitext(os.path.basename(filename))[0] or 'design',
        'history': [],
        'conversion': None,
        'files': None,
    }
    return token


def get_session(token):
    s = _sessions.get(token)
    if s and time.time() - s['created'] > SESSION_TTL:
        _sessions.pop(token, None)
        return None
    return s


# ── Tools ───────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "inspect_design",
        "description": (
            "Examine the uploaded image: size, whether it is line art or "
            "colour work, how wide its finest strokes are, image quality, and "
            "the pin count it can support. Call this first, before asking the "
            "weaver anything, so the questions are informed."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "convert",
        "description": (
            "Convert the design at a given pin count and score the result "
            "against the original. Returns the verdict, how much thread "
            "coverage drifted, how many design gaps survived, and what the "
            "alternatives would give. Does NOT produce files — it reports what "
            "would happen. Omit pins to let it search for the best."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer",
                         "description": "Pin count this job requires. Omit to search."},
                "n_colors": {"type": "integer",
                             "description": "Colours to detect. 2 for line art."},
            },
        },
    },
    {
        "name": "generate_files",
        "description": (
            "Produce the downloadable BMP files using the last conversion. "
            "Only call this after the weaver has confirmed the pin count and "
            "shuttle assignment."),
        "input_schema": {
            "type": "object",
            "properties": {
                "shuttle_count": {"type": "integer",
                                  "description": "Physical shuttles, 1-4."},
                "design_name": {"type": "string"},
            },
        },
    },
]

SYSTEM_PROMPT = """You convert saree and brocade designs into loom-ready BMP \
files, one per shuttle. You are talking to a weaver or a mill operator.

How to work:
1. Call inspect_design first. Never ask a question you could have answered by \
looking.
2. Report what you found in one or two short sentences, then ask how many pins \
the job needs. Suggest the count the image can actually support.
3. When they answer, call convert with that pin count.
4. Report the verdict honestly. If detail will be lost, say WHAT will be lost \
in craft terms — "the fine scrollwork inside the small motifs" beats "12% ink \
drift". Mention a better pin count only if one genuinely exists.
5. When they confirm, call generate_files.

What matters:
- Black lifts the thread, white leaves it down. A BMP is an instruction sheet.
- Shuttles are hardware. A loom weaves exactly shuttle_count threads and no \
more.
- Never claim a conversion is good when the tools reported WARN or FAIL. If \
the source is too small to carry the design, say so plainly — a rescan is the \
only fix and the weaver needs to hear it before the cloth is woven, not after.

Be brief and concrete. Talk like a colleague on the mill floor. Do not explain \
your tools or narrate what you are about to do."""


def _tool_inspect(session, args):
    from bmp_engine import assess_image_quality
    from enhanced_engine import estimate_noise
    from loom_utils import source_resolution_check
    from vision_engine import _is_achromatic, _prepare_lineart

    img = session['image']
    w, h = img.size
    arr = np.asarray(_prepare_lineart(img).resize(
        (min(400, w), min(400, h))), dtype=np.uint8)
    achromatic = bool(_is_achromatic(arr))

    q = assess_image_quality(img)
    n = estimate_noise(img)

    best_pins, stroke = None, None
    for cand in (240, 360, 480, 600, 720, 960, 1200):
        chk = source_resolution_check(img, cand)
        stroke = chk.get('stroke_px', stroke)
        if chk.get('ok'):
            best_pins = cand
            break

    return {
        'width': w, 'height': h,
        'type': 'line art (no colour)' if achromatic else 'colour design',
        'finest_stroke_px': stroke,
        'noise_sigma': n.get('noise_sigma'),
        'needs_denoise': n.get('recommend_enhance'),
        'blurry': q.get('blur_score', 1e9) < 200,
        'supported_pin_count': best_pins,
        'note': (f"Detail supports about {best_pins} pins."
                 if best_pins else
                 "No standard pin count keeps two threads per stroke; the "
                 "source is low resolution for its detail."),
    }


def _tool_convert(session, args):
    from auto_convert import auto_convert

    pins = args.get('pins')
    if pins is not None:
        try:
            pins = int(pins)
        except (TypeError, ValueError):
            return {'error': 'Pins must be a whole number.'}
        if not (10 <= pins <= 2640):
            return {'error': 'Pins must be between 10 and 2640.'}

    try:
        n_colors = max(2, min(8, int(args.get('n_colors', 2) or 2)))
    except (TypeError, ValueError):
        n_colors = 2

    result = auto_convert(session['image'], pins=pins, n_colors=n_colors)
    if not result.get('best'):
        return {'error': result.get('summary', 'Conversion failed.')}

    session['conversion'] = result
    best, rep = result['best'], result['best']['report']
    return {
        'pins': best['pins'], 'cards': best['cards'],
        'verdict': result['verdict'],
        'thread_drift_pct': rep['ink_drift_pct'],
        'design_gaps_source': rep['source_white_regions'],
        'design_gaps_result': rep['output_white_regions'],
        'settings_tried': result['attempts'],
        'notes': result['advice'],
        'alternatives': [
            {'pins': a['pins'], 'verdict': a['verdict'],
             'thread_drift_pct': a['ink_drift_pct'], 'gaps': a['gaps']}
            for a in result['alternatives']],
    }


def _tool_generate(session, args):
    import bmp_engine as be

    conv = session.get('conversion')
    if not conv or not conv.get('best'):
        return {'error': 'Run convert first.'}

    try:
        shuttles = max(1, min(4, int(args.get('shuttle_count', 2) or 2)))
    except (TypeError, ValueError):
        shuttles = 2

    best = conv['best']
    label_map = np.asarray(best['label_map'])
    n_labels = int(label_map.max()) + 1

    # Background is the largest region; remaining labels take shuttles in
    # order. The model does not choose this — an unvalidated mapping could
    # assign more threads than the loom carries.
    names = ['zari', 'meena1', 'meena2']
    counts = np.bincount(label_map.ravel(), minlength=n_labels)
    order = list(np.argsort(-counts))
    assignments = {int(order[0]): 'background'}
    for i, idx in enumerate(order[1:]):
        if i < min(shuttles, len(names)):
            assignments[int(idx)] = names[i]
        else:
            assignments[int(idx)] = 'background'

    satin = {n: {'n': 8, 'flip': False} for n in names}
    name = str(args.get('design_name') or session['filename'])[:40]

    files = be.generate_bmps(
        image=session['image'], pins=best['pins'], cards=best['cards'],
        shuttle_count=shuttles, color_assignments=assignments,
        satin_settings=satin, design_name=name,
        label_map=best['label_map'], stroke_mode=False, reed=60)

    from loom_utils import count_long_floats, physical_size
    summary, worst_float = [], 0
    for fn, data in sorted(files.items()):
        info = be.verify_bmp(data)
        mask = np.array(Image.open(io.BytesIO(data)).convert('L')) < 128
        _, longest = count_long_floats(mask, 12)
        worst_float = max(worst_float, longest)
        summary.append({'file': fn, 'bytes': len(data),
                        'clean_1bit': info['is_clean'],
                        'threads_up': info['pure_black'],
                        'longest_float': longest})

    session['files'] = files
    size = physical_size(best['pins'], best['cards'], 60)
    return {
        'ready': True, 'files': summary,
        'physical_size_in': f"{size['width_in']} x {size['height_in']} at 60 reed",
        'longest_float': worst_float,
        'float_warning': (
            f"Longest float is {worst_float} picks — check with the loom "
            f"operator before running." if worst_float > 30 else None),
    }


_DISPATCH = {
    'inspect_design': _tool_inspect,
    'convert': _tool_convert,
    'generate_files': _tool_generate,
}


def run_tool(name, args, session):
    """Execute a tool by name. Unknown tools and failures return an error dict
    rather than raising, so a confused model gets a correction instead of
    crashing the request."""
    fn = _DISPATCH.get(name)
    if not fn:
        return {'error': f'No such tool: {name}'}
    try:
        return fn(session, args if isinstance(args, dict) else {})
    except Exception as e:
        return {'error': f'{name} failed: {e}'}


# ── Conversation loop ───────────────────────────────────────────────────────

def _call_api(messages):
    key = _key()
    if not key:
        return None, 'No API key configured. Set ANTHROPIC_API_KEY or add it to config.json.'
    body = json.dumps({
        "model": model_id(),
        "max_tokens": 1400,
        "system": SYSTEM_PROMPT,
        "tools": TOOLS,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": API_VERSION,
    })
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as e:
        detail = 'check the API key' if e.code in (401, 403) else f'HTTP {e.code}'
        return None, f'Assistant unavailable ({detail}).'
    except Exception:
        return None, 'Assistant unreachable. Check the network connection.'


def converse(session, user_message):
    """
    Run one user turn, executing tools until the model produces a reply.

    Returns {'ok', 'reply', 'tools_used', 'has_files'}.
    """
    history = session['history']
    history.append({'role': 'user', 'content': user_message})

    tools_used = []
    for _ in range(MAX_TOOL_ROUNDS):
        data, err = _call_api(history[-MAX_HISTORY:])
        if err:
            history.pop()
            return {'ok': False, 'reply': err, 'tools_used': tools_used,
                    'has_files': bool(session.get('files'))}

        blocks = data.get('content') or []
        history.append({'role': 'assistant', 'content': blocks})

        calls = [b for b in blocks if b.get('type') == 'tool_use']
        if not calls:
            text = ' '.join(b['text'].strip() for b in blocks
                            if b.get('type') == 'text' and b.get('text'))
            return {'ok': True, 'reply': text or 'Done.',
                    'tools_used': tools_used,
                    'has_files': bool(session.get('files'))}

        results = []
        for call in calls:
            tools_used.append(call.get('name'))
            out = run_tool(call.get('name'), call.get('input'), session)
            results.append({'type': 'tool_result', 'tool_use_id': call.get('id'),
                            'content': json.dumps(out, default=str)})
        history.append({'role': 'user', 'content': results})

    return {'ok': True,
            'reply': 'That took more steps than expected — could you rephrase?',
            'tools_used': tools_used, 'has_files': bool(session.get('files'))}


def files_zip(session):
    """Package the generated BMPs. Returns (bytes, filename) or (None, None)."""
    files = session.get('files')
    if not files:
        return None, None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn, data in files.items():
            z.writestr(fn, data)
    return buf.getvalue(), f"{session['filename']}_bmps.zip"
