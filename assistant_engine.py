"""
Assistant engine — natural-language control of the Generator.

Design rule: the model NEVER produces pixels, masks, or BMP data. It emits a
structured patch against the Generator's settings, which is then validated
against loom and hardware constraints before anything runs. Every edit is
executed by the existing deterministic code paths, so the loom-safety
properties of the output are unchanged.

The trust boundary is validate_patch(). It is pure Python with no network
dependency, and it rejects anything physically unweavable regardless of what
the model asked for. Treat model output as untrusted input.

API access: the key is supplied by the operator, never bundled. Set the
ANTHROPIC_API_KEY environment variable, or place it in config.json (which is
gitignored). With no key configured the assistant is simply unavailable and
the rest of the app is unaffected.
"""
import json
import os
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
API_TIMEOUT = 60

# Model IDs from the 4.6 generation onward are PINNED, not evergreen aliases:
# a given ID always maps to one fixed snapshot and Anthropic never changes its
# weights. That matters here — it means the assistant's behaviour cannot drift
# under a design that is already in production. Overridable via the
# JQ_ASSISTANT_MODEL environment variable or "model" in config.json, so a site
# can pin an older ID or move to a newer one without editing source.
DEFAULT_MODEL = "claude-sonnet-5"


def _config():
    """Read config.json next to this file, or {} if absent/unreadable."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'config.json'), encoding='utf-8') as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def model_id():
    """Resolve the model ID: env var, then config.json, then the default."""
    return (os.environ.get('JQ_ASSISTANT_MODEL', '').strip()
            or str(_config().get('model', '')).strip()
            or DEFAULT_MODEL)

# ── Hardware and loom limits ────────────────────────────────────────────────
# These mirror the limits already enforced in app.py and loom_utils, and are
# the reason the assistant cannot produce an unweavable design.
SHUTTLE_NAMES = ('zari', 'meena1', 'meena2', 'background')
MAX_SHUTTLES = 4
MIN_PINS, MAX_PINS = 10, 2640
MIN_CARDS, MAX_CARDS = 10, 6000
MIN_SATIN_N, MAX_SATIN_N = 4, 16
MIN_STROKE, MAX_STROKE = 1, 5
RANI_WEAVES = ('plain', 'twill', 'matt')

# Settings the assistant is allowed to touch. Anything outside this set is
# dropped, so a model that invents a field cannot reach the generator.
EDITABLE_FIELDS = (
    'pins', 'cards', 'shuttle_count', 'color_assignments', 'satin_settings',
    'rani_weave', 'stroke_mode', 'stroke_thickness', 'reed', 'supersample',
    'curvilinear_satin', 'invert_output', 'outline_white',
)


def _key():
    """Operator-supplied API key, or None. Env var wins over config.json."""
    k = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if k:
        return k
    return (_config().get('anthropic_api_key') or '').strip() or None


def is_available():
    """True when an API key is configured. The UI hides the panel if False."""
    return _key() is not None


# ── Tool schema handed to the model ─────────────────────────────────────────
UPDATE_SETTINGS_TOOL = {
    "name": "update_settings",
    "description": (
        "Apply a change to the jacquard Generator settings. Only include fields "
        "that should change; omit everything else. Never invent colour indices "
        "that were not detected, and never assign more distinct shuttles than "
        "the loom's shuttle_count allows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pins": {"type": "integer", "description": "Loom width in threads."},
            "cards": {"type": "integer", "description": "Design height in cards."},
            "shuttle_count": {"type": "integer",
                              "description": "Number of physical shuttles, 1-4."},
            "color_assignments": {
                "type": "object",
                "description": ("Map of detected colour index (as a string) to "
                                "shuttle name: zari, meena1, meena2, or background."),
            },
            "satin_settings": {
                "type": "object",
                "description": ("Per-shuttle weave, e.g. "
                                "{\"zari\": {\"n\": 8, \"flip\": false}}. "
                                "Higher n means a longer, glossier float."),
            },
            "rani_weave": {"type": "string", "enum": list(RANI_WEAVES)},
            "stroke_mode": {"type": "boolean"},
            "stroke_thickness": {"type": "integer", "description": "Outline width 1-5."},
            "reed": {"type": "integer", "description": "Reed count, typically 60/80/100."},
            "supersample": {"type": "boolean"},
            "curvilinear_satin": {"type": "boolean"},
            "explanation": {
                "type": "string",
                "description": "One short sentence telling the weaver what changed.",
            },
        },
        "required": ["explanation"],
    },
}

SYSTEM_PROMPT = """You help a weaver operate a jacquard design tool that converts \
saree designs into loom-ready BMP files, one file per shuttle.

Physical constraints you must respect:
- Shuttles are hardware. The loom weaves exactly shuttle_count threads. You \
cannot add a colour beyond that budget. If the weaver asks for a new colour and \
no shuttle is free, say so plainly and offer the real options: replace a colour \
currently assigned, or raise shuttle_count if their loom has a spare shuttle.
- Only colour indices that were actually detected in the image can be assigned.
- Black in the output means thread UP (visible); white means thread DOWN.
- Higher satin n gives a longer float: glossier, but snags more easily. Above \
about 12 warn the weaver that floats may catch.

You do not edit images or pixels. You only adjust settings via the \
update_settings tool. If a request needs pixel-level work, say it belongs in \
the BMP Editor instead.

Be brief and concrete. Talk like a colleague on the mill floor, not a chatbot."""


def _coerce_int(value, lo, hi):
    """Clamp to [lo,hi]; return None if not a number."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return None


def validate_patch(patch: dict, state: dict) -> dict:
    """
    Validate a model-proposed settings patch against loom reality.

    This is the trust boundary. It runs on untrusted model output and must
    never let a physically unweavable configuration through.

    patch : proposed changes
    state : current settings, including 'detected_colors' (count of colours
            found in the detect step)

    Returns {'patch': cleaned_patch, 'rejected': [human-readable reasons]}.
    """
    clean, rejected = {}, []
    patch = patch if isinstance(patch, dict) else {}
    state = state if isinstance(state, dict) else {}

    n_detected = _coerce_int(state.get('detected_colors', 0), 0, 64) or 0

    # ── shuttle_count first: later checks depend on the final value ─────────
    shuttle_count = _coerce_int(state.get('shuttle_count', 1), 1, MAX_SHUTTLES) or 1
    if 'shuttle_count' in patch:
        v = _coerce_int(patch['shuttle_count'], 1, MAX_SHUTTLES)
        if v is None:
            rejected.append("Shuttle count must be a whole number.")
        else:
            if int(patch['shuttle_count']) != v:
                rejected.append(
                    f"A loom cannot use {patch['shuttle_count']} shuttles; "
                    f"clamped to {v} (the maximum is {MAX_SHUTTLES}).")
            clean['shuttle_count'] = shuttle_count = v

    # ── Dimensions ─────────────────────────────────────────────────────────
    for field, lo, hi in (('pins', MIN_PINS, MAX_PINS), ('cards', MIN_CARDS, MAX_CARDS)):
        if field in patch:
            v = _coerce_int(patch[field], lo, hi)
            if v is None:
                rejected.append(f"{field.capitalize()} must be a whole number.")
                continue
            if int(patch[field]) != v:
                rejected.append(
                    f"{int(patch[field])} {field} is outside the loom's range "
                    f"({lo}-{hi}); clamped to {v}.")
            clean[field] = v

    # ── Colour assignments: the shuttle budget lives here ──────────────────
    if 'color_assignments' in patch:
        raw = patch['color_assignments']
        if not isinstance(raw, dict):
            rejected.append("Colour assignments must be a mapping.")
        else:
            assigned, seen_shuttles = {}, set()
            for idx, shuttle in raw.items():
                i = _coerce_int(idx, 0, 63)
                if i is None or (n_detected and i >= n_detected):
                    rejected.append(
                        f"Colour {idx} was not detected in this image, so it "
                        f"cannot be assigned.")
                    continue
                if shuttle not in SHUTTLE_NAMES:
                    rejected.append(
                        f"'{shuttle}' is not a shuttle on this loom "
                        f"({', '.join(SHUTTLE_NAMES)}).")
                    continue
                # 'background' is the unwoven ground, not a shuttle, so it does
                # not consume the budget.
                if shuttle != 'background':
                    seen_shuttles.add(shuttle)
                assigned[str(i)] = shuttle

            if len(seen_shuttles) > shuttle_count:
                rejected.append(
                    f"That needs {len(seen_shuttles)} threads but the loom is set "
                    f"to {shuttle_count} shuttle"
                    f"{'s' if shuttle_count != 1 else ''}. Free one up, or raise "
                    f"the shuttle count if the loom has a spare.")
            elif assigned:
                clean['color_assignments'] = assigned

    # ── Per-shuttle weave ──────────────────────────────────────────────────
    if 'satin_settings' in patch:
        raw = patch['satin_settings']
        if not isinstance(raw, dict):
            rejected.append("Weave settings must be a mapping.")
        else:
            out = {}
            for shuttle, cfg in raw.items():
                if shuttle not in SHUTTLE_NAMES:
                    rejected.append(f"'{shuttle}' is not a shuttle on this loom.")
                    continue
                if not isinstance(cfg, dict):
                    rejected.append(f"Weave settings for {shuttle} are malformed.")
                    continue
                entry = {}
                if 'n' in cfg:
                    v = _coerce_int(cfg['n'], MIN_SATIN_N, MAX_SATIN_N)
                    if v is None:
                        rejected.append(f"Satin count for {shuttle} must be a number.")
                        continue
                    if int(cfg['n']) != v:
                        rejected.append(
                            f"Satin {int(cfg['n'])} for {shuttle} is out of range; "
                            f"clamped to {v}.")
                    entry['n'] = v
                if 'flip' in cfg:
                    entry['flip'] = bool(cfg['flip'])
                if 'weave_off' in cfg:
                    entry['weave_off'] = bool(cfg['weave_off'])
                if entry:
                    out[shuttle] = entry
            if out:
                clean['satin_settings'] = out

    # ── Simple scalars ─────────────────────────────────────────────────────
    if 'rani_weave' in patch:
        v = str(patch['rani_weave']).lower()
        if v in RANI_WEAVES:
            clean['rani_weave'] = v
        else:
            rejected.append(
                f"'{patch['rani_weave']}' is not a rani weave "
                f"({', '.join(RANI_WEAVES)}).")

    if 'stroke_thickness' in patch:
        v = _coerce_int(patch['stroke_thickness'], MIN_STROKE, MAX_STROKE)
        if v is None:
            rejected.append("Outline thickness must be a whole number.")
        else:
            if int(patch['stroke_thickness']) != v:
                rejected.append(
                    f"Outline thickness clamped to {v} "
                    f"(range {MIN_STROKE}-{MAX_STROKE}).")
            clean['stroke_thickness'] = v

    if 'reed' in patch:
        v = _coerce_int(patch['reed'], 1, 200)
        if v is None:
            rejected.append("Reed must be a whole number.")
        else:
            clean['reed'] = v

    for flag in ('stroke_mode', 'supersample', 'curvilinear_satin'):
        if flag in patch:
            clean[flag] = bool(patch[flag])

    # Drop anything not on the allow-list, so an invented field cannot reach
    # the generator even if every check above passed.
    clean = {k: v for k, v in clean.items() if k in EDITABLE_FIELDS}
    return {'patch': clean, 'rejected': rejected}


def advisories(patch: dict, state: dict) -> list:
    """
    Non-blocking warnings about weave-ability. These do not reject the change;
    they tell the weaver what to expect at the loom.
    """
    notes = []
    merged = dict(state or {})
    merged.update(patch or {})

    for shuttle, cfg in (patch.get('satin_settings') or {}).items():
        n = cfg.get('n')
        if isinstance(n, int) and n > 12:
            notes.append(
                f"Satin {n} on {shuttle} leaves long floats — glossy, but they "
                f"snag more easily.")

    pins, cards = merged.get('pins'), merged.get('cards')
    if isinstance(pins, int) and isinstance(cards, int):
        if pins * cards > 4_000_000:
            notes.append(
                f"{pins} x {cards} is a very large card count; generation will "
                f"be slow.")
    return notes


def ask(message: str, state: dict, history=None) -> dict:
    """
    Send one turn to the model and return a validated result.

    Returns:
        {'ok', 'reply', 'patch', 'rejected', 'advisories'}
    On any failure returns ok=False with a human-readable 'reply'.
    """
    key = _key()
    if not key:
        return {'ok': False, 'reply': (
            'No API key configured. Set ANTHROPIC_API_KEY or add '
            '"anthropic_api_key" to config.json to enable the assistant.'),
            'patch': {}, 'rejected': [], 'advisories': []}

    msgs = list(history or [])
    msgs.append({
        "role": "user",
        "content": (f"Current settings:\n{json.dumps(state, indent=2)}\n\n"
                    f"Weaver says: {message}"),
    })

    body = json.dumps({
        "model": model_id(),
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "tools": [UPDATE_SETTINGS_TOOL],
        "messages": msgs,
    }).encode()

    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": API_VERSION,
    })

    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = 'check that the API key is valid' if e.code in (401, 403) \
            else f'HTTP {e.code}'
        return {'ok': False, 'reply': f'Assistant unavailable ({detail}).',
                'patch': {}, 'rejected': [], 'advisories': []}
    except Exception:
        return {'ok': False,
                'reply': 'Assistant unreachable. Check the network connection.',
                'patch': {}, 'rejected': [], 'advisories': []}

    return interpret_response(data, state)


def interpret_response(data: dict, state: dict) -> dict:
    """
    Turn a raw API response into a validated result. Split out from ask() so
    it can be tested without network access.
    """
    text_parts, proposed = [], {}
    for block in (data.get('content') or []):
        btype = block.get('type')
        if btype == 'text' and block.get('text'):
            text_parts.append(block['text'])
        elif btype == 'tool_use' and block.get('name') == 'update_settings':
            proposed = dict(block.get('input') or {})

    explanation = proposed.pop('explanation', '')
    result = validate_patch(proposed, state)
    reply = ' '.join(p.strip() for p in text_parts if p.strip()) or explanation

    return {
        'ok': True,
        'reply': reply or 'Done.',
        'patch': result['patch'],
        'rejected': result['rejected'],
        'advisories': advisories(result['patch'], state),
    }


# ── Design analysis: colour grouping ────────────────────────────────────────
#
# This is the one task in the pipeline that is genuinely a judgement call
# rather than a measurement. Detection returns visual tones; a loom weaves
# threads. A saree motif rendered with cream, light pink and deep pink is
# usually ONE zari thread with shading, not three shuttles — but nothing in
# the pixels says so. Clustering can only report that the colours differ.
#
# Everything the model proposes is validated against the shuttle budget before
# it can reach the generator, exactly as with update_settings.

GROUP_COLOURS_TOOL = {
    "name": "group_colours",
    "description": (
        "Assign each detected colour to a loom shuttle. Colours that are "
        "tonal variations of the same thread (highlight, shadow, or shading of "
        "one yarn) must share a shuttle. Only visually distinct threads get "
        "their own. The ground colour of the cloth is 'background'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "object",
                "description": ("Map colour index (as a string) to a shuttle "
                                "name: zari, meena1, meena2, or background."),
            },
            "groups": {
                "type": "array",
                "description": ("One entry per shuttle used, explaining which "
                                "detected colours were merged and why."),
                "items": {
                    "type": "object",
                    "properties": {
                        "shuttle": {"type": "string"},
                        "colour_indices": {"type": "array", "items": {"type": "integer"}},
                        "rationale": {"type": "string"},
                    },
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Low when the grouping is genuinely ambiguous.",
            },
            "explanation": {"type": "string"},
        },
        "required": ["assignments", "explanation"],
    },
}

ANALYSIS_PROMPT = """You are reading a woven textile design to decide how its \
colours map onto loom shuttles.

The critical judgement: detection reports visual TONES, but a loom weaves \
THREADS. Artwork routinely renders one gold zari thread as cream in the \
highlights and deep ochre in the shadows. Those are one shuttle, not two. \
Assign separate shuttles only to threads a weaver would actually load \
separately — typically a metallic (zari) and one or two contrast colours \
(meena).

Rules you cannot break:
- Never use more distinct shuttles than the loom has.
- Exactly one colour should be 'background' — the ground of the cloth, \
usually the most dominant area.
- Every detected colour index must be assigned.

Say so plainly if the grouping is ambiguous, and set confidence to low. A \
weaver can correct a low-confidence guess; they cannot correct one presented \
as certain."""


def analyze_design(image_b64, colors, counts, shuttle_count, media_type='image/png'):
    """
    Ask the model how the detected colours should map onto shuttles.

    image_b64    : base64 of the design image (no data: prefix)
    colors       : list of (R,G,B) tuples, detection order
    counts       : list of pixel counts, parallel to colors
    shuttle_count: how many shuttles the loom has

    Returns {'ok', 'reply', 'assignments', 'groups', 'confidence', 'rejected'}.
    The assignments are validated against the shuttle budget, so an
    over-budget or hallucinated proposal never reaches the generator.
    """
    key = _key()
    if not key:
        return {'ok': False, 'reply': 'No API key configured.',
                'assignments': {}, 'groups': [], 'confidence': None, 'rejected': []}

    total = max(1, sum(counts or [1]))
    palette = "\n".join(
        f"  index {i}: RGB{tuple(int(v) for v in c)}  "
        f"{100.0 * (counts[i] if i < len(counts) else 0) / total:.1f}% of the design"
        for i, c in enumerate(colors or []))

    body = json.dumps({
        "model": model_id(),
        "max_tokens": 1024,
        "system": ANALYSIS_PROMPT,
        "tools": [GROUP_COLOURS_TOOL],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": media_type,
                                             "data": image_b64}},
                {"type": "text", "text": (
                    f"This design was reduced to {len(colors or [])} colours:\n"
                    f"{palette}\n\n"
                    f"The loom has {shuttle_count} shuttle"
                    f"{'s' if shuttle_count != 1 else ''}. Group these colours "
                    f"onto shuttles.")},
            ],
        }],
    }).encode()

    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": API_VERSION,
    })
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = 'check the API key' if e.code in (401, 403) else f'HTTP {e.code}'
        return {'ok': False, 'reply': f'Analysis unavailable ({detail}).',
                'assignments': {}, 'groups': [], 'confidence': None, 'rejected': []}
    except Exception:
        return {'ok': False, 'reply': 'Analysis unreachable.',
                'assignments': {}, 'groups': [], 'confidence': None, 'rejected': []}

    return interpret_analysis(data, colors, shuttle_count)


def interpret_analysis(data, colors, shuttle_count):
    """Parse and validate an analysis response. Split out so it is testable offline."""
    text, proposed = [], {}
    for block in (data.get('content') or []):
        if block.get('type') == 'text' and block.get('text'):
            text.append(block['text'])
        elif block.get('type') == 'tool_use' and block.get('name') == 'group_colours':
            proposed = dict(block.get('input') or {})

    assignments = proposed.get('assignments') or {}
    result = validate_patch({'color_assignments': assignments},
                            {'shuttle_count': shuttle_count,
                             'detected_colors': len(colors or [])})
    clean = result['patch'].get('color_assignments', {})
    rejected = list(result['rejected'])

    # Grouping is all-or-nothing, unlike a settings patch. If the model
    # referenced a colour that does not exist, or named a shuttle the loom does
    # not have, its reading of the palette is wrong and the rest of its
    # reasoning cannot be trusted either. Applying the surviving fragment would
    # silently hand the weaver a grouping the model never actually proposed.
    if rejected:
        clean = {}

    # Every detected colour must be assigned, or the generator would silently
    # drop one. An incomplete grouping is rejected rather than half-applied.
    missing = [i for i in range(len(colors or [])) if str(i) not in clean]
    if clean and missing:
        rejected.append(
            f"Colour{'s' if len(missing) != 1 else ''} "
            f"{', '.join(map(str, missing))} were not assigned to any shuttle.")
        clean = {}

    # Exactly one background keeps the ground unambiguous.
    if clean:
        bg = [i for i, sh in clean.items() if sh == 'background']
        if len(bg) != 1:
            rejected.append(
                f"{len(bg)} colours were marked as background; exactly one "
                f"ground colour is required.")
            clean = {}

    return {
        'ok': True,
        'reply': ' '.join(t.strip() for t in text if t.strip())
                 or proposed.get('explanation', ''),
        'assignments': clean,
        'groups': proposed.get('groups') or [],
        'confidence': proposed.get('confidence'),
        'rejected': rejected,
    }
