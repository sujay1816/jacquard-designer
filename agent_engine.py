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

MAX_TOOL_ROUNDS = 8          # tool calls per user turn before we stop
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
        'working': None,        # label map currently being edited
        'undo': [],             # previous working states, most recent last
        'shuttles': None,       # colour index -> shuttle name
        'shuttle_count': 2,
        'weave': {},            # shuttle -> {'pattern', 'n'}
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
    {
        "name": "generate_design",
        "description": (
            "Create a design from scratch instead of converting an upload. "
            "Renders as vector at the exact pin count, so stroke weight is "
            "chosen to be weavable and no detail is ever lost to resolution. "
            "Good for grounds, borders, fills and simple buttas. These are "
            "original geometric and stylised constructions, NOT traditional "
            "motifs — say so if the weaver asks for a named regional style, "
            "and offer the closest geometric equivalent."),
        "input_schema": {
            "type": "object",
            "properties": {
                "motif": {
                    "type": "string",
                    "enum": ["paisley", "lotus", "vine_border", "diamond_jaal",
                             "check_ground", "chevron_border", "dotted_field"],
                },
                "pins": {"type": "integer", "description": "Loom pin count."},
                "cards": {"type": "integer", "description": "Optional height; defaults to the motif ratio."},
                "complexity": {"type": "integer", "description": "paisley: nested outlines, 1-4."},
                "petals": {"type": "integer", "description": "lotus: 5-16."},
                "rings": {"type": "integer", "description": "lotus: 1-3."},
                "repeats": {"type": "integer", "description": "borders: repeats across the width."},
                "cells": {"type": "integer", "description": "grounds: lattice or check divisions."},
                "cols": {"type": "integer", "description": "dotted_field columns."},
                "rows": {"type": "integer", "description": "dotted_field rows."},
                "height": {"type": "integer", "description": "borders: band height in design units."},
            },
            "colours": {"type": "integer", "description": "Threads in the design: 2 for a single thread on the ground, 3 for two threads (zari plus meena). Default 2."},
                "required": ["motif", "pins"],
        },
    },
    {
        "name": "generate_allover",
        "description": (
            "Build an ALL-OVER brocade field — a full body of repeating motifs, "
            "not a single motif. This is what most saree and brocade work "
            "actually is. Layouts: half_drop (the usual body), straight, brick, "
            "banded (motif rows separated by border rules, as on a real "
            "brocade), jaal (diamond lattice with a motif in each cell), and "
            "stripe. Motifs are rebuilt at tile size so the linework stays "
            "weavable however many repeats are asked for."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer", "description": "Loom pin count."},
                "cards": {"type": "integer", "description": "Optional height in cards."},
                "layout": {
                    "type": "string",
                    "enum": ["half_drop", "straight", "brick", "banded", "jaal", "stripe"],
                },
                "motif": {
                    "type": "string",
                    "enum": ["paisley", "lotus", "vine_border", "diamond_jaal",
                             "check_ground", "chevron_border", "dotted_field"],
                    "description": "The repeating unit. paisley or lotus for a butta field.",
                },
                "cols": {"type": "integer", "description": "Motifs across the width, 1-24."},
                "rows": {"type": "integer", "description": "Motif rows down the length, 1-40."},
                "spacing": {"type": "number", "description": "Gap between motifs as a fraction, 0-1.5. Default 0.25."},
                "band_motif": {
                    "type": "string",
                    "enum": ["vine_border", "chevron_border"],
                    "description": "banded layout: which rule separates the motif rows.",
                },
                "band_every": {"type": "integer", "description": "banded layout: rule after every N motif rows."},
                "mirror": {"type": "boolean", "description": "Alternate motifs mirrored."},
            },
            "colours": {"type": "integer", "description": "Threads in the design: 2 for a single thread on the ground, 3 for two threads (zari plus meena). Default 2."},
                "required": ["pins", "layout"],
        },
    },
    {
        "name": "design_options",
        "description": (
            "Given a pin count, report what can actually be designed at that "
            "width: how many of each motif fit across before the internal "
            "detail stops reading, a comfortable count with room to breathe, "
            "and which layouts suit. Call this BEFORE generating when the "
            "weaver has given a pin count but not said what they want — it is "
            "how you design within the loom's real limits instead of guessing "
            "and being corrected."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer"},
                "cards": {"type": "integer"},
            },
            "required": ["pins"],
        },
    },
    {
        "name": "list_motifs",
        "description": "List the motifs that can be generated, with what each is for.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "edit_design",
        "description": (
            "Change the converted design: thicken or thin the lines, clean up "
            "specks, close small gaps, rotate, flip, or invert. Applies to the "
            "current working design and reports how the change affected "
            "fidelity against the original, so a change that ruins the motif "
            "is visible immediately. Use undo_edit to reverse it."),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["thicken", "thin", "clean_specks", "close_gaps",
                             "remove_isolated", "smooth", "invert",
                             "rotate_90", "rotate_180", "rotate_270",
                             "flip_horizontal", "flip_vertical"],
                    "description": (
                        "thicken/thin change stroke weight; clean_specks and "
                        "remove_isolated drop unweavable fragments; close_gaps "
                        "joins broken lines; smooth removes rough edges."),
                },
                "amount": {
                    "type": "integer",
                    "description": "Strength 1-5 for thicken, thin, close_gaps and smooth. Default 1.",
                },
            },
            "required": ["operation"],
        },
    },
    {
        "name": "undo_edit",
        "description": "Reverse the last edit. Use when a change made the design worse.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_shuttles",
        "description": (
            "Assign detected colours to loom shuttles. Colours that are tonal "
            "shades of one thread should share a shuttle. Exactly one colour "
            "is the background (the cloth ground). Never assign more distinct "
            "threads than the loom carries."),
        "input_schema": {
            "type": "object",
            "properties": {
                "shuttle_count": {"type": "integer", "description": "Threads the loom carries, 1-4."},
                "assignments": {
                    "type": "object",
                    "description": ("Colour index (as a string) to shuttle name: "
                                    "zari, meena1, meena2 or background."),
                },
            },
        },
    },
    {
        "name": "set_weave",
        "description": (
            "Set the weave for a shuttle. Higher satin count gives a longer, "
            "glossier float that snags more easily; above about 12 warn the "
            "weaver."),
        "input_schema": {
            "type": "object",
            "properties": {
                "shuttle": {"type": "string", "description": "zari, meena1 or meena2."},
                "pattern": {"type": "string",
                            "description": "satin, twill22, plain_weave, basket, herringbone, dots, diamond, crepe or rib."},
                "n": {"type": "integer", "description": "Satin count 4-16."},
                "flip": {"type": "boolean"},
            },
            "required": ["shuttle"],
        },
    },
    {
        "name": "describe_result",
        "description": (
            "Report the current state of the working design: size, thread "
            "coverage against the original, how many design gaps survive, "
            "stray pixels, longest float, physical size, shuttle assignment "
            "and edits applied. Call this when the weaver asks how it looks or "
            "what has changed."),
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """You convert saree and brocade designs into loom-ready BMP \
files, one per shuttle. You are talking to a weaver or a mill operator, and you \
can carry out the work they ask for — not just describe it.

You can either CONVERT a design the weaver uploads, or GENERATE one from the
motif library. If they ask for a design without uploading anything, use generate_design for a
single motif or generate_allover for a full body — an all-over field of
repeating motifs, which is what most saree and brocade work actually is. Reach
for generate_allover whenever they describe a body, a field, a jaal, an
all-over, or a repeat; use generate_design only when they clearly want one
motif on its own. list_motifs shows both.

When they give a pin count but leave the design to you, call design_options
first. It reports how many of each motif fit across before the detail stops
reading at that width, so you can propose something that works instead of
guessing and being corrected. Then say what you are proposing and why the width
led you there — "480 pins gives room for six paisleys across with space to
breathe; ten would fit but each motif would only get 48 threads and the
interiors would close" — and build it. Design decisions belong to you; just
show your reasoning so the weaver can overrule it.

About generated designs: they are original geometric and stylised
constructions, built as vector at the loom's own pin count so the stroke weight
is weavable by construction. They are NOT traditional regional motifs. If
someone asks for a named tradition — Chola, Banarasi, Kanjivaram — say plainly
that you can build a geometric butta or border in that spirit but not an
authentic period motif, and that a designer is the right answer for anything a
customer will recognise. Do not quietly pass off a generic paisley as Chola
work.

The usual path when converting an upload:
1. Call inspect_design first. Never ask something you could have found out by \
looking.
2. Say what you found in one or two short sentences, then ask how many pins the \
job needs. Suggest the count the image can actually support.
3. Call convert with their pin count.
4. Report the verdict honestly. If detail is lost, say WHAT is lost in craft \
terms — "the fine scrollwork inside the small motifs" beats "12% ink drift". \
Mention a better pin count only if one genuinely exists.
5. Do whatever they ask next: edit_design to change the linework, set_shuttles \
to assign colours to threads, set_weave to change the weave.
6. Call generate_files when they are happy, then tell them the files are ready.

Working with edits:
- edit_design reports fidelity after every change. If it comes back with a \
warning that the change made things worse, SAY SO and offer undo_edit. Never \
report an edit as a success when the tool told you it damaged the design.
- Weavers describe things in their own words. "Lines are too thin" means \
thicken. "Too many specks" means clean_specks. "The motif is breaking up" \
means close_gaps. Map what they say onto the operations rather than asking \
them to learn tool names.
- Prefer amount 1 first and check the result. Small steps are recoverable.
- Use describe_result when they ask how it looks or what has changed.
- Edits invalidate generated files, so regenerate after editing.

What matters:
- Black lifts the thread, white leaves it down. A BMP is an instruction sheet.
- Shuttle count includes the rani ground: 1 is zari alone, 2 is zari plus \
rani, 3 adds meena1, 4 adds meena2. So a design with two colours needs \
shuttle_count 3. If generate_files warns that threads were dropped, tell the \
weaver — do not hand over a file with a colour silently missing.
- Shuttles are hardware. The loom weaves exactly shuttle_count threads. If \
someone asks for a colour there is no thread for, say so and offer the real \
options: replace one, or raise the count if the loom has a spare shuttle.
- Colours that are tonal shades of one thread — cream, mid pink, deep pink of \
the same yarn — belong on ONE shuttle, not three.
- Never claim a conversion is good when the tools reported WARN or FAIL. If \
the source is too small to carry the design, say so plainly; a rescan is the \
only fix and the weaver needs to hear it before the cloth is woven.
- If they ask for something you have no tool for, say what you cannot do \
rather than pretending. Pixel-level drawing belongs in the BMP Editor.

Be brief and concrete. Talk like a colleague on the mill floor. Do not explain \
your tools or narrate what you are about to do — just do it and report."""


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
    """
    Produce the BMPs from the CURRENT working design, honouring any edits,
    shuttle assignment and weave the weaver has asked for.

    Regenerating from the original conversion here would silently discard
    every edit, which is the worst possible failure: the weaver sees a
    confirmation and downloads a file that ignores what they asked for.
    """
    import bmp_engine as be
    from loom_utils import count_long_floats, physical_size

    lm = _working(session)
    if lm is None:
        return {'error': 'Run convert first.'}

    conv = session['conversion']['best']
    try:
        shuttles = max(1, min(4, int(args.get('shuttle_count',
                                              session['shuttle_count']) or 2)))
    except (TypeError, ValueError):
        shuttles = session['shuttle_count']
    session['shuttle_count'] = shuttles

    lm = np.asarray(lm)
    n_labels = int(lm.max()) + 1

    if session.get('shuttles'):
        assignments = {int(k): v for k, v in session['shuttles'].items()}
    else:
        # Default: largest region is the ground, the rest take shuttles in
        # descending area order. Chosen here rather than by the model so the
        # loom's thread budget cannot be exceeded.
        #
        # shuttle_count follows the app's own convention, in which the rani
        # ground occupies a slot: 1 = zari alone, 2 = zari + rani, 3 = zari +
        # rani + meena1, 4 adds meena2. So the number of DESIGN threads is one
        # fewer than the shuttle count above 2. Treating every slot as a design
        # thread silently dropped the second colour of a two-thread design onto
        # the background, producing a file with the meena simply missing.
        names = ['zari', 'meena1', 'meena2']
        design_slots = 1 if shuttles <= 2 else min(shuttles - 1, len(names))
        counts = np.bincount(lm.ravel(), minlength=n_labels)
        order = list(np.argsort(-counts))
        assignments = {int(order[0]): 'background'}
        for i, idx in enumerate(order[1:]):
            assignments[int(idx)] = names[i] if i < design_slots else 'background'
        dropped = max(0, (n_labels - 1) - design_slots)

    dropped = locals().get('dropped', 0)
    default = {'n': 8, 'flip': False}
    satin = {n: dict(session['weave'].get(n, default)) for n in
             ('zari', 'meena1', 'meena2')}
    name = str(args.get('design_name') or session['filename'])[:40]

    files = be.generate_bmps(
        image=session['image'], pins=conv['pins'], cards=conv['cards'],
        shuttle_count=shuttles, color_assignments=assignments,
        satin_settings=satin, design_name=name,
        label_map=lm, stroke_mode=False, reed=60)

    summary, worst = [], 0
    for fn, data in sorted(files.items()):
        info = be.verify_bmp(data)
        mask = np.array(Image.open(io.BytesIO(data)).convert('L')) < 128
        _, longest = count_long_floats(mask, 12)
        worst = max(worst, longest)
        summary.append({'file': fn, 'bytes': len(data),
                        'clean_1bit': info['is_clean'],
                        'threads_up': info['pure_black'],
                        'longest_float': longest})

    session['files'] = files
    size = physical_size(conv['pins'], conv['cards'], 60)
    rep = _rescore(session)
    return {
        'ready': True, 'files': summary,
        'pins': conv['pins'], 'cards': conv['cards'],
        'physical_size_in': f"{size['width_in']} x {size['height_in']} at 60 reed",
        'edits_applied': len(session['undo']),
        'fidelity_verdict': rep['verdict'],
        'longest_float': worst,
        'float_warning': (
            f"Longest float is {worst} picks — check with the loom operator "
            f"before running." if worst > 30 else None),
        'threads_dropped': dropped or None,
        'thread_warning': (
            f"The design has {dropped} more thread"
            f"{'s' if dropped != 1 else ''} than this loom carries, so "
            f"{'they were' if dropped != 1 else 'it was'} folded into the "
            f"ground. Raise shuttle_count to keep "
            f"{'them' if dropped != 1 else 'it'}." if dropped else None),
    }


def _working(session):
    """Current design being edited, falling back to the conversion result."""
    if session.get('working') is None:
        conv = session.get('conversion')
        if not conv or not conv.get('best'):
            return None
        session['working'] = np.asarray(conv['best']['label_map']).copy()
    return session['working']


def _rescore(session):
    """Fidelity of the working design against the uploaded image."""
    from fidelity import fidelity_report
    lm = _working(session)
    if lm is None:
        return None
    return fidelity_report(session['image'], lm > 0)


def _tool_edit(session, args):
    from scipy import ndimage

    lm = _working(session)
    if lm is None:
        return {'error': 'Convert the design first.'}

    op = str(args.get('operation', '')).strip()
    try:
        amount = max(1, min(5, int(args.get('amount', 1) or 1)))
    except (TypeError, ValueError):
        amount = 1

    before = _rescore(session)
    prev = lm.copy()
    design = lm > 0
    bg = 0
    # Preserve which label each design cell had, so multi-colour designs keep
    # their shuttle separation through a shape edit.
    labels = lm.copy()

    def disk(r):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return (yy * yy + xx * xx) <= r * r

    if op == 'thicken':
        grown = ndimage.binary_dilation(design, structure=disk(amount))
        # New cells take the label of the nearest existing design cell.
        idx = ndimage.distance_transform_edt(~design, return_distances=False,
                                             return_indices=True)
        new = labels[tuple(idx)]
        lm = np.where(grown, new, bg).astype(np.uint8)
    elif op == 'thin':
        kept = ndimage.binary_erosion(design, structure=disk(amount))
        lm = np.where(kept, labels, bg).astype(np.uint8)
    elif op in ('clean_specks', 'remove_isolated'):
        min_size = 2 if op == 'remove_isolated' else max(2, amount * 3)
        keep = np.zeros_like(design)
        for c in np.unique(labels):
            if c == bg:
                continue
            m = labels == c
            l2, n = ndimage.label(m, structure=np.ones((3, 3)))
            if n == 0:
                continue
            sizes = np.bincount(l2.ravel())
            sizes[0] = 0
            big = np.isin(l2, np.where(sizes >= min_size)[0])
            keep |= big
        lm = np.where(keep, labels, bg).astype(np.uint8)
    elif op == 'close_gaps':
        closed = ndimage.binary_closing(design, structure=disk(amount))
        idx = ndimage.distance_transform_edt(~design, return_distances=False,
                                             return_indices=True)
        lm = np.where(closed, labels[tuple(idx)], bg).astype(np.uint8)
    elif op == 'smooth':
        opened = ndimage.binary_opening(design, structure=disk(amount))
        lm = np.where(opened, labels, bg).astype(np.uint8)
    elif op == 'invert':
        # Swap design and ground: the largest class becomes design and vice
        # versa. Only meaningful on two-class designs.
        lm = np.where(design, bg, 1).astype(np.uint8)
    elif op == 'rotate_90':
        lm = np.rot90(lm, 1).astype(np.uint8)
    elif op == 'rotate_180':
        lm = np.rot90(lm, 2).astype(np.uint8)
    elif op == 'rotate_270':
        lm = np.rot90(lm, 3).astype(np.uint8)
    elif op == 'flip_horizontal':
        lm = np.fliplr(lm).astype(np.uint8)
    elif op == 'flip_vertical':
        lm = np.flipud(lm).astype(np.uint8)
    else:
        return {'error': f'No such operation: {op}'}

    session['undo'].append(prev)
    session['undo'] = session['undo'][-10:]
    session['working'] = lm
    session['files'] = None            # generated files are now stale
    after = _rescore(session)

    result = {'applied': op, 'amount': amount,
              'thread_drift_pct': after['ink_drift_pct'],
              'design_gaps': after['output_white_regions'],
              'design_gaps_source': after['source_white_regions'],
              'isolated_cells': after['isolated_cells'],
              'verdict': after['verdict'],
              'notes': after['messages']}
    # Say plainly when an edit made things worse, so the model does not report
    # success on a change that damaged the design.
    if before:
        worse = (abs(after['ink_drift_pct']) > abs(before['ink_drift_pct']) + 8
                 or after['verdict'] == 'fail' and before['verdict'] != 'fail')
        if worse:
            result['warning'] = (
                f"This made fidelity worse (thread drift "
                f"{before['ink_drift_pct']:+.0f}% -> {after['ink_drift_pct']:+.0f}%). "
                f"Tell the weaver and offer undo_edit.")
    return result


def _tool_undo(session, args):
    if not session.get('undo'):
        return {'error': 'Nothing to undo.'}
    session['working'] = session['undo'].pop()
    session['files'] = None
    rep = _rescore(session)
    return {'undone': True, 'verdict': rep['verdict'],
            'thread_drift_pct': rep['ink_drift_pct'],
            'design_gaps': rep['output_white_regions'],
            'edits_remaining': len(session['undo'])}


def _tool_set_shuttles(session, args):
    lm = _working(session)
    if lm is None:
        return {'error': 'Convert the design first.'}

    from assistant_engine import validate_patch

    try:
        count = max(1, min(4, int(args.get('shuttle_count', session['shuttle_count']))))
    except (TypeError, ValueError):
        count = session['shuttle_count']

    raw = args.get('assignments')
    n_labels = int(np.asarray(lm).max()) + 1
    if not raw:
        session['shuttle_count'] = count
        return {'shuttle_count': count, 'assignments': session.get('shuttles'),
                'colours_detected': n_labels}

    # Same validator the settings assistant uses: shuttle budget, valid names,
    # and colour indices that actually exist.
    checked = validate_patch({'color_assignments': raw},
                             {'shuttle_count': count, 'detected_colors': n_labels})
    if checked['rejected'] or not checked['patch'].get('color_assignments'):
        return {'error': ' '.join(checked['rejected']) or 'Invalid assignment.'}

    assigned = checked['patch']['color_assignments']
    if sum(1 for v in assigned.values() if v == 'background') != 1:
        return {'error': 'Exactly one colour must be the background.'}

    session['shuttles'] = {int(k): v for k, v in assigned.items()}
    session['shuttle_count'] = count
    session['files'] = None
    return {'shuttle_count': count, 'assignments': assigned,
            'colours_detected': n_labels}


def _tool_set_weave(session, args):
    from bmp_engine import FILL_PATTERNS

    shuttle = str(args.get('shuttle', '')).strip()
    if shuttle not in ('zari', 'meena1', 'meena2'):
        return {'error': "Shuttle must be zari, meena1 or meena2."}

    entry = dict(session['weave'].get(shuttle, {'pattern': 'satin', 'n': 8, 'flip': False}))
    if 'pattern' in args:
        pat = str(args['pattern']).strip()
        if pat not in FILL_PATTERNS:
            return {'error': f"Unknown weave '{pat}'. Available: {', '.join(sorted(FILL_PATTERNS))}"}
        entry['pattern'] = pat
    if 'n' in args:
        try:
            entry['n'] = max(4, min(16, int(args['n'])))
        except (TypeError, ValueError):
            return {'error': 'Satin count must be a whole number.'}
    if 'flip' in args:
        entry['flip'] = bool(args['flip'])

    session['weave'][shuttle] = entry
    session['files'] = None
    out = {'shuttle': shuttle, **entry}
    if entry['n'] > 12:
        out['warning'] = (f"Satin {entry['n']} leaves long floats — glossy, but "
                          f"they snag more easily.")
    return out


def _tool_describe(session, args):
    from loom_utils import physical_size

    lm = _working(session)
    if lm is None:
        return {'error': 'Nothing converted yet.'}
    rep = _rescore(session)
    conv = session['conversion']['best']
    size = physical_size(conv['pins'], conv['cards'], 60)
    return {
        'pins': conv['pins'], 'cards': conv['cards'],
        'physical_size_in': f"{size['width_in']} x {size['height_in']} at 60 reed",
        'verdict': rep['verdict'],
        'thread_drift_pct': rep['ink_drift_pct'],
        'design_gaps': rep['output_white_regions'],
        'design_gaps_source': rep['source_white_regions'],
        'isolated_cells': rep['isolated_cells'],
        'shuttle_count': session['shuttle_count'],
        'shuttles': session.get('shuttles'),
        'weave': session.get('weave') or 'satin 8 (default)',
        'edits_applied': len(session['undo']),
        'files_ready': bool(session.get('files')),
        'notes': rep['messages'],
    }


def _tool_generate_allover(session, args):
    """
    Build a full all-over brocade field and make it the working design.

    Distinct from generate_design, which makes one motif. Real saree and
    brocade work is a field: motif rows with band rules, a jaal lattice, a
    half-drop repeat across the whole body.
    """
    import motif_library as ml
    from auto_convert import auto_convert

    try:
        pins = int(args.get('pins', 0))
    except (TypeError, ValueError):
        return {'error': 'Pins must be a whole number.'}
    if not (10 <= pins <= 2640):
        return {'error': 'Pins must be between 10 and 2640.'}

    layout = str(args.get('layout', 'half_drop')).strip()
    if layout not in ml.ALLOVER_LAYOUTS:
        return {'error': f"Unknown layout '{layout}'. Available: "
                         f"{', '.join(sorted(ml.ALLOVER_LAYOUTS))}"}

    motif = str(args.get('motif', 'paisley')).strip()
    if motif not in ml.MOTIFS:
        return {'error': f"Unknown motif '{motif}'."}

    cards = args.get('cards')
    try:
        cards = int(cards) if cards else None
    except (TypeError, ValueError):
        cards = None
    if cards is not None and not (10 <= cards <= 6000):
        return {'error': 'Cards must be between 10 and 6000.'}

    kw = {k: v for k, v in args.items()
          if k in ('cols', 'rows', 'spacing', 'band_motif', 'band_every',
                   'mirror', 'colours')
          and v is not None}
    try:
        svg = ml.allover(pins, layout=layout, motif=motif, cards=cards, **kw)
        img = ml.render(svg, pins)
    except Exception as e:
        return {'error': f'Could not build that field: {e}'}

    if img.size[1] > 6000:
        return {'error': f'That would be {img.size[1]} cards, over the 6000 limit. '
                         f'Reduce rows.'}

    session['image'] = img
    session['filename'] = f'{layout}_{motif}_{pins}'
    session['working'] = None
    session['undo'] = []
    session['shuttles'] = None
    session['weave'] = {}
    session['files'] = None

    n_colors = max(2, min(4, int(args.get('colours', 2) or 2)))
    result = auto_convert(img, pins=pins, n_colors=n_colors)
    if not result.get('best'):
        return {'error': result.get('summary', 'That field would not convert.')}
    session['conversion'] = result
    best, rep = result['best'], result['best']['report']

    return {
        'created': f'{layout} field of {motif}',
        'threads': n_colors - 1,
        'pins': best['pins'], 'cards': best['cards'],
        'cols': kw.get('cols', 5), 'rows': kw.get('rows', 6),
        'verdict': result['verdict'],
        'thread_drift_pct': rep['ink_drift_pct'],
        'design_gaps': rep['output_white_regions'],
        'note': ('Motifs were rebuilt at tile size, so the linework is weavable '
                 'at this repeat count. Editable like any design.'),
    }


def _tool_design_options(session, args):
    import motif_library as ml
    try:
        pins = int(args.get('pins', 0))
    except (TypeError, ValueError):
        return {'error': 'Pins must be a whole number.'}
    if not (10 <= pins <= 2640):
        return {'error': 'Pins must be between 10 and 2640.'}
    cards = args.get('cards')
    try:
        cards = int(cards) if cards else None
    except (TypeError, ValueError):
        cards = None
    return ml.design_options(pins, cards)


def _tool_list_motifs(session, args):
    import motif_library as ml
    return {'motifs': [{'name': k, 'description': v[1]} for k, v in sorted(ml.MOTIFS.items())],
            'allover_layouts': [{'name': k, 'description': v}
                                for k, v in sorted(ml.ALLOVER_LAYOUTS.items())],
            'note': ('These are original parametric constructions, not traditional '
                     'regional motifs. They suit grounds, borders, fills and simple '
                     'buttas.')}


def _tool_generate_design(session, args):
    """
    Build a design from the motif library and make it the working design.

    Rendered from vector at the loom's own resolution, so the stroke weight is
    chosen FOR the pin count rather than inherited from whatever a scan
    happened to contain. Every resolution failure this app handles elsewhere —
    thickening, closed gaps, lost interiors — cannot arise here.
    """
    import motif_library as ml
    from auto_convert import auto_convert

    motif = str(args.get('motif', '')).strip()
    if motif not in ml.MOTIFS:
        return {'error': f"Unknown motif '{motif}'. Available: "
                         f"{', '.join(sorted(ml.MOTIFS))}"}
    try:
        pins = int(args.get('pins', 0))
    except (TypeError, ValueError):
        return {'error': 'Pins must be a whole number.'}
    if not (10 <= pins <= 2640):
        return {'error': 'Pins must be between 10 and 2640.'}

    cards = args.get('cards')
    try:
        cards = int(cards) if cards else None
    except (TypeError, ValueError):
        cards = None
    if cards is not None and not (10 <= cards <= 6000):
        return {'error': 'Cards must be between 10 and 6000.'}

    try:
        svg = ml.build_svg(motif, pins, **{k: v for k, v in args.items()
                                           if k not in ('motif', 'pins', 'cards')})
        img = ml.render(svg, pins, cards)
    except Exception as e:
        return {'error': f'Could not build that design: {e}'}

    # The generated image becomes the design under discussion, so every
    # existing tool — edit, shuttles, weave, generate_files — works on it
    # unchanged.
    session['image'] = img
    session['filename'] = f'{motif}_{pins}'
    session['working'] = None
    session['undo'] = []
    session['shuttles'] = None
    session['weave'] = {}
    session['files'] = None

    n_colors = max(2, min(4, int(args.get('colours', 2) or 2)))
    result = auto_convert(img, pins=pins, n_colors=n_colors)
    if not result.get('best'):
        return {'error': result.get('summary', 'Generated design would not convert.')}
    session['conversion'] = result
    best, rep = result['best'], result['best']['report']

    return {
        'created': motif, 'pins': best['pins'], 'cards': best['cards'],
        'threads': n_colors - 1,
        'verdict': result['verdict'],
        'thread_drift_pct': rep['ink_drift_pct'],
        'design_gaps': rep['output_white_regions'],
        'note': ('Built as vector at the loom resolution, so stroke weight is '
                 'weavable by construction. It can now be edited like any '
                 'uploaded design.'),
    }



_DISPATCH = {
    'inspect_design': _tool_inspect,
    'convert': _tool_convert,
    'generate_files': _tool_generate,
    'edit_design': _tool_edit,
    'undo_edit': _tool_undo,
    'set_shuttles': _tool_set_shuttles,
    'set_weave': _tool_set_weave,
    'describe_result': _tool_describe,
    'generate_design': _tool_generate_design,
    'list_motifs': _tool_list_motifs,
    'generate_allover': _tool_generate_allover,
    'design_options': _tool_design_options,
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
    # Thinking is disabled explicitly. Sonnet 5 turns adaptive thinking on by
    # default, which has two costs here: thinking tokens count against
    # max_tokens (so a cap tuned without it can truncate the visible answer),
    # and thinking blocks enter the assistant turns that this loop feeds back
    # as history, which is extra state to preserve correctly for no benefit.
    # The work in this agent is tool dispatch and short explanations, not
    # reasoning the model needs scratch space for.
    body = json.dumps({
        "model": model_id(),
        "thinking": {"type": "disabled"},
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
        # Read the body. The API explains exactly what it rejected — an unknown
        # model, a malformed tool schema, a bad message shape — and discarding
        # that in favour of a bare status code makes the failure undiagnosable.
        detail = f'HTTP {e.code}'
        try:
            payload = json.loads(e.read().decode())
            msg = (payload.get('error') or {}).get('message')
            if msg:
                detail = msg
        except Exception:
            pass
        if e.code in (401, 403):
            detail = f'{detail} — check the API key'
        return None, f'Assistant unavailable: {detail}'
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
