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
import zipfile

import numpy as np
from PIL import Image

import llm

MAX_TOOL_ROUNDS = 14         # tool calls per user turn before we stop
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


def new_session(image=None, filename='design'):
    """
    Open a session, with or without an uploaded image.

    image=None is a design-from-scratch session. It used to be faked with an
    8x8 white PNG so that the upload path could be reused, which meant the
    weaver was shown a white square as their design and inspect_design was
    offered an image with nothing in it.
    """
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
        'plan': None,           # visible step list for the current job
        'clipboard': None,      # region lifted by region/copy
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
        "name": "design",
        "description": (
            "Design a whole panel from a brief and score it against the loom. "
            "This is the main design tool — use it whenever the weaver wants "
            "something made rather than converted. Give it INTENT, not "
            "geometry: the width (pins, or width_in with a reed), how the "
            "cloth should feel, how many threads. It works out the motif, the "
            "repeat, the border widths and the row count itself, builds the "
            "side borders and body together, converts it and reports the "
            "verdict. Do not ask the weaver for cols, rows, spacing or layout "
            "— those are yours to decide and theirs to overrule."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer", "description": "Loom width in threads."},
                "width_in": {"type": "number",
                             "description": "Finished width in inches. Needs reed. Use instead of pins."},
                "reed": {"type": "number",
                         "description": "Reed count, ends per inch — typically 60, 80 or 100. Ask if unknown; it decides what the pin count physically measures."},
                "picks": {"type": "number", "description": "Picks per inch. Defaults to the reed."},
                "cards": {"type": "integer", "description": "Panel height in cards."},
                "length_in": {"type": "number", "description": "Finished length in inches."},
                "feel": {"type": "string",
                         "description": "How the cloth should read: rich, dense, traditional, open, light, minimal, geometric, formal."},
                "threads": {"type": "integer",
                            "description": "Ink threads excluding the ground, 1-3."},
                "motif": {"type": "string",
                          "description": "Force a specific motif. Omit to let the design choose what fits."},
                "borders": {"type": "boolean",
                            "description": "Side borders up both selvedges. Default true."},
                "pallu": {"type": "boolean",
                          "description": "Cross border across the width at the foot."}
            },
        },
    },
    {
        "name": "canvas_info",
        "description": (
            "Report the cloth: size in threads and cards, how much carries "
            "thread, where the design actually sits, and how much blank cloth "
            "is around it. Call this before any canvas or region work so you "
            "are moving real coordinates rather than guessing. It also lists "
            "the region names you can use."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "canvas",
        "description": (
            "Change the cloth itself — its size, or where the design sits on "
            "it. `extend` adds bare cloth on any side. `crop` keeps a region "
            "and discards the rest. `trim` cuts away blank cloth around the "
            "design. `resize` re-mounts the design on a canvas of an exact "
            "size WITHOUT resampling it. `scale` resamples the design to a "
            "different thread count — that one loses detail going down and "
            "blocks it going up, so say so. `move` shifts the design, with "
            "wrap for an all-over repeat. `centre` centres it. `mirror` folds "
            "one half onto the other to make the panel symmetric, which is how "
            "a matched border pair is built."),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string",
                              "enum": ["extend", "crop", "trim", "resize", "scale",
                                       "move", "centre", "mirror"]},
                "left": {"type": "integer", "description": "extend: threads to add."},
                "right": {"type": "integer"},
                "top": {"type": "integer", "description": "extend: cards to add."},
                "bottom": {"type": "integer"},
                "margin": {"type": "integer", "description": "trim: cloth to leave."},
                "pins": {"type": "integer", "description": "resize/scale: target width."},
                "cards": {"type": "integer", "description": "resize/scale: target height."},
                "anchor": {"type": "string",
                           "description": "resize: left, center, right, top, bottom."},
                "dx": {"type": "integer", "description": "move: threads right, negative for left."},
                "dy": {"type": "integer", "description": "move: cards down, negative for up."},
                "wrap": {"type": "boolean",
                         "description": "move: roll round the edges. Use for all-over repeats."},
                "axis": {"type": "string", "description": "mirror: vertical or horizontal."},
                "region": {"type": "string", "description": "crop: a named region."},
                "box": {"type": "array", "items": {"type": "integer"},
                        "description": "crop: [x0, y0, x1, y1] in threads and cards."},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "region",
        "description": (
            "Work on one part of the cloth and leave the rest alone. `clear` "
            "erases back to bare cloth. `copy` and `paste` move a piece "
            "elsewhere. `mirror`, `flip_vertical`, `rotate_180` and `invert` "
            "transform just that part. `tile` repeats a region across the "
            "whole cloth, which is how one drawn butta becomes a field. "
            "`weave_fill` textures a region with satin, twill, basket and the "
            "rest. `stamp` draws a motif from the library straight onto the "
            "canvas at a position and size you choose. Name a region or give a "
            "box; call canvas_info first if you are unsure of the coordinates."),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string",
                              "enum": ["clear", "copy", "paste", "mirror",
                                       "flip_vertical", "rotate_180", "invert",
                                       "tile", "weave_fill", "stamp"]},
                "region": {"type": "string",
                           "description": "A named region — top, bottom, left_border, body, pallu, centre and so on."},
                "box": {"type": "array", "items": {"type": "integer"},
                        "description": "[x0, y0, x1, y1] in threads and cards from the top-left."},
                "x": {"type": "integer", "description": "paste/stamp: position across."},
                "y": {"type": "integer", "description": "paste/stamp: position down."},
                "mode": {"type": "string",
                         "description": ("'blend' lays the piece over what is there, showing "
                                         "the ground through its gaps. 'over' replaces the "
                                         "rectangle entirely. Default blend.")},
                "cols": {"type": "integer", "description": "tile: repeats across."},
                "rows": {"type": "integer", "description": "tile: repeats down."},
                "pattern": {"type": "string",
                            "description": ("weave_fill: satin, satin_inv, plain_weave, "
                                            "twill22, twill31, basket, honeycomb, diamond, "
                                            "crepe, rib, herringbone, dots, diagonal, "
                                            "crosshatch. 'twill', 'plain' and 'matt' are "
                                            "understood too.")},
                "n": {"type": "integer", "description": "weave_fill: float length, 4-16."},
                "design_only": {"type": "boolean",
                                "description": ("weave_fill: true textures only the thread that "
                                                "is there (a motif); false fills the whole "
                                                "rectangle (a patterned ground). Default true.")},
                "motif": {"type": "string", "description": "stamp: motif name."},
                "width_threads": {"type": "integer",
                                  "description": "stamp: how many threads wide to draw it."},
            },
            "required": ["operation"],
        },
    },
    {
        "name": "plan_work",
        "description": (
            "Write down what you are about to do, as 2-6 short steps, then "
            "mark them done as you go. Call this FIRST on any job that will "
            "take more than a couple of tool calls — designing from a brief, "
            "converting and finishing an upload. The weaver can see the plan "
            "while you work, so they know whether you are nearly there or have "
            "lost the thread. Write steps in their language: 'work out the "
            "finished width', not 'call loom_geometry'. Call again with `done` "
            "to tick steps off."),
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"},
                          "description": "The plan, 2-6 steps. Omit when ticking off."},
                "done": {"type": "array", "items": {"type": "integer"},
                         "description": "Zero-based step numbers now finished."},
            },
        },
    },
    {
        "name": "auto_design",
        "description": (
            "Work out the best weavable design for a brief on your own. "
            "Builds a candidate, measures it against the loom, tries the "
            "changes that might improve it, keeps what worked, and repeats. "
            "PREFER THIS over `design` whenever the weaver has given you a "
            "brief and left the rest to you — it is the difference between "
            "handing over a first attempt and handing over a worked answer. "
            "Returns the trail of what it tried, which you should summarise in "
            "plain terms."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer"},
                "width_in": {"type": "number", "description": "Finished width in inches; needs reed."},
                "reed": {"type": "number", "description": "Ends per inch — 60, 80, 100."},
                "cards": {"type": "integer"},
                "length_in": {"type": "number"},
                "feel": {"type": "string",
                         "description": "rich, dense, traditional, open, light, minimal, geometric, formal."},
                "threads": {"type": "integer", "description": "Ink threads, 1-3."},
                "motif": {"type": "string", "description": "Force one. Omit to let it choose."},
                "borders": {"type": "boolean"},
                "pallu": {"type": "boolean"},
                "effort": {"type": "integer",
                           "description": "Improvement rounds, 1-8. Default 4. Higher is slower."}
            },
        },
    },
    {
        "name": "look_at_design",
        "description": (
            "Look at the current design yourself. Use it after building "
            "something, before handing it over, and after any refinement you "
            "are unsure about. Judge COMPOSITION — border balance, whether the "
            "field reads as cloth, whether the repeat is obtrusive. Do not "
            "judge line quality or resolution from it; the fidelity score "
            "already measured those and is more reliable than a thumbnail. If "
            "the backend has no vision this returns an error, which is not a "
            "problem — carry on with the measurements."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_designs",
        "description": (
            "See the explored candidates side by side. Use after "
            "explore_designs when the scores are close and the choice is "
            "really about how the cloth looks."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "explore_designs",
        "description": (
            "Build several alternative designs from the current one and score "
            "each, ranked. Use when the weaver is undecided, asks to see "
            "options, or the first attempt was not quite right. Describe the "
            "candidates in craft terms and let them pick — then call "
            "choose_design."),
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "How many, 2-4."},
                "pins": {"type": "integer",
                         "description": "Only if no design exists yet."},
                "reed": {"type": "number"},
                "feel": {"type": "string"},
            },
        },
    },
    {
        "name": "choose_design",
        "description": "Adopt one of the explored candidates by its index.",
        "input_schema": {
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    },
    {
        "name": "refine_design",
        "description": (
            "Adjust a generated design and rebuild it from vector at full "
            "quality. Use this rather than edit_design for anything you "
            "generated — edit_design pushes pixels around, this changes the "
            "design itself. Map what the weaver says onto a change: 'too busy' "
            "is more_open, 'looks empty' is denser, 'motifs too small' is "
            "fewer_motifs."),
        "input_schema": {
            "type": "object",
            "properties": {
                "change": {"type": "string",
                           "description": ("One of: more_open, denser, fewer_motifs, "
                                           "more_motifs, taller, shorter, wider_border, "
                                           "narrower_border.")},
            },
            "required": ["change"],
        },
    },
    {
        "name": "loom_geometry",
        "description": (
            "Convert between threads and finished cloth size at a given reed, "
            "in either direction. Call this before proposing a pin count when "
            "the weaver mentions a width in inches or centimetres, or when "
            "they ask how big something will come out. A pin count means "
            "nothing physical without the reed: 480 pins is 8 inches at reed "
            "60 and 4.8 at reed 100."),
        "input_schema": {
            "type": "object",
            "properties": {
                "pins": {"type": "integer"},
                "cards": {"type": "integer"},
                "reed": {"type": "number"},
                "picks": {"type": "number"},
                "width_in": {"type": "number",
                             "description": "Give this to get the pin count back."},
                "length_in": {"type": "number"},
            },
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

SYSTEM_PROMPT = """You are the designer for a jacquard weaving mill. You are \
talking to a weaver or a mill operator, and you do the work — you do not hand \
them a form to fill in.

HOW YOU WORK

Take a brief and run it to a finished answer before coming back. That means:
work out the geometry, build the design, look at it, fix what is wrong, and
present something finished with the reasoning attached. Do not narrate each
step as you go and do not stop halfway to ask permission to continue.

The loop you should be running, most of the time:

  1. `loom_geometry` if they mentioned a size in inches or the finished size
     matters. A pin count means nothing physical without the reed — 480 pins
     is 8 inches at reed 60 and 4.8 at reed 100.
  2. `auto_design` with the brief. It builds a candidate, measures it against
     the loom, tries improvements, keeps what worked, and repeats. Prefer it
     over `design` whenever the weaver has left the design to you.
  3. `look_at_design`. The score tells you the linework survived; it cannot
     tell you the borders overpower the body or the repeat reads as wallpaper.
     Look, and fix what you see with `refine_design`.
  4. Present it: what you made, why that shape, what it will measure, and what
     the conversion cost. Then the next question, or the files.

Use `explore_designs` and `compare_designs` when the choice is genuinely a
matter of taste rather than measurement — then let them pick.

WHAT YOU DECIDE, AND WHAT YOU ASK

Yours: motif, layout, how many across, border widths, row counts, spacing.
Never ask a weaver for cols, rows, spacing or layout. Asking makes them do your
job with less information than you have.

Theirs, and worth asking for when missing: how wide, the reed, how many
threads the loom has, how the cloth should feel in their words.

Ask at most one or two questions before building something. "I need a saree
body, 800 pins, reed 80" is enough — build it. If they are vague, pick sensible
defaults, build it, say what you chose, and offer to change it. A design on the
table beats a questionnaire.

WORKING ON THE CLOTH ITSELF

You can change the cloth, not just the linework. `canvas_info` first — it tells
you the size, where the design sits and how much blank cloth surrounds it, so
you are moving real coordinates instead of guessing.

  * `canvas` changes the cloth: extend it for a wider loom, crop or trim it,
    re-mount it at an exact size, move the design off a fold, mirror one half
    onto the other to build a matched border pair.
  * `region` works on part of it: clear a panel and put something else there,
    copy a butta and paste it, tile one motif into a field, texture an area
    with satin or twill, stamp a motif from the library at a size and place
    you choose.

Two distinctions worth keeping straight, because getting them wrong quietly
ruins cloth:
  * `resize` re-mounts the design on a different canvas. `scale` resamples the
    design itself, which loses detail going down and blocks it going up. If a
    weaver asks to fit a design to a different loom, ask which they meant.
  * `blend` lays a piece over what is there and lets the ground show through
    its gaps. `over` replaces the whole rectangle. Stamping a butta with `over`
    punches a bare rectangle through a lattice.

Canvas work is undoable like any edit. Say what changed and by how much, and
check the result with look_at_design rather than assuming.

REFINING

`refine_design` for anything you generated — it rebuilds from the vector at
full quality. `edit_design` is for uploaded images and pushes pixels around;
do not reach for it on a generated design. Translate what they say: "too busy"
is more_open, "looks empty" is denser, "motifs are too small" is fewer_motifs.
Change one thing, look at it, then report.

CONVERTING AN UPLOAD

`inspect_design` first — never ask what you could have looked at. Say what you
found in a sentence, suggest a pin count the image can support, `convert`, then
report honestly. If detail is lost say WHAT is lost: "the fine scrollwork
inside the small motifs" beats "12% ink drift". Then `set_shuttles`,
`set_weave`, `generate_files`.

HONESTY — the part that matters most

  * Never call a conversion good when a tool reported warn or fail. If the
    source is too small to carry the design, say so plainly: a rescan is the
    only fix, and the weaver needs to hear it before the cloth is on the loom.
  * If a tool returns a warning, lead with it. A refinement that made things
    worse gets said out loud with an offer to go back — not buried under a
    description of the new version.
  * Report what the trail actually shows. If auto_design improved drift from
    26% to 20%, that is an improvement and still a warn; say both.
  * Generated motifs are original geometric constructions, not traditional
    regional work. Asked for Chola, Banarasi or Kanjivaram: say you can build
    something in that spirit but not an authentic period motif, and that a
    designer is the right answer for anything a customer will recognise. Never
    pass off a generic paisley as Chola work.
  * When you looked at a design, say what you saw — including when it looked
    fine. Do not invent a flaw to seem thorough, and do not claim to have
    looked when the backend had no vision.
  * If you have no tool for what they want, say what you cannot do and what you
    can. Do not improvise around it.

THE LOOM

Black lifts the thread, white leaves it down; a BMP is an instruction sheet.
Shuttle count includes the rani ground: 1 is zari alone, 2 adds rani, 3 adds
meena1, 4 adds meena2 — so a two-colour design needs shuttle_count 3. Shuttles
are hardware: if they ask for a colour there is no thread for, say so and give
the real options rather than silently dropping it. Tonal shades of one yarn —
cream, mid pink, deep pink — belong on ONE shuttle.

TONE

Talk like a colleague who knows looms. Short sentences. No parameter dumps, no
bulleted settings lists, no restating tool arguments back at them. Two or three
sentences about what you made and why, then the next question or the files.
"""


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


def _spec_of(session):
    import design_studio as ds
    d = session.get('spec')
    return ds.LayoutSpec(**d) if d else None


def _adopt(session, spec, conv, img, name=None):
    """Make a spec the working design and clear everything derived from it."""
    import design_studio as ds
    session['spec'] = spec.dict()
    session['image'] = img
    session['conversion'] = conv
    session['filename'] = name or f'{spec.body_motif}_{spec.pins}'
    session['working'] = None
    session['undo'] = []
    session['shuttles'] = None
    session['weave'] = {}
    session['files'] = None
    session['variants'] = None
    return ds.describe(spec)


def _tool_design(session, args):
    """
    Design a panel from a brief — plan, compose, convert and score in one call.

    This exists because the old path made the weaver supply layout, motif,
    cols, rows and spacing before anything could be built. Those are the
    designer's decisions, and the numbers needed to make them well (how many
    threads a motif needs, what the reed makes the cloth measure) live in this
    codebase, not in the weaver's head. So the brief carries intent — width,
    reed, how the cloth should feel — and the geometry is worked out here.
    """
    import design_studio as ds
    from auto_convert import auto_convert

    pins, reed = args.get('pins'), args.get('reed')
    if not pins and not args.get('width_in'):
        return {'error': 'I need either a pin count or a finished width in inches.'}
    if args.get('width_in') and not pins:
        pins = ds.geometry(None, reed=reed, width_in=args['width_in'])['pins']

    motif = args.get('motif')
    if motif and motif not in ml_motifs():
        return {'error': f"Unknown motif '{motif}'. Available: "
                         f"{', '.join(sorted(ml_motifs()))}"}

    try:
        p = ds.plan(pins=pins, reed=reed, picks=args.get('picks'),
                    cards=args.get('cards'), feel=args.get('feel'),
                    threads=int(args.get('threads', 2) or 2),
                    motif=motif,
                    borders=args.get('borders', True) is not False,
                    pallu=bool(args.get('pallu')),
                    length_in=args.get('length_in'))
        spec = ds.LayoutSpec(**p['spec'])
        img = ds.render(spec)
    except Exception as e:
        return {'error': f'Could not build that: {e}'}

    if img.size[1] > 6000:
        return {'error': f'That comes to {img.size[1]} cards, over the 6000 limit.'}

    conv = auto_convert(img, pins=spec.pins,
                        n_colors=max(2, min(4, spec.threads + 1)))
    if not conv.get('best'):
        return {'error': conv.get('summary', 'That design would not convert.')}

    summary = _adopt(session, spec, conv, img)
    rep = conv['best']['report']
    return {
        'design': summary,
        'why_this': p['why'],
        'geometry': {k: v for k, v in p['geometry'].items() if k != 'raw'},
        'threads': spec.threads,
        'verdict': conv['verdict'],
        'thread_drift_pct': rep['ink_drift_pct'],
        'design_gaps': rep['output_white_regions'],
        'other_motifs_that_fit': [
            f"{o['cols']} {o['motif']} ({o['threads_per_motif']} threads each)"
            for o in p['options'][1:4]],
        'note': ('A whole panel: two side borders and the body between them. '
                 'Every region was built at its own width, so the border '
                 'linework is weavable and not just the body scaled down.'),
    }


def _tool_explore(session, args):
    """
    Build several designs and score each against the loom, ranked best first.

    The search belongs in code: whether a field weaves is a measurement, and
    measuring three candidates takes seconds. Reading the scores and saying
    which suits the cloth is the judgement, and that is the model's part.
    """
    import design_studio as ds

    spec = _spec_of(session)
    if spec is None:
        base = _tool_design(session, args)
        if 'error' in base:
            return base
        spec = _spec_of(session)

    n = max(2, min(4, int(args.get('count', 3) or 3)))
    try:
        ranked = ds.explore(spec, n=n)
    except Exception as e:
        return {'error': f'Could not explore alternatives: {e}'}

    session['variants'] = ranked
    return {
        'candidates': [
            {'index': i,
             'design': r.get('summary') or r.get('error'),
             'verdict': r.get('verdict'),
             'thread_drift_pct': r.get('thread_drift_pct'),
             'design_gaps': r.get('design_gaps')}
            for i, r in enumerate(ranked)],
        'note': ('Ranked by how cleanly each converts, not by how it looks. '
                 'Describe them to the weaver and let them choose; call '
                 'choose_design with the index once they have.'),
    }


def _tool_choose(session, args):
    """Adopt one of the explored candidates as the working design."""
    import design_studio as ds
    ranked = session.get('variants')
    if not ranked:
        return {'error': 'Nothing to choose from — run explore_designs first.'}
    try:
        i = int(args.get('index', 0))
    except (TypeError, ValueError):
        return {'error': 'Index must be a whole number.'}
    if not (0 <= i < len(ranked)):
        return {'error': f'Pick an index between 0 and {len(ranked) - 1}.'}
    r = ranked[i]
    if 'error' in r:
        return {'error': f"That candidate did not build: {r['error']}"}
    spec = ds.LayoutSpec(**r['spec'])
    summary = _adopt(session, spec, r['_conversion'], r['_image'])
    return {'chosen': summary, 'verdict': r['verdict']}


def _tool_refine(session, args):
    """
    Adjust the design in the weaver's own terms and rebuild it from vector.

    The change is applied to the SPEC, not to the raster. That matters: a spec
    re-renders at the exact pin count with stroke weights recomputed for the
    new geometry, whereas editing pixels degrades what came before, so ten
    small tweaks leave a design that no single step broke and none can undo.
    """
    import design_studio as ds
    from auto_convert import auto_convert

    spec = _spec_of(session)
    if spec is None:
        return {'error': 'There is no generated design to refine yet. '
                         'Use edit_design for an uploaded image.'}

    new_spec, desc, err = ds.refine(spec, args.get('change', ''))
    if err:
        return {'error': err}

    try:
        img = ds.render(new_spec)
    except Exception as e:
        return {'error': f'That change would not render: {e}'}

    conv = auto_convert(img, pins=new_spec.pins,
                        n_colors=max(2, min(4, new_spec.threads + 1)))
    if not conv.get('best'):
        return {'error': 'That change made a design that will not convert. '
                         'The previous one is still in place.'}

    # Verdicts are 'ok' / 'warn' / 'fail' — lower case, from fidelity.py. A
    # comparison against 'PASS' silently never fires, which is the worst
    # possible failure for a guard whose entire job is to speak up.
    rank = {'ok': 0, 'warn': 1, 'fail': 2}
    prev = session.get('conversion') or {}
    before_v = str(prev.get('verdict', 'ok')).lower()
    before_drift = ((prev.get('best') or {}).get('report') or {}).get('ink_drift_pct')

    summary = _adopt(session, new_spec, conv, img)
    rep = conv['best']['report']
    after_v = str(conv['verdict']).lower()
    out = {'changed': desc, 'design': summary, 'verdict': after_v,
           'thread_drift_pct': rep['ink_drift_pct'],
           'design_gaps': rep['output_white_regions']}

    if rank.get(after_v, 2) > rank.get(before_v, 0):
        out['warning'] = (f'This change dropped the verdict from {before_v} to '
                          f'{after_v}. Say so and offer to go back.')
    elif before_drift is not None and rep['ink_drift_pct'] > before_drift * 1.5 + 5:
        # A change can hold its verdict and still make the cloth materially
        # worse. Drift going from 26% to 55% inside the same 'warn' band is a
        # real degradation, and reporting only the verdict would hide it.
        out['warning'] = (f'Thread coverage drifted further, from '
                          f'{before_drift}% to {rep["ink_drift_pct"]}%, even '
                          f'though the verdict held. Mention it.')
    return out


def _tool_loom_geometry(session, args):
    """Threads to inches and back, at a given reed."""
    import design_studio as ds
    try:
        g = ds.geometry(pins=args.get('pins'), cards=args.get('cards'),
                        reed=args.get('reed'), picks=args.get('picks'),
                        width_in=args.get('width_in'),
                        length_in=args.get('length_in'))
    except Exception as e:
        return {'error': f'Could not work that out: {e}'}

    # Report clamping instead of hiding it. Asking for 45 inches at reed 80
    # needs 3600 threads; returning a quiet 2640 and calling it 33 inches
    # answers a question nobody asked, and the weaver finds out at the loom.
    asked_w = args.get('width_in')
    if asked_w and abs(float(asked_w) - g['width_in']) > 0.05:
        need = int(round(float(asked_w) * g['reed_epi']))
        g['warning'] = (f"{asked_w}in at reed {int(g['reed_epi'])} needs {need} "
                        f"threads, past the 2640 limit. The widest this loom "
                        f"weaves at that reed is {g['width_in']}in. A coarser "
                        f"reed or a narrower panel is the real choice.")
    asked_l = args.get('length_in')
    if asked_l and abs(float(asked_l) - g['length_in']) > 0.05:
        g['length_warning'] = (f"{asked_l}in of length needs "
                               f"{int(round(float(asked_l) * g['picks_ppi']))} cards, "
                               f"past the 6000 limit. Capped at {g['length_in']}in.")
    return g


def ml_motifs():
    import motif_library as ml
    return ml.MOTIFS





def _apply_canvas(session, fn, label):
    """
    Run a canvas operation, keeping undo and re-scoring afterwards.

    Every path through the canvas tools goes through here so that none of them
    can forget to push undo or to re-measure. An edit that silently skipped the
    re-score would let the model report a design as fine when the change had
    just broken it.
    """
    lm = _working(session)
    if lm is None:
        return {'error': 'There is no converted design yet. Design or convert one first.'}

    before = _rescore(session)
    prev = lm.copy()
    try:
        new_lm = fn(lm)
    except ValueError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f'That did not work: {e}'}

    import canvas_ops as co
    if new_lm is None or new_lm.size == 0:
        return {'error': 'That would leave nothing on the canvas.'}

    session['undo'].append(prev)
    session['undo'] = session['undo'][-10:]
    session['working'] = new_lm.astype(np.uint8)
    session['files'] = None            # generated files are now stale

    # Canvas size changes invalidate the spec: the design on screen is no
    # longer what the spec would render, so refinement must not silently go
    # back to the old geometry.
    if new_lm.shape != prev.shape:
        session['spec'] = None

    out = {'applied': label, 'canvas': co.stats(session['working'])}
    after = _rescore(session)
    if after:
        out.update({'verdict': after['verdict'],
                    'thread_drift_pct': after['ink_drift_pct'],
                    'design_gaps': after['output_white_regions']})
        if before and (abs(after['ink_drift_pct']) > abs(before['ink_drift_pct']) + 8
                       or (after['verdict'] == 'fail' and before['verdict'] != 'fail')):
            out['warning'] = (
                f"This made things worse (thread drift "
                f"{before['ink_drift_pct']:+.0f}% -> {after['ink_drift_pct']:+.0f}%). "
                f"Say so and offer undo_edit.")
    if new_lm.shape != prev.shape:
        out['note'] = (f'Canvas went from {prev.shape[1]}x{prev.shape[0]} to '
                       f'{new_lm.shape[1]}x{new_lm.shape[0]} threads x cards.')
    return out


def _tool_canvas(session, args):
    """Change the size or position of the cloth itself."""
    import canvas_ops as co

    op = str(args.get('operation', '')).strip()
    region, box = args.get('region'), args.get('box')

    def num(k, d=0):
        try:
            return int(args.get(k, d) or d)
        except (TypeError, ValueError):
            return d

    if op == 'extend':
        return _apply_canvas(session, lambda lm: co.extend(
            lm, left=num('left'), right=num('right'),
            top=num('top'), bottom=num('bottom')), 'extend canvas')
    if op == 'crop':
        return _apply_canvas(session, lambda lm: co.crop(lm, region, box), 'crop canvas')
    if op == 'trim':
        return _apply_canvas(session, lambda lm: co.trim(lm, margin=num('margin')),
                             'trim blank cloth')
    if op == 'resize':
        return _apply_canvas(session, lambda lm: co.resize_canvas(
            lm, pins=args.get('pins'), cards=args.get('cards'),
            anchor=str(args.get('anchor', 'center'))), 'resize canvas')
    if op == 'scale':
        # Distinct from resize on purpose: resize re-mounts the design on a
        # different canvas, scale resamples the design itself. Conflating them
        # would resample a design when the weaver asked to re-mount it.
        return _apply_canvas(session, lambda lm: co.scale(
            lm, pins=args.get('pins'), cards=args.get('cards')),
            'scale the design')
    if op == 'move':
        return _apply_canvas(session, lambda lm: co.move(
            lm, dx=num('dx'), dy=num('dy'), wrap=bool(args.get('wrap'))),
            'move the design')
    if op == 'centre' or op == 'center':
        return _apply_canvas(session, co.centre, 'centre the design')
    if op == 'mirror':
        return _apply_canvas(session, lambda lm: co.mirror_across(
            lm, axis=str(args.get('axis', 'vertical'))), 'mirror the panel')
    return {'error': (f"No such canvas operation: '{op}'. Available: extend, "
                      f"crop, trim, resize, scale, move, centre, mirror.")}


def _tool_region(session, args):
    """Work on one part of the cloth, leaving the rest alone."""
    import canvas_ops as co

    op = str(args.get('operation', '')).strip()
    region, box = args.get('region'), args.get('box')

    if op == 'clear':
        return _apply_canvas(session, lambda lm: co.clear(lm, region, box),
                             f'clear {region or "region"}')
    if op == 'copy':
        lm = _working(session)
        if lm is None:
            return {'error': 'There is no converted design yet.'}
        try:
            patch = co.copy_region(lm, region, box)
        except ValueError as e:
            return {'error': str(e)}
        session['clipboard'] = patch
        return {'copied': f'{patch.shape[1]}x{patch.shape[0]} threads x cards',
                'note': 'Held on the clipboard. Paste it with operation=paste.'}
    if op == 'paste':
        patch = session.get('clipboard')
        if patch is None:
            return {'error': 'Nothing on the clipboard — copy a region first.'}
        try:
            x, y = int(args.get('x', 0) or 0), int(args.get('y', 0) or 0)
        except (TypeError, ValueError):
            return {'error': 'x and y must be whole numbers.'}
        mode = 'over' if str(args.get('mode', 'blend')) == 'over' else 'blend'
        return _apply_canvas(session, lambda lm: co.paste(lm, patch, x, y, mode),
                             f'paste at {x},{y}')
    if op in ('mirror', 'flip_vertical', 'rotate_180', 'invert'):
        return _apply_canvas(session, lambda lm: co.transform_region(
            lm, op, region, box), f'{op} on {region or "region"}')
    if op == 'tile':
        return _apply_canvas(session, lambda lm: co.tile_region(
            lm, cols=int(args.get('cols', 2) or 2),
            rows=int(args.get('rows', 2) or 2), region=region, box=box),
            'repeat across the cloth')
    if op == 'weave_fill':
        return _apply_canvas(session, lambda lm: co.fill_region_weave(
            lm, pattern=str(args.get('pattern', 'satin')),
            n=int(args.get('n', 8) or 8), region=region, box=box,
            design_only=args.get('design_only', True) is not False),
            f"{args.get('pattern', 'satin')} fill")
    if op == 'stamp':
        motif = str(args.get('motif', '')).strip()
        try:
            width = int(args.get('width_threads', 0) or 0)
            x, y = int(args.get('x', 0) or 0), int(args.get('y', 0) or 0)
        except (TypeError, ValueError):
            return {'error': 'width_threads, x and y must be whole numbers.'}
        if width < 16:
            return {'error': 'A motif needs at least 16 threads to be drawn.'}
        return _apply_canvas(session, lambda lm: co.stamp_motif(
            lm, motif, width, x, y, mode=str(args.get('mode', 'blend'))),
            f'stamp {motif}')
    return {'error': (f"No such region operation: '{op}'. Available: clear, "
                      f"copy, paste, mirror, flip_vertical, rotate_180, invert, "
                      f"tile, weave_fill, stamp.")}


def _tool_canvas_info(session, args):
    """Report the canvas: size, coverage, where the design sits, blank margins."""
    import canvas_ops as co
    lm = _working(session)
    if lm is None:
        return {'error': 'There is no converted design yet.'}
    out = co.stats(lm)
    out['named_regions'] = sorted(co.NAMED_REGIONS)
    out['note'] = ('Coordinates are threads across and cards down from the '
                   'top-left. You can name a region or give a box.')
    return out


def _tool_plan(session, args):
    """
    Write down the job as steps, and tick them off as they are done.

    Not bookkeeping for its own sake. A weaver watching an agent work for a
    minute has no idea whether it is nearly finished or has lost the thread,
    and a visible plan is the difference between waiting and worrying. It also
    keeps the model honest: a plan stated up front is one it can be held to,
    where a plan kept in its head can quietly become a different job.

    Steps are free text — this deliberately does not constrain them to tool
    names, because the useful unit is "size the loom and check it fits", not
    "call loom_geometry".
    """
    steps = args.get('steps')
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.split('\n') if s.strip()]
    if steps is not None:
        if not isinstance(steps, list) or not steps:
            return {'error': 'Give me a list of steps.'}
        session['plan'] = [{'step': str(s)[:120], 'done': False}
                           for s in steps[:8]]
        return {'plan': session['plan'],
                'note': 'Now work through it. Mark steps done as you finish them.'}

    plan = session.get('plan')
    if not plan:
        return {'error': 'No plan yet — call plan_work with steps first.'}

    done = args.get('done')
    if done is not None:
        try:
            idx = [int(done)] if not isinstance(done, list) else [int(i) for i in done]
        except (TypeError, ValueError):
            return {'error': 'done must be a step number or a list of them.'}
        for i in idx:
            if 0 <= i < len(plan):
                plan[i]['done'] = True
    return {'plan': plan,
            'remaining': [p['step'] for p in plan if not p['done']]}


def _tool_auto_design(session, args):
    """
    Work toward the best weavable design for a brief, iterating unaided.

    This is the difference between an assistant and an agent. `design` builds
    one candidate from an estimate; this builds one, measures it, tries the
    changes that might improve it, keeps what did, and repeats until nothing
    helps. Every step is scored by fidelity against the render, so the loop
    cannot climb toward a design that will not weave.

    The trail comes back with the result. A weaver told only "six across,
    drift 11%" has to take it on trust; one shown that eight across drifted
    27% and five drifted 9% can see the shape of the trade and argue with it.
    """
    import design_studio as ds

    if not any(args.get(k) for k in ('pins', 'width_in')):
        return {'error': 'I need a width first — pins, or inches with the reed.'}

    motif = args.get('motif')
    if motif and motif not in ml_motifs():
        return {'error': f"Unknown motif '{motif}'."}

    try:
        out = ds.auto_design(
            pins=args.get('pins'), width_in=args.get('width_in'),
            reed=args.get('reed'), cards=args.get('cards'),
            length_in=args.get('length_in'), feel=args.get('feel'),
            threads=int(args.get('threads', 2) or 2), motif=motif,
            borders=args.get('borders', True) is not False,
            pallu=bool(args.get('pallu')),
            rounds=int(args.get('effort', 4) or 4))
    except Exception as e:
        return {'error': f'Could not work that through: {e}'}
    if 'error' in out:
        return out

    best = out['best']
    _adopt(session, best['spec'], best['conversion'], best['image'])
    p = out['plan']
    return {
        'design': ds.describe(best['spec']),
        'why_this': p['why'],
        'geometry': {k: v for k, v in p['geometry'].items() if k != 'raw'},
        'verdict': best['verdict'],
        'thread_drift_pct': best['drift'],
        'design_gaps': best['gaps'],
        'steps_taken': out['rounds_used'],
        'trail': [{k: v for k, v in t.items() if k != 'design'} for t in out['trail']],
        'note': ('Each step was rendered and measured, not guessed. Tell the '
                 'weaver what improved and by how much, in plain terms.'),
    }


def _tool_look(session, args):
    """
    Render the current design and hand the picture back so the model sees it.

    This widens the trust boundary deliberately and by exactly one step: the
    model may now LOOK at a design, but it still cannot MAKE one. Every pixel
    is produced by deterministic code, and anything the model concludes from
    the image has to come back as a named tool call that is range-checked like
    any other. A misread image yields a bad suggestion, never a bad BMP.

    Worth it because composition is the one judgement the fidelity score cannot
    make. Drift and gap counts say whether the linework survived the thread
    grid; they say nothing about whether the borders overpower the body, or
    whether the field reads as cloth or as wallpaper.
    """
    import design_studio as ds

    img = session.get('image')
    if img is None:
        return {'error': 'There is no design to look at yet.'}
    if not llm.provider().supports_vision:
        return {'error': ('This model backend cannot read images, so I cannot '
                          'look at the design. The measurements from convert '
                          'and auto_design still apply — use those.'),
                'measurements_only': True}

    spec = _spec_of(session)
    conv = session.get('conversion') or {}
    rep = ((conv.get('best') or {}).get('report')) or {}
    try:
        thumb = ds.thumbnail(img)
    except Exception as e:
        return {'error': f'Could not render a view: {e}'}

    return {
        'showing': ds.describe(spec) if spec else 'the uploaded design',
        'verdict': str(conv.get('verdict', 'unknown')).lower(),
        'thread_drift_pct': rep.get('ink_drift_pct'),
        'guidance': ('Judge the COMPOSITION, which the numbers cannot: do the '
                     'borders balance the body or overpower it, does the field '
                     'read as cloth, is the repeat obvious in a bad way, does '
                     'the pallu sit right. If something is wrong, fix it with '
                     'refine_design and look again. Do not comment on line '
                     'quality or resolution — that is what the fidelity score '
                     'already measured, and it is more reliable than your eye '
                     'on a thumbnail.'),
        '_images': [thumb],
    }


def _tool_compare(session, args):
    """Show the explored candidates side by side for a composition judgement."""
    import design_studio as ds
    from PIL import Image

    ranked = [r for r in (session.get('variants') or []) if 'error' not in r]
    if len(ranked) < 2:
        return {'error': 'Run explore_designs first — I need at least two to compare.'}
    if not llm.provider().supports_vision:
        return {'error': 'This model backend cannot read images.',
                'measurements_only': True}

    ranked = ranked[:3]
    thumbs = []
    for r in ranked:
        im = r['_image'].convert('L')
        scale = 300 / max(im.size)
        thumbs.append(im.resize((max(1, int(im.size[0] * scale)),
                                 max(1, int(im.size[1] * scale)))))
    h = max(t.size[1] for t in thumbs)
    sheet = Image.new('L', (sum(t.size[0] for t in thumbs) + 20 * len(thumbs), h), 255)
    x = 0
    for t in thumbs:
        sheet.paste(t, (x, 0))
        x += t.size[0] + 20

    return {
        'candidates': [{'index': i, 'design': r['summary'],
                        'verdict': r['verdict'], 'thread_drift_pct': r['thread_drift_pct']}
                       for i, r in enumerate(ranked)],
        'guidance': ('Left to right, in the order listed. Say which reads best '
                     'as cloth and why, then call choose_design.'),
        '_images': [ds.thumbnail(sheet, max_px=960)],
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
    # Intent-level tools. The parameter-level ones above stay callable so
    # nothing that already depends on them breaks, but the model is only
    # offered these.
    'design': _tool_design,
    'explore_designs': _tool_explore,
    'choose_design': _tool_choose,
    'refine_design': _tool_refine,
    'loom_geometry': _tool_loom_geometry,
    'auto_design': _tool_auto_design,
    'look_at_design': _tool_look,
    'compare_designs': _tool_compare,
    'plan_work': _tool_plan,
    'canvas': _tool_canvas,
    'region': _tool_region,
    'canvas_info': _tool_canvas_info,
}


# What each tool is doing, in words a weaver would use. Streaming
# 'auto_design · look_at_design' at someone is a stack trace; streaming
# "working out the best design for this width" is progress.
TOOL_NARRATION = {
    'inspect_design': 'Looking at your design',
    'convert': 'Converting it to threads',
    'generate_files': 'Building the loom files',
    'design': 'Designing the panel',
    'auto_design': 'Working out the best design for this width',
    'look_at_design': 'Looking at what I made',
    'compare_designs': 'Putting the options side by side',
    'explore_designs': 'Trying some alternatives',
    'choose_design': 'Going with that one',
    'refine_design': 'Adjusting the design',
    'loom_geometry': 'Working out the finished size',
    'list_motifs': 'Checking what I can build',
    'edit_design': 'Editing the linework',
    'undo_edit': 'Undoing that',
    'set_shuttles': 'Assigning colours to shuttles',
    'set_weave': 'Setting the weave',
    'describe_result': 'Checking how it looks',
    'plan_work': 'Planning the job',
}


def narrate(name):
    return TOOL_NARRATION.get(name, f'Running {name}')


# Tools that cannot run without a source image. On a design-from-scratch
# session these would otherwise fail deep inside on a None, and hand the model
# a Python traceback instead of a usable correction.
NEEDS_IMAGE = ('inspect_design', 'convert', 'edit_design', 'undo_edit',
               'set_shuttles', 'set_weave', 'describe_result',
               'generate_files')


def run_tool(name, args, session):
    """Execute a tool by name. Unknown tools and failures return an error dict
    rather than raising, so a confused model gets a correction instead of
    crashing the request."""
    fn = _DISPATCH.get(name)
    if not fn:
        return {'error': f'No such tool: {name}'}
    if name in NEEDS_IMAGE and session.get('image') is None:
        return {'error': ('There is no design in this session yet. Build one '
                          'first with auto_design, or ask the weaver to upload '
                          'an image.')}
    try:
        return fn(session, args if isinstance(args, dict) else {})
    except Exception as e:
        return {'error': f'{name} failed: {e}'}


# ── Conversation loop ───────────────────────────────────────────────────────

def _call_api(messages):
    """
    One model turn through whichever provider is configured.

    Kept as a single seam: this is the only function in the agent that touches
    a backend, so tests replace it wholesale and a provider swap changes
    nothing above it. Returns (Reply, None) or (None, message).
    """
    try:
        p = llm.provider()
        if not p.is_available():
            return None, ('No model backend configured. Set ANTHROPIC_API_KEY, '
                          'or set "llm_provider" in config.json to use a local model.')
        return p.complete(SYSTEM_PROMPT, messages, TOOLS, max_tokens=1400), None
    except llm.ProviderError as e:
        return None, e.message
    except Exception as e:
        return None, f'Assistant unavailable: {e}'


def _trim(history):
    """
    Take the last MAX_HISTORY messages without orphaning a tool exchange.

    A tool_results message whose matching assistant tool_calls turn has been
    trimmed away is rejected by both wire formats — Anthropic requires every
    tool_result to reference a tool_use in the preceding assistant turn, and
    OpenAI requires every role="tool" message to follow its tool_calls. So the
    window is advanced to the next clean boundary rather than cut mid-exchange.
    """
    window = history[-MAX_HISTORY:]
    while window and window[0].get('role') == 'tool_results':
        window = window[1:]
    return window


def converse(session, user_message, on_event=None):
    """
    Run one user turn, executing tools until the model produces a reply.

    on_event, if given, is called with dicts as the turn progresses:
    {'type': 'tool', 'name', 'label'}, {'type': 'tool_done', ...},
    {'type': 'plan', 'steps': [...]}. A full agentic turn can run a dozen
    tools over a minute or more, and a silent spinner for that long is
    indistinguishable from a hang — the caller needs to be able to show the
    work as it happens.

    Returns {'ok', 'reply', 'tools_used', 'has_files'}.
    """
    def emit(ev):
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass          # a broken listener must not kill the turn
    history = session['history']
    mark = len(history)
    history.append(llm.user_msg(user_message))

    tools_used = []
    for _ in range(MAX_TOOL_ROUNDS):
        reply, err = _call_api(_trim(history))
        if err:
            # Roll the whole turn back, not just the user message: a failure
            # part-way through a tool loop would otherwise leave assistant
            # turns with unanswered tool calls, which the next request rejects.
            del history[mark:]
            return {'ok': False, 'reply': err, 'tools_used': tools_used,
                    'has_files': bool(session.get('files'))}

        history.append(llm.assistant_msg(reply))
        session['usage'] = _add_usage(session.get('usage'), reply.usage)

        if not reply.wants_tools:
            emit({'type': 'reply', 'text': reply.text or 'Done.'})
            return {'ok': True, 'reply': reply.text or 'Done.',
                    'tools_used': tools_used,
                    'has_files': bool(session.get('files'))}

        results = []
        for call in reply.tool_calls:
            tools_used.append(call.name)
            emit({'type': 'tool', 'name': call.name, 'label': narrate(call.name)})
            out = run_tool(call.name, call.args, session)
            emit({'type': 'tool_done', 'name': call.name,
                  'ok': not (isinstance(out, dict) and 'error' in out),
                  'plan': session.get('plan')})
            # A tool may hand back pictures as well as numbers. They travel
            # under a private key so they are never serialised into the JSON
            # the model reads as text — base64 in a text field would blow the
            # context window and tell the model nothing.
            images = out.pop('_images', None) if isinstance(out, dict) else None
            entry = {'id': call.id, 'name': call.name,
                     'content': json.dumps(out, default=str)}
            if images and llm.provider().supports_vision:
                entry['images'] = images
            results.append(entry)
        history.append(llm.tool_results_msg(results))

    return {'ok': True,
            'reply': 'That took more steps than expected — could you rephrase?',
            'tools_used': tools_used, 'has_files': bool(session.get('files'))}


def _add_usage(total, delta):
    """Accumulate token counts per session, for cost reporting and budgets."""
    total = total or {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}
    for k, v in (delta or {}).items():
        total[k] = total.get(k, 0) + int(v or 0)
    return total


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
