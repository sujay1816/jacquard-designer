# AI designer — intent-driven design generation

The agent now designs cloth from a brief instead of taking geometry parameters.
Built entirely on the engines already in the product: `motif_library` for
geometry, `auto_convert` for conversion, `fidelity` for scoring, `loom_utils`
for physical size. Nothing here touches a pixel directly.

**283 tests passing** (95 agent + 52 studio + 49 llm + 34 assistant + 16
auto-convert + 37 nav), all offline.

---

## What changed in the conversation

Before, the model had to supply `layout="jaal"`, `motif="lotus"`, `cols=5`,
`rows=4`, `spacing=0.25` before anything could be built — so the weaver was
asked for them. That is a form with a chat box in front of it, and it puts the
design decisions on the person least equipped to make them at that moment.

Now the brief carries intent and the geometry is worked out in code:

> **Weaver:** I need a traditional saree body, about 9 inches at reed 80, gold
> and one colour.
>
> **Assistant:** 9 inches at reed 80 is 720 threads. Two 64-thread borders
> leave 592 for the body, so six paisleys across gives each one 98 threads —
> enough for the interior detail to read. Built it with a vine border up both
> selvedges. Want a pallu band across the foot?

The numbers in that reply come from `plan()`, not from the model. The model's
job is deciding what to ask, which tool to call, and how to say it.

---

## New files

| File | Purpose |
| ---- | ------- |
| `design_studio.py` | Planning, whole-panel composition, exploration, refinement |
| `llm/` (5 files) | Swappable backends — Claude, Llama, any OpenAI-compatible server |
| `tools/test_studio.py` | 52 tests |
| `tools/test_llm.py` | 49 tests |

Changed: `agent_engine.py` (new tools, rewritten system prompt), `app.py`
(availability check), `tools/test_agent.py` (fake-response helpers only — no
assertion changed).

---

## The generated image is now a panel, not a field

`compose()` builds **side borders + body + optional pallu band** as one design.
Previously `generate_allover` produced a rectangle of repeating motifs, which is
not a saree — it is the middle of one. This was the largest gap between what the
generator produced and what a mill can use.

Each region is built at *its own width*. A border occupying 9% of the cloth
would otherwise inherit strokes scaled to the full width and land far under the
weavable threshold. This is the rule `motif_library` established for tiles,
applied to regions.

---

## New tools

| Tool | Does |
| ---- | ---- |
| `design` | Brief → plan → compose → convert → score, in one call |
| `explore_designs` | Builds and scores alternatives, ranked |
| `choose_design` | Adopts one by index |
| `refine_design` | Weaver-language adjustment, re-rendered from vector |
| `loom_geometry` | Threads ↔ inches at a given reed, either direction |

`generate_allover` and `design_options` were removed from the model's tool list
but stay callable in `_DISPATCH`, so nothing depending on them breaks. Net tool
count 15.

**The reed is now reachable.** `loom_utils.physical_size()` always existed but
nothing exposed it, so pin counts were being chosen with no idea of finished
cloth width. 480 pins is 8 inches at reed 60 and 4.8 at reed 100.

---

## Refinement changes the spec, not the pixels

A `LayoutSpec` is about twenty numbers. Any change re-renders at the exact pin
count with stroke weights recomputed for the new geometry. Editing the raster
instead means every adjustment degrades what came before, and ten small tweaks
leave a design that no single step broke and none of them can undo.

`refine()` maps weaver vocabulary onto spec changes — "too busy" → `more_open`,
"looks empty" → `denser`, "motifs too small" → `fewer_motifs`. An unrecognised
instruction is **refused**, not ignored: silently doing nothing looks identical
to doing the wrong thing, and the weaver would only find out at the loom.

---

## Four bugs the tests caught

Each is now covered by a named test, so a future edit that reintroduces it
fails the suite.

**The regression guard could never fire.** `refine_design` compared the verdict
against `'PASS'`, but `fidelity.py` emits `'ok'` / `'warn'` / `'fail'` in lower
case. A change that doubled thread drift was reported as a plain success. Now
ranked properly — and it also warns when drift worsens materially *within* one
verdict band, since 26% → 55% inside `warn` is a real degradation the verdict
alone hides.

**Geometry clamped impossible requests silently.** 45 inches at reed 80 needs
3,600 threads; it returned 2,640 and reported 33 inches as though that were the
answer. Now it says what the request actually needs and what the loom can do.

**Ranking always chose scattered dots.** Sorting on thread headroom alone
always wins with `dotted_field` — it needs 12 threads against paisley's 48, so
it scores best at every width. A weaver asking for a saree body was handed a
dotted ground. Buttas now outrank grounds when both read well; grounds are the
fallback for cloth too narrow to hold a butta.

**Rows left bare cloth above the pallu.** Spacing does *not* add row height in
`allover()` — the gap divides into the tile scale and cancels exactly
(`row_h = tile_w × aspect`). Treating it as extra vertical pitch under-counted
rows. Row counts are now measured from the motif's real aspect ratio and
recomputed at compose time, so a refinement to `cols` re-fills the panel.

---

## Still open

**Sharpening uploaded scans.** Not built — it needs your decision first.
Generative upscalers invent plausible detail, which is fine in a photo and not
fine in a manufacturing instruction: a confidently wrong vine gets woven into
silk, and the weaver cannot tell invented detail from recovered detail. The
workable version is deterministic pre-processing, vision-model *guidance* that
picks parameters while deterministic code moves pixels, and vectorisation —
trace to curves, render at the exact pin count.

Vectorisation would also let `refine_design` work on *uploaded* designs, not
just generated ones. Right now an upload can only be edited pixel-wise.

**Budgets.** `session['usage']` accumulates token counts; nothing enforces a
limit yet.

**Motif range.** Still seven parametric motifs, all geometric and stylised. The
composition layer makes them go much further, but it cannot make them
traditional — that limit is real and the system prompt tells the weaver so.

**Untested against a live model.** Both adapters and the whole tool loop are
covered by scripted tests. First live run should be one conversation per
backend.
