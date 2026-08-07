# Complete AI designer — canvas and pixel work

The agent can now change the cloth, not just the linework. **433 tests
passing**, all offline.

Copy the contents of this folder into your repo root; paths already match.

---

## What it can do now

Before, the agent could thicken, thin and clean linework. It could not widen a
panel for a bigger loom, trim blank selvedge, shift a motif off a fold, clear a
region and put something else there, or repeat a butta into a field. Those are
what a designer actually spends the day on.

**`canvas`** — the cloth itself:

| Operation | Does |
| --------- | ---- |
| `extend` | Add bare cloth on any side |
| `crop` | Keep a region, discard the rest |
| `trim` | Cut away blank cloth around the design |
| `resize` | Re-mount on an exact canvas **without resampling** |
| `scale` | Resample the design to a different thread count |
| `move` | Shift it, with `wrap` for all-over repeats |
| `centre` | Balance the margins |
| `mirror` | Fold one half onto the other — how a matched border pair is built |

**`region`** — part of the cloth: `clear`, `copy`, `paste`, `mirror`,
`flip_vertical`, `rotate_180`, `invert`, `tile` (one butta becomes a field),
`weave_fill` (satin, twill, basket, honeycomb, diamond, crepe, rib,
herringbone…), and `stamp` — draw a library motif onto the canvas at a size and
position you choose.

**`canvas_info`** — size, coverage, where the design sits, blank margins, and
the region names it accepts. The agent is told to call this before coordinate
work so it moves real numbers rather than guessing.

Regions can be named — `body`, `pallu`, `left_border`, `centre`, `top` — or
given as an explicit `[x0, y0, x1, y1]` box in threads and cards.

---

## Everything works on the label map, not on RGB

That is what makes it safe:

- **Class indices survive every operation**, so a three-colour design keeps its
  shuttle separation through a move, a crop or a paste. Working in RGB and
  re-clustering afterwards would let boundaries drift on every single edit.
- **New cloth is bare cloth.** Extending fills with class 0. Mirroring the edge
  or repeating the last column would put thread lifts on the loom that nobody
  asked for, and they are very hard to spot inside a repeat.
- **Nothing is resampled unless asked.** Moving and cropping are index
  operations; nothing softens.

Every canvas change goes through one function that pushes undo, re-scores the
result, invalidates generated files, and drops a spec that no longer matches the
canvas. None of the paths can forget.

---

## Two bugs the tests caught, both mine

**The weave fill was inverted.** `bmp_engine` uses `0 = thread UP, 1 = DOWN`. I
read `1` as lift, so every weave came out as its own negative — the float landed
exactly where the binding should be. Now fixed and covered by a test.

**Asking for twill silently gave you satin.** `generate_fill_pattern` falls back
to satin for any name it does not recognise rather than raising, and the real
key is `twill22`, not `twill` — so my own tool schema was documenting a name
that produced the wrong weave. A weaver who asks for twill and gets satin cannot
tell until the cloth is off the loom. Names are now validated before the call,
with aliases so `twill`, `plain` and `matt` work as spoken.

---

## Two distinctions the agent is told to keep straight

**`resize` vs `scale`** — resize re-mounts the design on a different canvas;
scale resamples the design itself, losing detail going down and blocking it
going up. If a weaver asks to fit a design to another loom, the agent asks which
they meant.

**`blend` vs `over`** — blend lays a piece over what is there and lets the
ground show through its gaps; over replaces the whole rectangle. Stamping a
butta with `over` punches a bare rectangle through a lattice.

---

## Files

**New:** `canvas_ops.py`, `tools/test_canvas.py`
**Also in this drop:** `design_studio.py`, `llm/` (5), `templates/agent.html`,
and the other test suites — the full set from this session.

```
python tools/test_canvas.py       # 72
python tools/test_agent_page.py   # 34
python tools/test_agentic.py      # 44   (slowest, ~4 min)
python tools/test_studio.py       # 52
python tools/test_llm.py          # 49
python tools/test_agent.py        # 95
```

---

## Worth knowing

**The tool surface is now 22 tools, ~5,700 tokens per round.** That is a lot of
context re-sent on every tool call. Prompt caching covers it on Claude. On a
local 70B model, 22 tools is past where tool-selection accuracy starts to
degrade — if you go that route, consider splitting the design tools and the
canvas tools into two modes.

**Canvas work needs a converted design.** These operate on the label map, so
`design` or `convert` has to have run first. The agent gets a clear sentence
saying so rather than a `NoneType` traceback.

**Still not built:** sharpening uploads. Generative upscalers invent plausible
detail, fine in a photo and not fine in a manufacturing instruction.
Vectorisation remains the version worth building — and now more than before,
since it would let all of this canvas work apply to scanned designs at full
quality rather than at whatever resolution the scan happened to have.
