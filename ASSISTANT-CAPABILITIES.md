# What the Jacquard Designer assistant can do

A working reference for the AI assistant at `/agent`. Everything listed here is
built and covered by tests. The limits section at the end matters as much as the
rest — knowing what it *can't* do is what stops a design going wrong at the loom.

---

## In one sentence

You describe cloth in a weaver's terms; it designs it, measures it against the
loom, improves it, shows you, and writes the BMPs — or takes a design you upload
and converts it, saying honestly what survived.

---

## The two things it does

### 1. Designing from a brief

You give it intent. It works out the geometry.

> **You:** traditional saree body, about 9 inches at reed 80, gold and one colour
>
> **It:** 9 inches at reed 80 is 720 threads. Two 64-thread borders leave 592 for
> the body, so six paisleys across gives each one 98 threads — enough for the
> interior detail to read. Vine border up both selvedges. Want a pallu across
> the foot?

You never supply columns, rows, spacing or layout. Those are design decisions,
and the numbers needed to make them well live in the codebase, not in your head.

### 2. Converting a design you upload

Photo, scan or drawing → loom-ready 1-bit BMPs, one per shuttle. It inspects the
image first, suggests a pin count the source can actually support, converts, then
reports what was lost in craft terms rather than percentages.

---

## What it produces

A **complete panel**, not a bare field:

- **Side borders** up both selvedges
- **Body field** between them
- **Pallu** — a cross border at the foot (optional)

Each region is built at *its own width*, so a border occupying 9% of the cloth
gets linework sized for that space rather than inheriting strokes scaled to the
full width and landing under the weavable threshold.

Output: one 1-bit BMP per shuttle, plus a zip. Every file is verified as a real
1-bit BMP before it's offered.

---

## The motif library

Ten parametric motifs. The thread count is the minimum each needs before its
detail stops reading — the assistant uses these to decide how many will fit.

**Buttas** — the anchor motif in a field

| Motif | | Threads |
| --- | --- | ---: |
| `diamond_medallion` | Layered lozenge, concentric bands, rosette centre | 56 |
| `paisley` | Teardrop butta, nested outlines, seed detail | 48 |
| `lotus` | Radial rosette, several rings deep | 48 |
| `daisy` | Petal rosette with a contrasting eye | 40 |

**Grounds** — all-over fills

| Motif | | Threads |
| --- | --- | ---: |
| `diamond_jaal` | Diamond lattice | 24 |
| `check_ground` | Square check | 16 |
| `dotted_field` | Scattered dots, lightest fill | 12 |

**Bands and fillers**

| Motif | | Threads |
| --- | --- | ---: |
| `leaf_sprig` | Curved stem with paired leaves | 44 |
| `vine_border` | Running creeper band | 30 |
| `chevron_border` | Zig-zag separator | 20 |

All are original geometric constructions, deliberately not derived from scraped
artwork.

### Seven layouts

`straight` · `half_drop` · `brick` · `banded` · `jaal` · `stripe` · `interlock`

**`interlock` is the one that makes a field look woven** rather than stamped. Two
lattices offset by half a cell: medallions on one, a filler rotated per cell on
the other. The repeat stays exactly as regular as the loom needs — only the
*appearance* of regularity goes.

### Thirteen "feels"

Say it in your own words and it maps to geometry:

`rich` · `dense` · `traditional` · `classic` · `open` · `light` · `minimal` ·
`geometric` · `formal` · **`flowing` · `ornate` · `floral` · `brocade`**

The last four use the interlock layout — that's what to ask for when you have a
busy traditional reference in mind.

---

## Working autonomously

`auto_design` is a hill climb, not a single guess: build a candidate, measure it
against the loom, try the five changes that might improve it, keep whichever did,
repeat. Every step is scored against a real render, so it cannot climb toward
something unweavable.

It returns **the trail** — what it tried and what each attempt cost:

```
start          warn   drift 27.8%
fewer_motifs   warn   drift 26.2%
fewer_motifs   warn   drift 23.7%
stop           no further change improved it
```

That matters as much as the answer. "Five across, 19% drift" has to be taken on
trust; being shown that eight across drifted 27% lets you see the trade and argue
with it.

It also **looks at its own work** — renders a thumbnail and reads it — to judge
composition, which the fidelity score cannot: whether the borders overpower the
body, whether the repeat is obtrusive. Needs a vision-capable model; without one
it says so rather than pretending.

`explore_designs` builds alternatives and ranks them; `compare_designs` shows
them side by side.

---

## Editing

**Linework**
`thicken` · `thin` · `clean_specks` · `close_gaps` · `remove_isolated` ·
`smooth` · `invert` · `rotate_90/180/270` · `flip_horizontal` · `flip_vertical`

**The cloth itself**
`extend` · `crop` · `trim` · `resize` · `scale` · `move` · `centre` · `mirror`

Two distinctions it keeps straight because getting them wrong ruins cloth:
**`resize`** re-mounts the design on a different canvas; **`scale`** resamples
the design itself, losing detail going down and blocking it going up.

**Regions**
`clear` · `copy` · `paste` · `mirror` · `flip_vertical` · `rotate_180` ·
`invert` · `tile` · `weave_fill` · `stamp`

Name a region — `body`, `pallu`, `left_border`, `centre`, `top`, `bottom` — or
give an exact box in threads and cards.

`tile` turns one drawn butta into a field. `stamp` draws a library motif onto the
canvas at a size and place you choose. `weave_fill` textures an area with satin,
twill (2/2 or 3/1), plain, basket, honeycomb, diamond, herringbone, crepe, rib,
dots, diagonal or crosshatch — and understands `twill`, `plain`, `matt` and
`hopsack` as spoken.

**Refining in your own words**
"Too busy" → `more_open`. "Looks empty" → `denser`. "Motifs too small" →
`fewer_motifs`. Also `taller`, `shorter`, `wider_border`, `narrower_border`.

Refinements change the **spec** and re-render from vector at full quality — not
the pixels. Editing the raster instead means every tweak degrades what came
before, and ten small ones leave a design that no single step broke and none can
undo.

---

## Keeping track

- **`checkpoint`** — save named versions and return to them. Undo is a ten-deep
  stack that can't be aimed; "go back to before I widened the border" can.
- **`undo_edit`** — ten steps.
- **`plan_work`** — writes the job down as steps and ticks them off, visible on
  screen while it works.
- **`files`** — checks the download actually exists and verifies as 1-bit BMP.
  Any edit clears generated files, so it checks before promising one.

---

## Loom knowledge

- **`loom_geometry`** — threads ↔ inches at a given reed, either direction. A pin
  count means nothing physical without the reed: 480 pins is 8 inches at reed 60
  and 4.8 at reed 100.
- **`set_shuttles`** — colour to shuttle. The count includes the rani ground, so
  a two-colour design needs `shuttle_count 3`.
- **`set_weave`** — per-shuttle weave and float length.
- Float and long-run warnings before any file is written.

Limits: **10–2,640 pins**, **10–6,000 cards**, **3 ink threads** plus ground.

---

## On screen

- Live design preview beside the conversation, refreshed every step
- **Thread-level inspection** — full screen at one pixel per thread, no
  smoothing, so a two-thread vine is visibly two threads or visibly gone
- **Now / Before / Source** tabs to see what a change actually did
- Verdict, canvas size, finished inches, thread drift and coverage, always visible
- Saved versions as clickable buttons
- Each BMP downloadable on its own, or all as a zip
- Work streams as it happens — "Working out the best design for this width",
  "Looking at what I made" — rather than a spinner

---

## Honesty rules it follows

These are the reason to use it rather than a generator:

- It will not call a conversion good when a tool reported `warn` or `fail`.
- If detail is lost it says **what** was lost — "the fine scrollwork inside the
  small motifs" — not "12% ink drift".
- If a refinement made things worse, it leads with that and offers to go back.
- If your source is too small to carry the design, it says a rescan is the only
  fix, before the cloth is on the loom.
- Asked for Chola, Banarasi or Kanjivaram, it says it can build something in that
  spirit but not an authentic period motif, and that a designer is the right
  answer for anything a customer will recognise.
- If it has no tool for what you want, it says so rather than improvising.

---

## What it cannot do

**Three ink threads, not five.** A design with cream ground, brown, deep red,
white and near-black is beyond it. That's a shuttle question before it's a code
one.

**Ten motifs is a vocabulary, not a designer's hand.** It will build a convincing
medallion field all day. It will not invent a motif nobody has drawn.

**No authentic regional work.** Parametric geometry approximates centuries of
convention at best, and it says so rather than passing a generic paisley off as
Chola work.

**It cannot sharpen a soft scan.** Deliberately: generative upscalers invent
plausible detail, which is fine in a photo and not fine in a manufacturing
instruction — a confidently wrong vine gets woven into silk, and nobody can tell
invented detail from recovered detail.

**It cannot edit an uploaded design as a design.** Uploads are pixels, so only
linework and canvas operations apply; `refine_design` works on generated designs
only. Vectorising uploads would close this, and is the single highest-value thing
left to build.

**It has no memory of your preferences between turns.** Say "too heavy", it
adjusts — and may suggest the same thing again later.

**Nothing has been verified against a live model.** Every test scripts the model,
so they prove the engine can't be broken *by* it, not that its judgement is good.
Worth checking yourself: give it a brief, then deliberately ask for something
that won't weave, and see whether it tells you plainly.

---

## Getting it running

`config.json` next to `app.py`:

```json
{ "anthropic_api_key": "sk-ant-..." }
```

Or a local model, no key needed:

```json
{ "llm_provider": "ollama", "llm_model": "llama3.3:70b" }
```

Then `python run.py` → `http://localhost:5000/agent`.

Vision, for `look_at_design`, needs a vision-capable model. Claude has it;
`llava`, `qwen-vl` and `gemma3` are auto-detected locally.

**No native libraries required.** Cairo is optional — there's a built-in renderer
that agrees with it to within a few percent.

---

## Cost, roughly

At Sonnet-class pricing with prompt caching on:

| | |
| --- | ---: |
| A simple question | $0.02 |
| One full agentic turn | $0.09 |
| A ten-turn design session | $0.88 |
| 100 sessions per month | ~$88 |

Compute may cost more than the API: `auto_design` renders and scores about
twenty candidates per call.

---

## The 22 tools, for reference

**Designing** — `auto_design`, `explore_designs`, `choose_design`,
`compare_designs`, `refine_design`, `list_motifs`
**Converting** — `inspect_design`, `convert`, `describe_result`
**Editing** — `edit_design`, `undo_edit`, `canvas`, `canvas_info`, `region`
**Loom** — `loom_geometry`, `set_shuttles`, `set_weave`, `generate_files`, `files`
**Working** — `plan_work`, `checkpoint`, `look_at_design`

You never need to name these. They're what it reaches for while you talk in
weaver's terms.
