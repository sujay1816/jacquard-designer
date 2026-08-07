# Complex fields — built

**725 tests passing.** Copy the contents of this folder into your repo root.

Last time I said no. The gap was three things: the motifs didn't exist, the
layout couldn't interlock, and nobody had drawn the vocabulary your reference
uses. All three are now built.

---

## Three new motifs

| Motif | What it is | Threads needed |
| --- | --- | ---: |
| `diamond_medallion` | Layered lozenge, concentric bands, rosette centre | 56 |
| `daisy` | Petal rosette with a contrasting eye | 40 |
| `leaf_sprig` | Curved stem with paired leaves, follows its own curve | 44 |

All parametric, all sized to the pin count like the existing seven — original
geometric constructions, not traced from anything.

Two decisions worth knowing:

**The medallion's innermost band is always ground**, whatever the band count.
Letting the alternation decide meant an even band count drew a dark rosette on a
dark field, where it disappears entirely.

**The sprig's leaves lean along the stem's tangent**, not a fixed angle. Leaves
placed on a straight line are what make machine-made sprigs look pinned rather
than grown.

---

## The `interlock` layout — this is the real change

Two lattices, offset by half a cell in both directions. Medallions on one, a
filler on the other, rotated per cell.

A single lattice repeats one upright unit and the eye finds the grid
immediately — that's why the old output read as wallpaper. Interleaving a
second, rotated element hides the grid **while keeping the repeat exactly as
regular as the loom needs.** The card sequence is unchanged; only the
appearance of regularity goes.

Reachable through four new feels: **flowing, ornate, floral, brocade**. So a
weaver says *"something ornate, 900 pins at reed 80"* and gets medallions with
sprigs flowing between them, borders, and a pallu.

---

## Four bugs found while building it

**The rosette was invisible.** Drawn dark on the dark innermost band.

**Every leaf bunched into the first third of the stem.** The stem curve was
evaluated with an inline conditional that got the branches the wrong way round.
Now a proper two-segment sampler returning point *and* tangent.

**The body overran into the pallu.** The field is composed *into* a panel with a
cross border beneath it — an overrun doesn't crop, it overlaps, so medallions
printed straight through the chevrons. Nothing is now placed that would hang
past the bottom edge.

**Ten medallions across.** An interlock field carries a filler in every gap, so
the same column count reads twice as busy. Traditional cloth of this kind has
four or five across a body. Interlock density is now scaled to 0.55.

And one ranking fix: `auto_design` on an "ornate" brief was choosing paisley —
technically fine, not the cloth asked for. Medallions and daisies now outrank
other buttas when the layout is interlock.

---

## What it still isn't

**Three ink threads, not five.** Your reference has cream, brown, deep red,
white and near-black. That ceiling is a loom question before it's a code one —
a five-colour weave needs the shuttles.

**A vocabulary of ten motifs, not a designer's hand.** Everything is still
parametric geometry. It will build you a convincing medallion field all day; it
won't invent a motif nobody has drawn.

**Vectorising uploads remains the real unlock.** Trace a designer's motif into
curves, then stamp and tile it at any pin count through the same interlock
layout. That's how the library grows without me writing each motif — and it's
the thing I'd build next.

---

## Files

**Changed:** `motif_library.py`, `design_studio.py`, `agent_engine.py`
**New:** `tools/test_motifs.py`

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        #  95
python tools/test_agent_page.py   #  85
python tools/test_install.py      #  57
python tools/test_studio.py       #  52
python tools/test_svg_raster.py   #  52
python tools/test_llm.py          #  49
python tools/test_agentic.py      #  44
python tools/test_hardening.py    #  43
python tools/test_motifs.py       #  40
python tools/test_nav.py          #  37
python tools/test_assistant.py    #  34
python tools/test_auto_convert.py #  16
```
