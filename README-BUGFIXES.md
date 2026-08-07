# Bug audit — seven found, all fixed

**471 tests passing.** Copy the contents of this folder into your repo root.

I went looking for the same *classes* of bug as the download failure: stale
derived state, aliasing, silent fallbacks, and comparisons that can never fire.
All seven below now have a named test, so a future edit that reintroduces one
fails the suite.

---

## 1. Every canvas resize reported the design as damaged

Extending a canvas with **blank cloth** scored −54.8% drift and a `warn`. The
720-thread label map was being compared against the 320-thread source, so the
difference the weaver deliberately asked for was indistinguishable from damage.

Fidelity answers "how much of the reference survived". Once the cloth is
resized, the original no longer describes what will be woven. The reference is
now **rebased** to the design as it stands, and the weaver is told the
comparison back to the original ends there — rather than being handed numbers
that quietly changed meaning.

**My first fix for this was wrong**, and the test suite caught it. I made
`_rescore` refuse to score when sizes differed — but for a converted upload the
source photo and the loom-resolution mask *never* have the same dimensions, and
`fidelity_report` exists precisely to bridge that. It broke ordinary conversion
entirely. The guard belonged at the resize, not at the measurement.

## 2. Undo destroyed the comparison to the original

A knock-on from the above: rebasing on *every* undo made the design its own
reference, so drift read ~0 no matter what was undone. Now it rebases only when
undo actually changes the canvas size.

## 3. `twill` worked in one tool and was refused by its sibling

`region/weave_fill` accepted it; `set_weave` rejected it. Same word, same
conversation. The engine's key is `twill22`, which nobody at a loom says. Both
tools now share one alias table — `twill`, `plain`, `matt` all understood,
made-up names still refused.

## 4. Sessions grew to 1.4 GB

A worked session reached **35 MB**, and the server keeps 40 — 1.4 GB.

- Each explored variant held a full PIL image *and* a full conversion record.
- Each checkpoint held the label map, the reference image, *and* the
  conversion's own copy of the label map — the same data three times.
- Undo held ten raw label maps.

Fixed by deriving rather than duplicating. Variants keep a thumbnail and are
**rebuilt from their spec when chosen** — safe only because rendering is
deterministic, which `test_studio` already asserts. Checkpoints store the label
map alone; the reference and the conversion's copy are derived on restore.
Label maps are zlib-packed: a handful of distinct values over long flat runs
compresses to a few percent, losslessly.

**35 MB → 6 MB. Worst case 1.4 GB → 241 MB.**

## 5. Restore read a field that no longer existed

Introduced by fix 4 and caught immediately — `cp['image']` after I stopped
saving it.

## 6. Shuttle errors couldn't be acted on

*"Exactly one colour must be the background"* said nothing about which colours
existed or which were left out, so the model had to guess the correction and
often guessed wrong. It now names the colours in the design, what was assigned,
what was omitted, and that the largest area is usually the ground — a refusal
that can be fixed in one step instead of three.

## 7. Physical size was reported at a hardcoded reed 60

Carried over from the last drop, retested here. A design built at reed 80 was
reported 33% too wide. The reed now lives on the session — it is a property of
the loom, not of one design — so it survives the canvas resizes that drop the
spec.

---

## What I checked and found clean

`set_shuttles` validates against colours that actually exist. `generate_files`
regenerates from the working design rather than the original conversion, so
edits are never silently discarded. Session pruning and the tool-round cap both
work. `verify_bmp` confirms every generated file is a real 1-bit BMP.

---

## Still open

**No cross-turn memory.** The weaver says "too heavy", the agent adjusts — and
next turn it will suggest the same thing again. Nothing carries a preference
forward. This is the largest remaining gap between this and a full assistant,
and it is a feature rather than a bug.

**24 tools, ~6,100 tokens per round.** Fine with prompt caching on Claude; past
where a local 70B holds tool-selection accuracy.

**Nothing has run against a live model.** Every suite here is scripted. The
agentic loop and `look_at_design` are the two pieces no offline test can
validate — a model that always says "looks good" would pass everything here and
still be useless.

```
python tools/test_canvas.py       # 110
python tools/test_agent.py        # 95
python tools/test_studio.py       # 52
python tools/test_llm.py          # 49
python tools/test_agentic.py      # 44   (~4 min)
python tools/test_agent_page.py   # 34
```
