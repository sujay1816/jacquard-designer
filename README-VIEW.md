# The design view

**508 tests passing.** Your push was clean. This drop changes only
`agent_engine.py`, `app.py`, `templates/agent.html` and five test files.

---

## The gap

The panel could show the design but nothing *about* it. The verdict, the drift,
the finished size, what had been saved, which files existed — all of it lived
only inside the agent's prose. A weaver reading "it comes out a warn" in a
sentence cannot glance at it four turns later, and cannot watch it change as
they work.

And the preview was 290px. A jacquard design is meaningful **thread by thread**
— whether a two-thread vine survived the reduction is the whole question, and at
290px it looks fine either way.

---

## What's new in the view

**A measurements strip** under the preview: verdict (colour-coded ok / warn /
fail), canvas in threads × cards, finished cloth in inches at the real reed,
thread drift, coverage. Always visible, updates every turn.

**Thread-level inspection.** Click the design for a full-screen view with a
**1 thread = 1 pixel** button and `image-rendering: pixelated`. No smoothing —
a two-thread vine is visibly two threads, or visibly gone. This is the one that
matters for checking work before it goes to the loom. Fit / zoom in / out, Esc
to close.

**Now / Before / Source tabs.** See what a refinement actually changed rather
than only reading about it. The tabs only appear when the comparison means
something: there is no "Before" until something has been edited, and a
generated design has no source — comparing it with itself isn't a comparison.

**Saved versions, clickable.** Checkpoints were invisible to the weaver — the
agent could save them and only the agent knew. They now appear as buttons.
Clicking one **asks the agent** rather than calling restore directly, so it
stays in the loop and can say what changed; the move becomes part of the
conversation instead of a silent state change the transcript never mentions.

**Files listed individually**, each downloadable on its own. A zip is right for
a loom operator and wrong for checking one shuttle before committing.

**A note when the reference has been rebased.** After a canvas resize, fidelity
is measured against the design rather than the original. Without saying so the
numbers look continuous across that boundary when they aren't.

---

## New routes

| Route | Does |
| ----- | ---- |
| `GET /api/agent/state` | Verdict, size, drift, checkpoints, files, plan — one call |
| `GET /api/agent/file?name=` | One BMP on its own |
| `GET /api/agent/preview?which=` | `design` (default), `previous`, `source` |

Preview resolution raised from 1400px to 2400px, since the detail is the point.

---

## Two bugs found while building this

**The route registered the wrong function.** I put a helper between
`@app.route` and its view, so Flask bound the route to the helper —
`/api/agent/preview` returned a 500 on every call. Caught immediately because
the test exercised all three view modes.

**State reported no canvas at all.** `session['working']` is materialised
lazily from the conversion, so reading the key directly showed nothing until
some tool happened to touch it first. Fixed by using the same accessor the
tools use.

---

## What I'd do next, in order

**1. Show the explored candidates as images.** `explore_designs` builds three
alternatives and the weaver only ever reads about them. The thumbnails already
exist in the session for `compare_designs` — they just aren't exposed to the
page. Small change, large effect on a design tool.

**2. Region highlighting.** When the agent says "the pallu", the weaver has to
guess which part it means. `canvas_info` already returns exact boxes; drawing
one on the preview would make every region instruction unambiguous.

**3. Markdown in replies.** Messages use `textContent`, so the agent's prose
renders flat. It writes measurements and comparisons that would read far better
with light formatting.

**4. A thread ruler on the zoom view.** At 1:1 you can see individual threads
but not count them. Gridlines every 10 or 50 would make "the vine is 2 threads
here" checkable rather than estimable.

Still open from before: no cross-turn memory, and nothing has run against a
live model.

```
python tools/test_agent_page.py   # 60
python tools/test_canvas.py       # 121
python tools/test_agent.py        # 95
python tools/test_studio.py       # 52
python tools/test_agentic.py      # 44
```
