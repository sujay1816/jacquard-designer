# Download fixed, plus checkpoints

**453 tests passing.** Copy the contents of this folder into your repo root.

Your push landed correctly this time — 22 tools, canvas ops, streaming page all
present and byte-identical.

---

## You asked about downloads. They were broken.

Files generated fine, and the zip downloaded fine — **until anyone touched the
canvas.** After that the session could never produce files again:

```
generate_files failed: label_map shape (448, 360) does not match canvas
(448 cards x 320 pins). Please re-run Detect Colours before generating.
```

Advice that cannot be followed from a conversation. The cause: a canvas resize
changed the label map but left the conversion record claiming the old pin
count, and `generate_files` reads that record.

**A second, nastier version of the same bug** turned up in the new checkpoint
code. A checkpoint stored a *reference* to the conversion dict, so a later
canvas edit mutated the saved pin count underneath it — restoring a known-good
version produced a record describing a canvas that no longer existed.

Both are fixed by one enforced invariant, `_sync_conversion_size()`: the
conversion record's pins and cards are always set **from the array's actual
shape**, never trusted from whoever last touched it. That is what makes the
aliasing harmless rather than requiring every path to remember to deep-copy.

**Also fixed:** physical size was reported at a hardcoded reed 60, so a weaver
who designed at reed 80 was told their 4-inch panel measured 5.3 inches. The
reed now lives on the session — it is a property of the loom, not of one
design, and it was being dropped along with the spec on every canvas resize.

---

## Two new tools, both about the agent tracking its own work

**`checkpoint`** — save the design under a name, list what is saved, restore
one. Undo is a stack ten deep: good for "that was wrong", useless for "go back
to the version before I added the border", which is what a designer actually
asks after twenty minutes. An agent that cannot return to a known-good state
has to rebuild from the brief and will not reproduce it exactly. Checkpoints
hold the label map, canvas size, spec and reed together, so restoring one puts
the whole session back rather than leaving a design that disagrees with its own
measurements. Capped at 8, since each holds a full label map.

**`files`** — check whether the loom files exist, still match the design, and
verify as real 1-bit BMPs. The agent could produce files and then had no way to
look at them; after any edit they are silently cleared, so it could tell a
weaver the download was ready when there was nothing behind the button. It now
also distinguishes *"no files have been generated yet"* from *"the design
changed, so the files were cleared"* — those need different next steps.

The system prompt tells it to checkpoint before anything risky, and to check
`files` before promising a download.

---

## Files

**Changed:** `agent_engine.py`, `tools/test_canvas.py`
**Unchanged from your push but included** so the folder copies cleanly:
`canvas_ops.py`, `design_studio.py`, `app.py`, `llm/`, `templates/agent.html`,
the other suites.

```
python tools/test_canvas.py       # 92
python tools/test_agentic.py      # 44   (~4 min)
python tools/test_studio.py       # 52
python tools/test_llm.py          # 49
python tools/test_agent_page.py   # 34
python tools/test_agent.py        # 95
```

Housekeeping: `README-AI-DESIGNER.md`, `README-ASSISTANT-PAGE.md`,
`README-CANVAS.md` and `changes.patch` are all committed in your repo root.
They are notes, not code — worth deleting or moving to a `docs/` folder.

---

## What would make it more of an agent, in order

**1. It cannot see the weaver's reaction to its own work.** It looks at a
design, forms a view, and then the weaver says "no, too heavy" — and nothing
carries that forward. A preferences memory ("this weaver likes open fields,
dislikes jaal grounds") applied across a session would change the feel more
than any new tool.

**2. It cannot act on a schedule or in the background.** Everything is one
request, one turn. Real agentic work means "try six variations while I get
lunch". The search already runs unattended; nothing lets it run detached.

**3. It has no access to the mill's own past work.** Every conversation starts
from nothing. Being able to say "like the one we did for the Coimbatore order"
needs a design library with saved specs — which checkpoints are now most of the
machinery for.

**4. It cannot read a reference image the weaver points at.** Vision exists but
is only pointed at its own output. "Make something in this spirit" from an
uploaded photo is a natural request it currently cannot take.

---

## One thing to watch

**24 tools, ~6,100 tokens per round.** Prompt caching covers this on Claude.
On a local 70B this is well past where tool-selection accuracy degrades — if
you go that route, split design tools and canvas tools into two modes rather
than offering all 24.

Still unbuilt: sharpening uploads. Vectorisation remains the version worth
doing, and it matters more now — all the canvas work would then apply to
scanned designs at full quality rather than at whatever the scan happened to
have.
