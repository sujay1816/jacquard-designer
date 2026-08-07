# Upload / download audit — final files

**533 tests passing.** Copy the contents of this folder into your repo root.
This is the complete current build, not just a delta.

---

## The serious one: transparent PNGs silently destroyed the design

A design exported with a transparent background — the normal output of
Illustrator, Photoshop, Figma, CorelDRAW, almost anything — uploaded, converted,
and produced a **completely blank BMP**. The reported verdict was `warn`, not an
error, so there was nothing obviously wrong until the cloth was on the loom.

The cause: PIL's `convert('RGB')` on an RGBA image **discards the alpha and
keeps whatever RGB sat underneath it** — `(0, 0, 0)` in almost every exporter.
The whole canvas went black, the motif disappeared into it, and the converter
correctly found no contrast to work with.

```
before:  mean luminance 0.0   →  0% ink  →  -100% drift  →  "warn"
after:   mean luminance 182   →  29% ink →     0% drift  →  "ok"
```

Fixed by compositing onto white before flattening. Transparent means no ink,
which means bare cloth, which means white. EXIF rotation is applied too, so a
design photographed in portrait is no longer converted sideways.

## There were two upload loaders, and the wrong one was winning

`app.py` had `_open_upload` defined twice — once taking bytes, once taking a
file object. In Python the later definition wins, so the first was dead code and
every route calling it with bytes broke the moment I touched it. That is also
exactly how one upload path ends up without the fixes the other one got.

Collapsed to a single loader accepting bytes, a Werkzeug upload, or an open
Image. All fourteen call sites across the app now share it — so the transparency
fix reaches the BMP editor and the other converters, not just the assistant.

## Smaller download fixes

**A failed download showed raw JSON.** Clicking Download after an edit had
cleared the files navigated straight to the endpoint and put
`{"success": false, ...}` on screen. It now fetches, checks, and puts the
problem in the conversation where the weaver can act on it.

**Zip names kept the source extension** — `saree.png_bmps.zip` looks like a
mistake, and on Windows an archive named for a PNG is one people hesitate to
open. Now `saree_bmps.zip`, with filesystem-hostile characters stripped
(`a/b:c*.jpg` → `abc_bmps.zip`). Unicode names survive intact.

**Drag-and-drop bypassed `accept="image/*"`.** That attribute only filters the
picker, so a 40 MB PDF dropped on the panel uploaded in full before the server
refused it. Now checked client-side, with a message that says what to drop
instead.

---

## What was already correct

Worth saying, since I went looking: no-file, corrupt-file, empty-file and
oversized uploads were all refused with clear messages. Greyscale, palette and
CMYK images work. Zip integrity is sound, every BMP inside is genuinely 1-bit,
and bad or missing tokens are refused on every download route.

---

## Turning it on

`config.json` next to `app.py`:

```json
{ "anthropic_api_key": "sk-ant-..." }
```

or, for a local model with no key:

```json
{ "llm_provider": "ollama", "llm_model": "llama3.3:70b" }
```

Then `python run.py` → `http://localhost:5000/agent`.

---

## Tests

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        #  95
python tools/test_agent_page.py   #  85
python tools/test_studio.py       #  52
python tools/test_llm.py          #  49
python tools/test_agentic.py      #  44   (~4 min)
python tools/test_assistant.py    #  34
python tools/test_nav.py          #  37
python tools/test_auto_convert.py #  16
```

Housekeeping: seven `README-*.md` files and `changes.patch` are sitting in your
repo root from previous drops. They are notes, not code — worth moving to
`docs/` or deleting.

---

## Still open

**Nothing has run against a live model.** Every suite is scripted. The agentic
loop and `look_at_design` are the two pieces no offline test can validate — a
model that always says "looks good" passes all 533 of these and is useless.

**No cross-turn memory.** The weaver says "too heavy", the agent adjusts, and
next turn it suggests the same thing again.

**Explored candidates are still text-only.** `explore_designs` builds three
alternatives and the weaver only reads about them, even though the thumbnails
already exist in the session. Smallest remaining change with the largest effect
on something calling itself a design tool.

**Sharpening uploads is unbuilt.** Vectorisation is the version worth doing —
generative upscalers invent detail, which is fine in a photo and not fine in a
manufacturing instruction.
