# The Cairo fix — and why my message was wrong

**629 tests passing.** Copy the contents of this folder into your repo root.

---

## Your actual fix

pip is not the answer here. **cairosvg is installed correctly.** What's missing
is Cairo itself — `libcairo-2.dll` — which is a Windows system library that pip
does not and cannot ship.

1. Download the GTK3 runtime installer from
   [github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases)
2. Run it and **tick "Set up PATH environment variable"**
3. Close and reopen PowerShell, then restart the app

If you use conda: `conda install -c conda-forge cairo`.

---

## In the meantime, you can work

`cairosvg` is used in **exactly one function** in the whole product —
`motif_library.render()`, which draws generated motifs. It only affects the
Assistant's `auto_design` and the motif library.

Everything on the screenshot works right now: upload a saree design, detect
colours, convert, assign shuttles, download BMPs. Butta, Border, BMP Editor and
Tracing Guide are all unaffected.

---

## Two bugs in what I shipped you

**My message gave you advice that couldn't work.** It said `pip install
"cairosvg>=2.7"`, pip replied "Requirement already satisfied", and the app kept
insisting the package was missing — with nothing anywhere to explain the
contradiction. That's worse than no message.

The checker treated every import failure as "not installed". There are two
kinds, and they read nothing alike:

| | Cause | Fix |
| --- | --- | --- |
| `No module named 'skimage'` | Package absent | pip |
| `no library called "cairo-2" was found` | Package present, native library absent | **not pip** |

Now told apart by exception type and message text, with platform-specific
instructions for Windows, macOS and Linux — and when the problem is a native
library, **no pip command is offered at all**, because offering one is what sent
you round in a circle.

**The banner overstated the damage.** "This install is incomplete" in red,
across the top of the Generator, where your entire workflow functions. A warning
that cries wolf gets dismissed — and then the one that matters gets dismissed
with it.

Dependencies are now tiered. `skimage` is **core**: without it, real things
break. `cairosvg` is a **feature**: one capability stops, the rest is fine. The
banner reads accordingly — red "This install is incomplete" only for core, amber
"One feature is unavailable. Everything else on this page works." otherwise.

---

## Also fixed

- `_dep_message` only matched `No module named`, so a Cairo `OSError` would have
  passed through raw to any page. It now recognises native-library failures too
  — `no library called`, `dll load failed`, `cannot load library`.
- `motif_library.render()` guards its import, so a Cairo problem produces an
  explanation rather than a raw `OSError` from deep inside cairocffi.
- Every fix now ends with "then restart the app" — installing into a running
  process changes nothing until it restarts, and someone who installs and sees
  no difference concludes the fix failed.
- Two test bugs of my own: `platform` was missing from the stdlib list, and a
  message assertion I'd broken while rewriting.

---

## Files changed

**Changed:** `deps.py`, `app.py`, `motif_library.py`, `templates/_nav.html`,
`tools/test_install.py`

The rest of the folder is the current build so it copies in one go.

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        #  95
python tools/test_agent_page.py   #  85
python tools/test_install.py      #  53
python tools/test_studio.py       #  52
python tools/test_llm.py          #  49
python tools/test_agentic.py      #  44
python tools/test_hardening.py    #  43
python tools/test_nav.py          #  37
python tools/test_assistant.py    #  34
python tools/test_auto_convert.py #  16
```

---

## What I'd still say plainly

Two rounds of install failures now, both found by you rather than by me,
because **my tests run in an environment where everything is installed.** I've
added what I can — a check that every import is declared, tiering so warnings
match severity, and diagnosis for native-library failures — but I cannot
reproduce a Windows machine from here. Your environment will keep being the
place these surface.

And still unverified: whether the assistant *behaves* well. Every test scripts
the model. Once Cairo is in and `auto_design` runs, that's the thing worth
checking — give it a brief, then deliberately ask for something that won't
weave, and see whether it tells you plainly.
