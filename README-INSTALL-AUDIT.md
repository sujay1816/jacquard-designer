# Final audit — install health

**618 tests passing.** Copy the contents of this folder into your repo root.

---

## Your immediate fix

```
pip install scikit-image
```

`scikit-image` **is** in `requirements.txt`. Your install didn't finish — most
likely it failed partway (scikit-image builds slowly and errors on some
Windows/Python combinations) and the failure scrolled past.

---

## Why my audit missed it, honestly

Every one of the 576 tests ran in my sandbox, where `skimage` is installed.
**The tests verified the code. They never verified the install.** No amount of
fuzzing the engine would have found this, because on my machine the import
always succeeded.

That's a real gap in how I was testing, not an oversight in one file — which is
why the fix is a new category of test rather than a patch.

---

## Why it surfaced the way it did

Three separate things had to go wrong for you to see that message:

1. **The import is inside the function.** `skimage.measure` is imported where
   the reduction happens, not at module load — so the app started fine, every
   page rendered fine, and it only failed when you pressed the button.
2. **Nothing checked the install.** No startup check, no health endpoint.
3. **The message named the wrong thing.** "No module named 'skimage'" is true
   and useless: the package you install is called **scikit-image**. Different
   word. Nothing told you to install anything at all.

---

## What now prevents this class of error

**A startup check.** `run.py` verifies every required package before serving
and prints what's missing, what it's for, and the exact pip line:

```
Some required packages are missing.

  skimage    needed for Butta Studio, Border Studio and thread reduction

  Fix:  pip install "scikit-image>=0.21.0"
  Or:   pip install -r requirements.txt
```

**A banner on every page.** In the shared `_nav.html`, so all six pages get it
from one place — a per-page copy is how `/border` ends up with the check and
`/butta` without it. It calls `/api/health` on load, so an incomplete install is
reported *before* ten minutes of work, not after.

**Every error message rewritten at the choke point.** Nearly all routes end in
`except Exception as e: return _json_error(...)`, so fixing `_json_error` covers
all thirty-one at once. What you saw becomes:

> Preview failed: a required package is not installed. skimage is needed for
> Butta Studio, Border Studio and thread reduction. Run: `pip install
> "scikit-image>=0.21.0"` (or: `pip install -r requirements.txt`), then restart
> the app.

**A test that the dependency table matches reality.** `tools/test_install.py`
parses every import in the codebase with Python's own AST and asserts each one
is declared. If someone adds a package and forgets `requirements.txt`, the suite
fails here rather than on a weaver's machine.

---

## Two more bugs found in the same sweep

**The Generator threw away real error messages.** `index.html` did
`if (!res.ok) throw new Error('Server error ' + res.status)` — so any 4xx showed
*"Server error 400: BAD REQUEST"* instead of the actual explanation. It would
have hidden the new pip instruction too. It now reads the body first.

**My own test scanner had a bug.** It matched `from under a second to nine`
inside a docstring as an import of a package called `under`. Fixed by parsing
with AST instead of regex — the tokenizer knows what is code and a pattern
doesn't.

---

## Full product sweep

| Check | Result |
| --- | --- |
| All 6 pages render | 200, no template errors, no tracebacks |
| All 31 API routes, empty/bad input | JSON 4xx, **zero 500s**, zero HTML |
| Every page carries the health banner | Yes |
| Every import declared in requirements | Yes |
| 1,554 hostile tool calls | 0 crashes |

---

## Files changed

**New:** `deps.py`, `tools/test_install.py`
**Changed:** `app.py`, `run.py`, `templates/_nav.html`, `templates/index.html`

The rest of the folder is the current build so it copies cleanly in one go.

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        #  95
python tools/test_agent_page.py   #  85
python tools/test_studio.py       #  52
python tools/test_llm.py          #  49
python tools/test_agentic.py      #  44
python tools/test_hardening.py    #  43
python tools/test_install.py      #  42
python tools/test_nav.py          #  37
python tools/test_assistant.py    #  34
python tools/test_auto_convert.py #  16
```

---

## What "no errors anywhere" can and cannot mean

The install-time and runtime error classes are now covered: I can't make the
engine crash, no route 500s, no page renders broken, and a missing package
reports itself with an instruction.

**What no test here can reach is whether the model behaves well.** Every suite
scripts it — the model is a stub returning what the test dictates. So these
prove the product can't be broken *by* the model. They say nothing about whether
`look_at_design` gives a real critique or just says "looks good", or whether the
honesty rules hold when a conversion genuinely fails.

One live conversation settles both, and it's the highest-value thing left: run a
brief end to end, then deliberately ask for something that won't weave, and see
whether it tells you plainly.
