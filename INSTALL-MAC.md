# Installing on macOS

Whole thing takes about ten minutes, most of it waiting on downloads.

---

## 1. Find out which Mac you have

```bash
uname -m
```

`arm64` is Apple Silicon (M1/M2/M3/M4). `x86_64` is Intel. It matters for one
step near the end, so it's worth knowing now.

---

## 2. Homebrew

Skip if `brew --version` already works.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**On Apple Silicon, read the last few lines of its output.** It prints two
`echo` commands to add Homebrew to your PATH, and it does *not* run them for
you. If you skip that, `brew` won't be found in a new terminal:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

---

## 3. Python

macOS ships an old Python that you shouldn't build on.

```bash
brew install python@3.11
python3 --version        # expect 3.11.x or newer
```

---

## 4. Cairo — do this before pip, not after

This is the step that bit on Windows, and it bites the same way here.
`pip install cairosvg` installs the Python wrapper but **not Cairo itself**,
which is a system library. Installing it first avoids the confusing state where
pip reports success and the app says the package is missing.

```bash
brew install cairo pango libffi
```

---

## 5. The app

```bash
cd ~/Downloads/jacquard-designer      # wherever you put it
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A virtual environment isn't strictly required, but on a Mac you'll otherwise
hit "externally-managed-environment" errors from pip, and it keeps this project's
packages away from anything else you're doing.

---

## 6. Run it

```bash
python run.py
```

It opens `http://localhost:5000` by itself. The terminal prints a startup check
— if anything is missing it names the package, what it's for, and the exact
command that fixes it.

---

## If Cairo still isn't found on Apple Silicon

You'll see this even though `brew install cairo` succeeded:

```
no library called "cairo-2" was found
```

Homebrew on Apple Silicon installs to `/opt/homebrew`, and `cairocffi` doesn't
search there — so the library is genuinely present and genuinely not found.

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH
```

To make it stick:

```bash
echo 'export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH' >> ~/.zshrc
```

Then open a **new** terminal and run the app again. On Intel Macs this doesn't
arise — Homebrew uses `/usr/local`, which is already searched.

Check it worked:

```bash
python3 -c "import cairosvg; print('cairo ok')"
```

---

## Two things worth doing on a Mac specifically

**iPhone photos.** If designs arrive as HEIC from a phone:

```bash
pip install pillow-heif
```

Without it the app tells you the file type isn't readable and names this
package. Not needed if your designs are JPG or PNG.

**The first colour detection is slow.** macOS has a known 10–30 second hang the
first time scikit-learn's KMeans runs with parallel workers. `run.py` already
sets `LOKY_MAX_CPU_COUNT=1` and `OMP_NUM_THREADS=1` to prevent it — which is
why you should start the app with `python run.py` rather than
`flask run` or `python app.py`.

---

## The API key

`config.json` next to `app.py`:

```json
{ "anthropic_api_key": "sk-ant-..." }
```

Or in the terminal, which takes precedence:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`config.json` is already in `.gitignore`.

For a local model instead, no key needed:

```json
{ "llm_provider": "ollama", "llm_model": "llama3.3:70b" }
```

---

## Checking the install

```bash
curl -s http://localhost:5000/api/health
```

`"ok": true` and empty `missing` means everything is there. Or just look at the
page — an incomplete install shows a banner at the top of every screen, and it
distinguishes "the app cannot run" from "one feature is unavailable, everything
else works".

---

## If something goes wrong

| Symptom | Cause |
| --- | --- |
| `command not found: brew` | PATH step in section 2 skipped |
| `externally-managed-environment` | Not in the venv — `source .venv/bin/activate` |
| `no library called "cairo-2"` | The Apple Silicon section above |
| `No module named 'skimage'` | `pip install -r requirements.txt` didn't finish; re-run and watch for errors |
| Port 5000 already in use | macOS AirPlay Receiver uses it — turn it off in System Settings › General › AirDrop & Handoff |

That last one catches people out regularly and looks nothing like a Python
problem.
