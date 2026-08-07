# Assistant page — agentic rebuild

**361 tests passing**, all offline. Copy the contents of this folder into your
repo root; paths already match.

---

## Turn it on

**Claude** — `config.json` next to `app.py`:

```json
{ "anthropic_api_key": "sk-ant-..." }
```

**Local Llama** — no key needed:

```json
{ "llm_provider": "ollama", "llm_model": "llama3.3:70b" }
```

Then `python run.py` → `http://localhost:5000/agent`.

For the agent to see its own designs, use a vision model. Claude has it. For
local, `llava` / `qwen-vl` / `gemma3` are auto-detected; anything else needs
`"llm_vision": true`. Without vision the agent skips looking and says so —
it does not pretend.

The page now shows which backend is live and whether it can see.

---

## The activation bug is fixed

`/api/assistant-status` called `assistant_engine.is_available()`, which only
looks for `ANTHROPIC_API_KEY`. A local-model setup was shown *"the assistant
needs an API key"* over a perfectly working backend. It now asks the provider
layer. This was my miss when I built the provider layer — I updated one gate and
not the other.

The offline notice also gave Anthropic-only instructions. It now covers both.

---

## What makes it feel like an agent now

**You can see the design.** This was the real gap — a design tool where the
weaver read *prose about* the design. There is now a live panel beside the
conversation, refreshed after every tool call, showing the converted view when
one exists (what will actually be woven, not the artwork it came from). It is
cache-busted, because a cached preview shows the previous design and makes a
refinement look like it did nothing.

**Work streams as it happens.** `/api/agent/stream` is server-sent events, and
`converse()` takes an `on_event` callback. You watch "Planning the job →
Working out the best design for this width → Looking at what I made → Adjusting
the design" arrive one at a time. A full turn runs a dozen tools over a minute;
one JSON blob at the end is indistinguishable from a hang, and it hides the
work, which is the part worth seeing.

**The agent writes a plan and ticks it off.** New `plan_work` tool. It is asked
to call it first on any multi-step job, in the weaver's language — "work out the
finished width", not "call loom_geometry". Not bookkeeping: someone watching an
agent work for a minute cannot tell whether it is nearly done or has lost the
thread. It also keeps the model honest, since a plan stated up front is one it
can be held to.

**Suggested openers and follow-ups** as clickable chips, so nobody stares at an
empty box wondering what this thing can do.

**Smaller things:** a Stop button; the session survives a reload (it lived an
hour server-side but the token was only in a JS variable); a failed turn gives
your message back instead of swallowing it; tool names are no longer printed as
`auto_design · look_at_design`, which is a stack trace, not information; and
there is a link through to the BMP editor.

**Designing from scratch is a real session.** The page used to paint an 8×8
white PNG and post it as an upload so it could reuse that code path — so a
generate-from-scratch job masqueraded as a conversion and you were shown a white
square as your design. There is now `/api/agent/blank`, and tools that need an
image refuse with a sentence the model can act on rather than a `NoneType`
traceback.

---

## New routes

| Route | Does |
| ----- | ---- |
| `POST /api/agent/blank` | Session with no upload, to design from scratch |
| `GET /api/agent/preview` | The current design as PNG |
| `GET /api/agent/stream` | One turn as server-sent events |

`/api/agent/message` still works unchanged, so anything else calling it is fine.

---

## Files

**New:** `design_studio.py`, `llm/` (5), `tools/test_studio.py`,
`test_agentic.py`, `test_agent_page.py`, `test_llm.py`
**Changed:** `agent_engine.py`, `app.py`, `templates/agent.html`,
`tools/test_agent.py`

```
python tools/test_agent_page.py   # 34
python tools/test_agentic.py      # 44   (slowest, ~4 min)
python tools/test_studio.py       # 52
python tools/test_llm.py          # 49
python tools/test_agent.py        # 95
```

Also delete `README-AI-DESIGNER.md` and `changes.patch` from your repo root —
those got committed last time and are notes, not code.

---

## What to watch for on your first real run

**`look_at_design` is the piece most likely to disappoint.** A model that always
says "looks good" adds nothing. The prompt tells it to say what it saw including
when the design looked fine, and not to invent a flaw to seem thorough. Whether
that lands is a live question the test suite cannot answer.

**SSE and Flask's dev server.** Streaming works on the threaded dev server, but
if you put this behind nginx or a proxy later, buffering will break it. The
`X-Accel-Buffering: no` header is already set for nginx specifically.

**Cost.** Agentic turns call the model far more than a chat turn — a dozen
rounds where there used to be two. Prompt caching is on, which covers the fixed
prefix, but keep an eye on the first few conversations. `session['usage']`
accumulates the token counts if you want to check.

**Still not built:** sharpening uploads. Generative upscalers invent plausible
detail, which is fine in a photo and not fine in a manufacturing instruction.
Vectorisation is the version worth building, and it would also let
`refine_design` work on scanned designs instead of only generated ones.
