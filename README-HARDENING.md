# Hardening audit — final

**576 tests passing**, including a new adversarial suite. Copy the contents of
this folder into your repo root.

---

## What I attacked it with

Not feature tests — deliberate attempts to break it.

| Attack | Result |
| --- | --- |
| 1,554 hostile tool calls (every tool × 37 malformed argument shapes × 2 session states) | **0 crashes, 0 leaked Python internals** |
| Model calls a tool that doesn't exist | Handled |
| Model sends arguments as a string, or a list, or nothing | Handled |
| Model sends an empty tool name | Handled |
| Two tool calls sharing one id | Handled |
| Backend throws mid-loop | Turn rolled back completely, clean message |
| 100,000-character message | Refused at the route |
| Null bytes, heavy unicode, `{"role":"system"}` injection | Handled |
| Six parallel turns on one session | No crash, history well formed |
| Session expires mid-conversation | Completes its turn |
| A tool fails mid-stream | Marked failed, stream still ends cleanly |
| Backend dies mid-stream | Stream terminates, no hang |
| Bad or missing token on all four GET routes | 400 on every one |

The standard I held it to: a refusal is fine, but a Python internal reaching
the weaver is not — they can do nothing with it, and the model can't correct
from it either. The fuzzer specifically flags `KeyError`, `NoneType`,
`not subscriptable` and friends appearing in any error message.

---

## What I fixed

**Stored history grew without bound.** Only the last 24 messages are ever sent,
but every message of every turn was kept forever — 160 after 40 turns, climbing
in a long session. Now capped at four windows, trimmed at a clean boundary,
because cutting mid-exchange would leave a `tool_results` whose assistant turn
is gone, which both wire formats reject.

**One route still demanded an Anthropic key.** I fixed `/api/assistant-status`
earlier but missed `/api/agent/start` — it told a local-model user to go and buy
an API key while their backend worked fine. There is now a test asserting no
route says this.

---

## What was already sound

Worth stating, since I went looking hard: the fidelity scoring, the label-map
invariant, the undo and checkpoint paths, the file generation, the zip
integrity, the streaming teardown, and every one of the twenty-four tools under
malformed input. The earlier fixes held.

---

## The new suite

`tools/test_hardening.py` — 43 tests, all adversarial. It's the one to run
first when something looks wrong, and the one that will catch a regression in
error handling that a feature test never would.

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        #  95
python tools/test_agent_page.py   #  85
python tools/test_studio.py       #  52
python tools/test_llm.py          #  49
python tools/test_agentic.py      #  44
python tools/test_hardening.py    #  43
python tools/test_assistant.py    #  34
python tools/test_nav.py          #  37
python tools/test_auto_convert.py #  16
```

---

## What "no errors" can and cannot mean here

The engine is robust: I could not make it crash, and 576 tests say the paths
through it behave. That is worth having.

But **every test in this repo is scripted.** The model is a stub that returns
whatever the test tells it to. So these prove the engine cannot be broken by
the model — they prove nothing about whether the model *uses it well*. The two
things no offline test can reach:

**`look_at_design`.** A model that replies "looks good" to every design passes
all 576 of these and is useless. Whether the critique is real is a live
question.

**The honesty behaviour.** The prompt asks it to say what detail was lost in
craft terms, to lead with warnings, to refuse to call a `warn` conversion good.
Nothing here can verify it actually does.

Both need one real conversation to check, and that's the highest-value thing you
can do next. Run a design brief end to end, deliberately ask for something that
won't weave, and see whether it tells you.

Still unbuilt: cross-turn memory, candidate thumbnails in the page, and
vectorising uploads.
