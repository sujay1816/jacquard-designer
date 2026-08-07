# Cost per iteration, and what makes it cheaper

**486 tests passing.** Your push was clean — all five core files byte-identical.

---

## What a turn actually costs

Measured, not estimated — by intercepting the real wire payload of a full
agentic turn (plan → geometry → auto_design → look → refine → look →
generate_files → reply, 8 rounds, 2 vision calls).

| | tokens | cost |
| --- | ---: | ---: |
| Before this drop | 57,228 | **$0.120** |
| After | 41,594 | **$0.088** |

At Sonnet-class list pricing ($3/M in, $15/M out, cache read 0.1×).

| Scenario | Cost |
| --- | ---: |
| A simple question (2 rounds, no vision) | $0.02 |
| One full agentic turn | $0.09 |
| A 10-turn design session | $0.88 |
| 100 sessions/month | $88 |

**Prompt caching is doing most of the work** — without it the same turn is
$0.23, so caching alone is a 48% saving. It's already on.

---

## Where the money was going

85% of all input was the *same fixed prefix re-sent every round*. Not history,
not tool results — the system prompt and tool schemas, 6,103 tokens, eight
times.

Tool schemas alone were 4,433 of those 6,103 — **73% of the prefix**.

Three fixes:

**1. Tools are scoped to what can actually run.** A fresh design session is
offered 7 tools instead of 22. Conversion tools can't run without an upload;
canvas tools can't run without a converted design. This removes options that
would have been refused anyway one round later.

*Honest limitation:* a generated design **is** an image, so `convert` and
`inspect_design` legitimately apply to it — the full set returns once a design
exists. The win is concentrated in the opening rounds, not throughout.

**2. Two redundant schemas removed.** `design` is `auto_design` without the
measure-and-improve loop; `generate_design` builds a single motif that
`auto_design` covers with one parameter. Together ~840 tokens *every round* for
options the prompt already told the model not to pick. A redundant option isn't
free even when unused — it's one more thing to choose wrongly between. Both
stay in `_DISPATCH`, so nothing that calls them breaks.

**3. Stale design views are dropped from history.** Each thumbnail is ~390
tokens and stayed in the transcript for every subsequent round. Looking at a
design twice meant paying for the first view repeatedly. Only the newest view
describes the current cloth — older ones show cloth that has since changed, so
keeping them was misleading as well as expensive.

---

## On the model question

**The judgement is where the value is, and it's the part a smaller model does
worst.** Loom safety is enforced in Python and holds on any backend. What
degrades is the honesty behaviour — saying what detail was lost in craft terms,
refusing to call a `warn` conversion good, admitting a refinement made things
worse. That's exactly what this product sells.

If you want to cut cost by model choice rather than by tokens, the split worth
making is **by turn, not by product**: a cheap model for "what's 480 pins at
reed 80" (that's a tool call and a sentence), a strong one for anything
involving a design judgement. The provider layer already supports this — it
would need a per-turn override rather than a per-session one.

**22 tools at ~3,600 tokens is still past where a local 70B holds tool-selection
accuracy.** If you go the Llama route, the scoping added here helps but won't
be enough on its own.

---

## Where the remaining cost is, and it isn't the LLM

`auto_design` at effort 4 renders and scores ~20 candidates. That's ~29 seconds
of CPU per call. At any real volume **your compute bill will exceed your API
bill**, and the latency is what a weaver actually notices. If you want the next
efficiency win, it's caching rendered specs — the render is deterministic, so
an identical spec never needs rendering twice, and the hill climb revisits
neighbouring specs constantly.

---

## The biggest capability gap is still not a tool

No cross-turn memory. The weaver says "too heavy", the agent adjusts, and next
turn it suggests the same thing again. Nothing carries a preference forward.
That's the real distance between this and a full assistant — and it would also
*save* money, by removing the corrections that currently take a round each.

Still nothing has run against a live model. A model that always says "looks
good" passes all 486 of these tests and is useless.

```
python tools/test_canvas.py       # 121
python tools/test_agent.py        # 95
python tools/test_studio.py       # 52
python tools/test_llm.py          # 49
python tools/test_agentic.py      # 44
python tools/test_agent_page.py   # 34
```
