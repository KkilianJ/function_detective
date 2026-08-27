# Technology Use Statement

> **Read this through and adjust it to your own experience before submitting.**
> Items marked ⚠️ must be filled in or verified by you. Everything else records
> what actually happened while building this; keep only what is true for you.

---

## 1. Tools used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic — chat and agentic mode) | Understanding the assignment, working out the architecture, generating first-draft code, drafting documentation |
| ⚠️ *(add anything else — GitHub Copilot, ChatGPT, Cursor, Codex)* | |

---

## 2. What each was used for

**Understanding the problem.** The assignment PDF went in first. The initial
explanations did not land; what built the intuition was asking for a minimal
runnable example — one function, a fake LLM, every step printed. That produced
no submitted artefact but determined every design decision after it.

**Architecture.** Before writing any of the six functions I worked through two
questions: how should the agent decide which inputs to try, and how should
"confident enough" be quantified. The answers — a structural battery that
chooses probe points, and a held-out set with a PAC-derived sample size —
became the skeleton of the whole system.

**Code.** First drafts of every module were AI-generated and then reviewed
section by section. Section 4 lists what that review changed.

**Documentation.** The README was drafted with assistance; every claim it makes
about the code was checked against the implementation.

---

## 3. How the tools changed the approach

**Testing came first, and shaped everything else.** With a spending cap,
debugging against the live API risks exhausting it before anything works. The
LLM backend was therefore made swappable, so `--mock` exercises the entire loop
offline. Probing, filtering, error handling and the controller were all finished
before a paid call was made.

That decision produced an unplanned benefit. The mock backend was not
implemented as a stub returning canned answers but as a genuine least-squares
fitter receiving exactly what the real model receives. It therefore doubles as a
**non-LLM baseline**, and its two distinct failure modes turned into the
clearest evidence of what the LLM contributes: it fails outright on `log(x²+1)`
and `√(x²+3y²)` because it has no function library, and on the polynomials it is
numerically right but never *exactly* right — which is precisely the difference
between fitting and identifying.

**Then the cost model was checked rather than assumed.** Published pricing for
`gpt-5.6-luna` is $0.20 per million input tokens and $1.20 per million output;
a full six-box run costs about $0.0006. The EUR 5 ceiling is not the hard
constraint it first appears to be. Calls are still minimised, but the
justification changed from "we cannot afford more" to "the marginal information
in the n-th counterexample decays while the cost of a call stays flat, and free
local analysis is exact where a paid call is only probable." That reframing
moved effort from *recovering from failure* to *not needing to*: richer
probing, full float precision, several candidates per call, free elimination.

**Prompt design changed last.** The first prompt just dumped a table and asked
what the function was. Four additions improved it: refuted candidates, the
specific counterexample, a mandatory output format, and a demand for
*structurally different* candidates rather than parameter variations of one
form.

---

## 4. Reviewing, testing and correcting the AI's output

Working rule: for every generated line, I have to be able to say why it is
there. Anything I could not explain was either understood or removed. The
following are concrete defects found and fixed.

### 4.1 Circular verification — the most important fix

The first implementation verified candidates against the same points that had
been sent to the model. That is circular: the model has already seen those
values, so of course its expression reproduces them. "Verification passed"
carried no information at all.

Fixed by extracting a held-out set that never enters a prompt.

### 4.2 Held-out contamination — the same bug, one level subtler

The fix above was still not enough. The held-out set was a **fixed list**, and
every counterexample fed back to the model *came from that list*. Once
`f(4.7)` had been quoted in a prompt, `4.7` was no longer held out — yet
verification kept treating it as if it were. The held-out set was being burned
down while still being trusted.

Fixed by generating verification points fresh on every check rather than
drawing from a finite fixed pool. A fixed held-out set is structurally unsound
whenever counterexamples are fed back, because it is a consumable resource
under continuous consumption.

### 4.3 Finite differences falsely converging — "locally polynomial" is not "polynomial"

The reconnaissance module classified `exp(x/10)` as a degree-5 polynomial. With
a small step the fifth difference of a slowly varying function is around `1e-8`,
which read as "constant".

The underlying issue is general: **every smooth function is locally polynomial**
(Taylor). A converging difference table is therefore not evidence on its own.

Fixed with the same propose-then-verify pattern the whole agent runs on: the
differences *propose* a degree, that degree interpolates through `deg+1` points,
and the resulting polynomial must then predict a **far** point that took no part
in the fit. `exp` misses by a factor of four; `x³ − 2x` is exact to the digit at
`x = 81`.

### 4.4 A periodicity detector that was wrong twice, in opposite directions

**First version: everything was periodic.** After removing only a linear trend,
a quadratic or logarithm still leaves a large smooth residual whose FFT peak
lands in the lowest bin — producing a "period" equal to the whole sampling
window. That is leftover trend, not periodicity.

**Second version: nothing was periodic.** Rejecting periods comparable to the
window fixed the false positives but introduced false negatives: FFT bin
resolution is too coarse to land on the true period, and `f(x+T) − f(x)` is not
constant when `T` is slightly off.

**Third version works**, with three corrections: reject candidate periods
comparable to the window length; refine `T` by local search around the FFT
proposal; and confirm by requiring `f(x+T) − f(x)` to be **constant** — constant
rather than zero, so a superimposed linear trend is tolerated. It now recovers
`2π` from `sin(x) + x/2` and correctly rejects `sin(x)·x`, which is
amplitude-modulated and looks periodic but is not.

### 4.5 The probe itself overflowing

The overflow-decade test climbed `10.0 ** k`. At `k = 309` the *probe
construction* raised `OverflowError` before the black box was ever called — my
diagnostic crashed and it looked like a property of the function under test.
Separately, `float(v)` overflowed on the factorial function's huge integer
results. Both are now caught, and the distinction matters: an error raised while
building a probe is a bug in the harness, an error raised by the black box is
data.

### 4.6 Unsandboxed `eval()`

The generated draft called `eval()` directly on the model's output string. Model
output is untrusted input, so this is arbitrary code execution — a one-line
demonstration listed the entire filesystem. Replaced with three layers:
`compile(..., "eval")` to permit a single expression only, an emptied
`__builtins__`, and a namespace containing nothing beyond pure `math` helpers.

### 4.7 A leak I nearly shipped

Arity discovery works by calling with the wrong number of arguments and reading
the `TypeError`. That exception's message is
`func_6() missing 1 required positional argument: 'y'` — **it contains the
function's name**, which the assignment forbids the model from learning. Probing
now records the exception **type** only, never the message.

### 4.8 An under-determined system in the mock fitter

The first probe round yields four observations while the two-variable fit uses a
six-term basis. `numpy.linalg.lstsq` returns a numerically valid but meaningless
solution. Guarded so an under-determined round emits a placeholder and gets
refuted normally.

### 4.10 A capability test that was itself invalid

To decide whether a visual fallback was worth building, I tested whether the
deployment accepts images. It replied, so it looked supported. But the reported
usage was **27 prompt tokens** — fewer than the question text alone should cost.
The image was evidently not being read; the reply was a coin flip on a two-way
question, and it happened to be **wrong**.

The fix was to change what the experiment measures. Instead of trusting the
answer, send the same question twice — once as plain text, once with the image —
and compare prompt token counts. The delta *is* the image.

```
text only     prompt=25    answer=''
64px/low      prompt=31    answer='Right'   -> +6 tokens, and wrong
512px/low     prompt=334   answer='Left'    -> +309 tokens, and correct
512px/high    prompt=334   answer='Left'    -> +309 tokens, and correct
```

So vision works, but only above a resolution threshold; my original 64×64 test
image was compressed to nothing. **The methodological lesson is that a
two-outcome question cannot validate a capability, because chance passes it half
the time. The token count is the only witness that cannot guess.**

### 4.11 A trigger condition that fired on everything

The first version of the visual trigger asked "did any test abstain?" — and
since several always abstain, it fired on a confirmed linear function. Sending a
rendered chart to identify `3x - 4` is both wasteful and impossible to defend.

The right question is the opposite one: **"did any test establish the family?"**
Only if all of them failed is there anything for a picture to add. Rewritten that
way it fires on exactly three of the seventeen test functions — the
amplitude-modulated impostor, an oscillation with a pole, and the integer-only
factorial — and on none of the six targets, because the router resolves all six
for free.

### 4.9 ⚠️ To be completed after your live API run

> Record what you actually observe against the real model. Likely candidates:
> replies with prose around the expression (the parser tolerates this), `ln`
> written instead of `math.log`, a missing `math.` prefix, fewer than `k`
> candidates returned, or candidates that are parameter variations rather than
> structurally distinct despite the instruction. Note the behaviour and how you
> handled it — the more specific, the better.

---

## 5. Representative prompts

**(a) Building understanding — no code produced, but nothing after it would
have been possible without it:**

> I have no background in this kind of task. Walk me through one concrete small
> example end to end: one function, a fake LLM, every step printed, so I can see
> exactly what "probe, guess, verify" means in practice.

**(b) An architecture question:**

> How should the agent decide which inputs to try? Why not sample randomly?
> Answer separately for one and two variables, and say what structural property
> each batch of probe points is meant to expose.

**(c) Auditing the AI's own output — this is the prompt that found §4.1:**

> Is there a circularity problem in this verification logic? If the points used
> for verification are the same ones fed to the LLM, what does "verification
> passed" actually tell us?

**(d) Pushing on a boundary I was uneasy about:**

> If my program computes the structure itself and tells the model, is the model
> still doing the deduction? Where exactly is the line?

The answer to (d) became a design rule that runs through the whole
implementation: *computed structure chooses probe points and refutes candidates;
it never enters the prompt as a claim.* Inferences appear in a prompt only as
exclusions backed by a refutation, because a wrong exclusion merely rules out
too little, while a wrong suggestion steers every later round with nothing to
retract it.

**(e) The prompt the agent itself sends** — see `synthesizer.py`; an example
appears in the README.

---

## 6. Ownership

I can account for each design decision: why probe points are chosen rather than
sampled, why the verification set must be held out *and* regenerated, why the
stopping condition is measurement rather than a round count, why the model is
never asked whether it is confident, why `eval` needs a sandbox, and why
agreement rates are meaningless without a stated tolerance and range.

I am equally clear about the limits. Passing verification demonstrates
ε-approximate agreement over a stated range, not symbolic equivalence.
Probing can only eliminate candidates, never generate them, so the diversity of
the initial committee is load-bearing. Several asymptotic diagnostics abstain
rather than risk a wrong reading, which is the designed behaviour but still a
miss. All of this is documented in the README's Known Limitations.

⚠️ *If any statement above does not match what you can actually explain, either
go back and understand it or rewrite it to match where you really are. The value
of this document is its honesty.*
