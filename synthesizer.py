"""
Prompt construction.

This module builds the information that is sent to the LLM.

The goal is to give the LLM useful evidence without directly telling it what
the function should be. The prompt can include:

    - selected input/output observations
    - candidates that have already been tested and rejected
    - reasons why previous candidates failed
    - common failure patterns
    - residual information
    - visual evidence when the visual fallback is used

The prompt also defines the rules for generating new hypotheses. For example,
the LLM must return exact Python expressions, check them against the given
observations, and propose structurally different candidates.

The agent avoids giving uncertain guesses such as "this looks logarithmic".
Instead, it provides observations and verified failure information so that the
LLM can form its own hypotheses.

In short:

    collected evidence
        -> organize useful information
        -> build the prompt
        -> ask the LLM for new candidate functions
"""
from __future__ import annotations

from planner import format_observations


def build_prompt(n_args, observations, k, refuted=None, common_failure=None,
                 residual_note=None, vision_panels=0, trim_refuted=False,
                 evidence=None):
    variables = "x" if n_args == 1 else "x, y"
    body = [
        f"An unknown deterministic function f({variables}) was probed.",]
    if evidence:
        body.append("\n" + evidence)
    body += [
        f"\nOBSERVATIONS (full precision as returned):\n\n"
        f"{format_observations(observations)}",
        "\nInfer the exact closed-form implementation.",
    ]

    if refuted and trim_refuted:
        # When the visual channel is engaged the family itself is in doubt, and a
        # long list of dead candidates anchors the model inside exactly the
        # region the picture is meant to pull it out of. Keep the count and the
        # shared failure mode; drop the individual corpses.
        body.append(f"\n{len(refuted)} candidates have already been tested against "
                    f"the black box and refuted. They are not listed individually "
                    f"because they were all of one kind; what they shared is below.")
    elif refuted:
        body.append("\nThese candidates were already tested against the black box "
                    "and REFUTED. Do not repeat them, or minor variations of them:")
        for expr, rate, why in (refuted if not trim_refuted else []):
            rate_s = f"  [agrees on {rate:.0%} of sampled points]" if rate is not None else ""
            body.append(f"  - {expr}{rate_s}")
            if why:
                body.append(f"      reason: {why}")

    if common_failure:
        body.append(f"\nWHAT ALL REFUTED CANDIDATES HAVE IN COMMON:\n  {common_failure}")

    if vision_panels:
        from vision import VISION_NOTE
        body.append(VISION_NOTE.format(panels=vision_panels))

    if residual_note:
        body.append(f"\nRESIDUAL ANALYSIS of the closest candidate:\n  {residual_note}")

    body.append(f"""
Rules:
- Use only the variable(s) {variables} and functions from Python's math module,
  written with the prefix (math.log, math.sqrt, math.sin, ...).
- Give the simplest EXACT expression consistent with ALL observations above.
  Do not return a numerical approximation or a fitted polynomial.
- Before answering, check your expression against every observation listed.
- Give {k} candidates, ranked most likely first. They must be STRUCTURALLY
  DIFFERENT from one another - different function families, not the same form
  with different constants. Candidates that agree with each other everywhere
  are worth no more than one candidate.
- Reply with exactly {k} lines and nothing else:""")
    body.append("\n".join(f"CANDIDATE {i}: <python expression>" for i in range(1, k + 1)))
    return "\n".join(body)


def summarise_common_failure(oracle, n_args, refuted_fns, rng):
    """What the whole committee got wrong is stronger than what any one
    candidate got wrong -- a shared failure mode rules out a family, a single
    counterexample rules out a point. This is only derivable once several
    candidates have died together."""
    from adjudicate import safe_eval
    if n_args != 1 or not refuted_fns:
        return None
    scale_pts = [10, 1000, 100000, 10000000]
    rows, truths = [], []
    for x in scale_pts:
        t = oracle.val(x)
        if t is None:
            continue
        truths.append((x, t))
        rows.append([safe_eval(f, (x,)) for f in refuted_fns])
    if len(truths) < 3:
        return None
    lo_x, lo_t = truths[0]
    hi_x, hi_t = truths[-1]
    all_over = all(v is not None and abs(v) > abs(hi_t) * 5
                   for v in rows[-1] if v is not None)
    all_under = all(v is not None and abs(v) * 5 < abs(hi_t)
                    for v in rows[-1] if v is not None)
    trend = (f"Between x={lo_x:g} and x={hi_x:g} the true function goes from "
             f"{lo_t!r} to {hi_t!r}.")
    if all_over:
        return (trend + " Every refuted candidate grows far faster than that over "
                        "the same range. The true growth is much slower than any of them.")
    if all_under:
        return (trend + " Every refuted candidate grows far slower than that over "
                        "the same range.")
    return trend + " None of the refuted candidates matches this trend."
