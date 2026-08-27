"""
Adjudication: deciding what to do with the model's candidates.

Four filters, ordered cheapest first. Every one of them is free -- they are
local computation against the black box, which costs nothing. Only the call
that produced the candidates cost money.

  a. STRUCTURAL   -- does the candidate contradict an established fact from the
                     router? Instant, and it refutes at the level of a whole
                     family rather than a single point.
  b. CONSISTENCY  -- does it reproduce the observations it was shown? A model
                     can return an expression that fails its own input data;
                     catching that here separates "failed to fit" from
                     "fitted but generalises wrongly", which need different
                     feedback.
  c. VERSION SPACE-- among survivors, probe where they disagree most. One
                     well-chosen free probe can eliminate most of a committee.
  d. PAC          -- the survivor is tested on stratified fresh samples. N
                     samples surviving bounds the disagreement region: passing
                     N independent points means measure < eps with confidence
                     1 - (1-eps)^N. N = 300 gives "under 1%" at about 95%.

Two tolerances, because one number cannot do two jobs. A strict tolerance is
nearly binary and is what decides accept/reject. A loose one is graded and is
what diagnoses how close a wrong answer was.

Winning the committee is not the same as being correct: if every candidate is
wrong, one of them still wins. The absolute PAC check is therefore run on the
survivor regardless.
"""

from __future__ import annotations

import math
import random

import numpy as np

STRICT_TOL = 1e-9        # accept / reject
LOOSE_TOL = 1e-2         # diagnosis only
PAC_SAMPLES = 300
SAMPLE_SCALES = [(-1, 1), (-6, 6), (-50, 50), (-500, 500)]

SAFE_NS = {n: getattr(math, n) for n in dir(math) if not n.startswith("_")}
SAFE_NS.update({"abs": abs, "min": min, "max": max, "round": round, "math": math})


def compile_candidate(expr: str, n_args: int):
    """Model output is untrusted input. Three restrictions: compile in "eval"
    mode so only a single expression is accepted (no statements, assignments or
    imports), evaluate with __builtins__ emptied, and expose nothing beyond
    pure math helpers."""
    names = ("x",) if n_args == 1 else ("x", "y")
    try:
        code = compile(expr, "<candidate>", "eval")
    except (SyntaxError, ValueError):
        return None

    def f(*args):
        scope = dict(SAFE_NS)
        scope.update(dict(zip(names, args)))
        return eval(code, {"__builtins__": {}}, scope)   # noqa: S307
    return f


def close(a, b, tol):
    """Relative comparison with an absolute floor.

    Absolute tolerance is meaningless across scales: near 5e299 consecutive
    floats are 7e283 apart, so no correct implementation could satisfy an
    absolute 1e-9. Relative comparison means 'the leading digits agree', which
    has the same meaning at 1.0 and at 5e299. The max(1, .) floor keeps it
    sane where the true value passes through zero."""
    if a is None or b is None or isinstance(a, complex) or isinstance(b, complex):
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def safe_eval(f, args):
    try:
        v = f(*args)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    except Exception:
        return None


# --------------------------------------------------------------- (a) structural

def structural_conflict(expr, f, profile, oracle):
    """Return a reason string if the candidate contradicts an established fact."""
    n = profile.arity
    if profile.parity in ("even", "odd") and n == 1:
        want_even = profile.parity == "even"
        for x in (1.7, 3.3, 8.1):
            a, b = safe_eval(f, (x,)), safe_eval(f, (-x,))
            if a is None or b is None:
                continue
            ok = close(a, b, 1e-7) if want_even else close(a, -b, 1e-7)
            if not ok:
                return (f"the function is {profile.parity} (verified on the black "
                        f"box), but this candidate is not: it gives {a!r} at "
                        f"x={x} and {b!r} at x={-x}")
    if profile.inflections and n == 1:
        want = profile.inflections[0]
        h = max(abs(want) * 1e-4, 1e-4)
        vals = [safe_eval(f, (want + d * h,)) for d in (-1, 0, 1)]
        if None not in vals:
            d2 = (vals[0] - 2 * vals[1] + vals[2]) / (h * h)
            scale = max(1.0, abs(vals[1]))
            if abs(d2) > 1e-3 * scale:
                return (f"the black box has an inflection point at x={want:.6f} "
                        f"(f''=0 there), but this candidate's second derivative "
                        f"there is {d2:.6g}")
    if profile.symmetric is not None and n == 2:
        a, b = safe_eval(f, (2.0, 5.0)), safe_eval(f, (5.0, 2.0))
        if a is not None and b is not None:
            sym = close(a, b, 1e-7)
            if sym != profile.symmetric:
                return (f"the black box is {'symmetric' if profile.symmetric else 'not symmetric'} "
                        f"in x and y, this candidate is the opposite")
    if profile.accepts_complex is False and n == 1:
        try:
            f(1j)
            return ("the black box rejects complex input, so it uses a math "
                    "module function; this candidate is pure arithmetic and "
                    "accepts complex input")
        except Exception:
            pass
    return None


# ------------------------------------------------------------- (b) consistency

def consistency_failure(f, observations):
    """Does the candidate reproduce the data it was shown?"""
    for args, v, err in observations:
        got = safe_eval(f, args)
        if v is None:
            continue                       # undefined there; not checked
        if not close(got, v, STRICT_TOL):
            sig = ", ".join(f"{a:g}" for a in args)
            return (f"it does not reproduce an observation it was given: at "
                    f"f({sig}) the table says {v!r}, this candidate gives {got!r}")
    return None


# ----------------------------------------------------------- (c) version space

def most_discriminating(cands, n_args, rng, pool_size=2000):
    """The input on which the surviving candidates disagree most. Probing it
    costs nothing and can eliminate most of the committee at once."""
    best, best_spread = None, -1.0
    for _ in range(pool_size):
        lo, hi = rng.choice(SAMPLE_SCALES)
        args = ((round(rng.uniform(lo, hi), 4),) if n_args == 1
                else (round(rng.uniform(lo, hi), 4), round(rng.uniform(lo, hi), 4)))
        vals = [safe_eval(f, args) for _, f in cands]
        if any(v is None for v in vals):
            continue
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / max(1.0, abs(mean))
        if spread > best_spread:
            best_spread, best = spread, args
    return best, best_spread


# --------------------------------------------------------------------- (d) PAC

def stratified_points(n_args, n, rng):
    pts = []
    for i in range(n):
        lo, hi = SAMPLE_SCALES[i % len(SAMPLE_SCALES)]
        pts.append((round(rng.uniform(lo, hi), 4),) if n_args == 1
                   else (round(rng.uniform(lo, hi), 4), round(rng.uniform(lo, hi), 4)))
    return pts


def agreement(f, oracle, n_args, rng, tol, n=PAC_SAMPLES):
    """Fraction of fresh points where candidate and black box agree.
    Also returns where the mismatches sit, which is what tells you whether the
    family is right and only the detail is wrong."""
    hits, tested, bad = 0, 0, []
    for args in stratified_points(n_args, n, rng):
        truth = oracle.val(*args)
        if truth is None:
            continue
        tested += 1
        got = safe_eval(f, args)
        if close(got, truth, tol):
            hits += 1
        elif len(bad) < 400:
            bad.append((args, got, truth))
    return (hits / tested if tested else 0.0), bad, tested


def pac_confidence(n, eps=0.01):
    """Confidence that the disagreement region has measure below eps, given n
    independent samples all agreed."""
    return 1.0 - (1.0 - eps) ** n


# ------------------------------------------------------------------ diagnosis

def residual_shape(f, oracle, n_args, rng):
    """The error is not only a verdict, it often contains the fix.

    A constant residual means the candidate is short an additive constant; a
    linear one means it is short a linear term. Both are repairable locally at
    zero cost, and the structure still came from the model -- the same split
    between skeleton and parameters that LLM-SR uses. A residual with shape but
    no simple form still localises where to probe next."""
    if n_args != 1:
        return None
    xs, res = [], []
    for x in [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]:
        t, g = oracle.val(x), safe_eval(f, (x,))
        if t is None or g is None:
            continue
        xs.append(x)
        res.append(t - g)
    if len(xs) < 5:
        return None
    a = np.array(res)
    if float(a.std()) < 1e-9 * max(1.0, abs(float(a.mean()))):
        return ("constant", float(a.mean()),
                f"the residual is constant at {a.mean():.10g}: the candidate is "
                f"short exactly that additive term")
    k, m = np.polyfit(xs, res, 1)
    if np.allclose(res, k * np.array(xs) + m, atol=1e-8 * max(1.0, float(np.abs(a).max()))):
        return ("linear", (float(k), float(m)),
                f"the residual is linear, {k:.10g}*x + {m:.10g}: the candidate is "
                f"short that linear term")
    worst = xs[int(np.argmax(np.abs(a)))]
    return ("structured", worst,
            f"the residual has structure; it is largest near x={worst:g} and "
            f"smallest at the far end, so the family may be right and only the "
            f"behaviour near the origin wrong")
