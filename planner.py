"""
Probe planning.

The router produced a profile. This module turns that profile into the set of
inputs to actually probe -- and therefore into the observation table the model
will see.

The governing rule, and the reason this module exists at all:

    DO NOT PUT CONCLUSIONS IN THE PROMPT. PUT IN POINTS THAT MAKE THE
    CONCLUSION VISIBLE IN THE RAW NUMBERS.

Telling the model "this function is even" is an inference; if the router is
wrong it poisons every later round and cannot be retracted. Probing at (x, -x)
pairs so the table literally shows f(3) == f(-3) is a fact, costs the same, and
carries the same information.

The same trick handles the rest: a logarithm is invisible in a table of small
integers and unmistakable in one that spans decades; a period T is invisible
until the table contains (x, x+T) pairs.

Black-box calls are free, so the plan is generous. Only the LLM call costs.
"""

from __future__ import annotations

import math

from reconnaissance import Oracle, Profile


def _legal(o: Oracle, pt):
    args = pt if isinstance(pt, tuple) else (pt,)
    return o.val(*args) is not None


def plan_1d(o: Oracle, p: Profile):
    """Returns (points, rationale)."""
    pts, why = [], []

    def add(cands, reason):
        kept = [c for c in cands if _legal(o, c) and c not in pts]
        if kept:
            pts.extend(kept)
            why.append((reason, list(kept)))

    if p.integer_only:
        add([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
            "integer-only domain: sample the integers, no fractional points")
        return pts, why

    add([0], "constant term")
    add([1, -1, 2, -2, 3, -3],
        "symmetric pairs so parity is a fact in the table, not a claim")
    add([0.5, -0.5, 1.5],
        "fractional inputs: shows the domain is not integer-only")

    # magnitude coverage, stopped short of where the box overflows
    top = 12 if p.overflow_decade is None else max(1, p.overflow_decade - 2)
    decades = [10.0 ** k for k in range(1, min(top, 12) + 1, 2)]
    if p.poly_degree in (1, 2, 3) and not p.poly_is_none:
        decades = decades[:2]                  # a low-degree polynomial needs no sweep
        add(decades, "modest magnitude check (degree already established)")
    else:
        add(decades,
            "geometric spread: growth order is only legible across decades, "
            "never across neighbouring integers")
        add([-(10.0 ** k) for k in range(1, min(top, 8) + 1, 3)],
            "same on the negative side")

    if p.period:
        T = p.period
        pairs = []
        for x in (0.3, 1.1, 2.7):
            pairs += [round(x, 4), round(x + T, 4), round(x + 2 * T, 4)]
        add(pairs,
            f"triples spaced by the measured period {T:.6f}, so the repetition "
            f"appears as equal differences in the table")

    if p.inflections:
        near = []
        for r in p.inflections[:2]:
            near += [round(r * 0.5, 4), round(r, 4), round(r * 1.5, 4)]
        add(near, "points bracketing each inflection, where curvature turns over")

    if isinstance(p.poly_degree, int):
        add([7, -7, 11], f"extra points beyond the {p.poly_degree + 1} a "
                         f"degree-{p.poly_degree} polynomial needs, as a cross-check")

    return pts, why


def plan_2d(o: Oracle, p: Profile):
    pts, why = [], []

    def add(cands, reason):
        kept = [c for c in cands if _legal(o, c) and c not in pts]
        if kept:
            pts.extend(kept)
            why.append((reason, list(kept)))

    add([(0, 0)], "baseline")
    add([(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)],
        "axis sweeps: each isolates one variable's contribution")
    add([(1, 1), (2, 2), (4, 4)],
        "diagonal triple: the ratios between these give the homogeneity degree")
    add([(2, 3), (3, 2)],
        "swapped pair, so symmetry (or its absence) is visible directly")
    add([(1, 2), (2, 1), (3, -2), (-2, 3)],
        "mixed signs and unequal magnitudes")
    if p.separable is False:
        add([(2, 5), (5, 2), (4, 3), (3, 4)],
            "extra off-diagonal points: an interaction term was detected, so the "
            "cross structure needs more coverage")
    top = 4 if p.overflow_decade is None else max(1, min(4, p.overflow_decade - 2))
    add([(10.0 ** k, 10.0 ** k) for k in range(1, top + 1)] +
        [(10.0 ** top, 1.0), (1.0, 10.0 ** top)],
        "magnitude coverage, jointly and one variable at a time")
    return pts, why


def build_observations(o: Oracle, p: Profile):
    """Probe the planned points. Returns (observations, rationale).

    An input that raises is kept as an observation too: 'undefined here' is
    information about the domain, and the LLM should see it. Only the exception
    TYPE is retained -- never its message, which on a TypeError would contain
    the function's own name."""
    pts, why = (plan_1d(o, p) if o.n_args == 1 else plan_2d(o, p))
    obs = []
    for pt in pts:
        args = pt if isinstance(pt, tuple) else (pt,)
        v, err = o(*args)
        obs.append((args, v, err))
    obs.sort(key=lambda r: sum(abs(a) for a in r[0]))
    return obs, why


def format_observations(obs):
    """Full float precision is deliberate: 0.6931471805599453 is recognisably
    ln 2 and 2.302585092994046 is recognisably ln 10, while 0.693 and 2.303 are
    just decimals. The extra tokens cost about a hundredth of a cent."""
    lines = []
    for args, v, err in obs:
        sig = ", ".join(f"{a:g}" for a in args)
        if v is None:
            lines.append(f"  f({sig}) = <undefined: raised {err}>")
        else:
            lines.append(f"  f({sig}) = {v!r}")
    return "\n".join(lines)
