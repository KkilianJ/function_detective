"""
The evidence block: what the free structural battery measured, stated as text.

WHAT MAY GO IN, AND WHY

Three kinds of statement can reach the model safely:

  * a MEASUREMENT       "f(2.9) and f(-2.9) agree to 16 significant figures"
  * an EXCLUSION        "no finite-difference order up to 6 converged"
  * a SCOPE             "over x in [2.2, 9.7e3]"

One kind may not: a positive guess about what the function IS. A wrong
measurement is impossible; a wrong exclusion merely rules out too little; but a
wrong suggestion steers every later round with nothing to retract it.

THREE RULES THAT MAKE THE DIFFERENCE

1. STATE THE MEASUREMENT, NOT THE ADJECTIVE. "approximately even" is a vague
   claim the model can only use vaguely. "agrees to 16 significant figures at
   five symmetric pairs" is a hard constraint, and the reader can judge for
   themselves how much to trust it.

2. EVERY EXCLUSION CARRIES ITS SCOPE. "no clear fit" may be a fact about the
   function or an artefact of too narrow a sampling window, and without the
   range there is no way to tell. An unscoped exclusion can push the model the
   wrong way with no trace back to why.

3. AN ABSTENTION IS NOT EVIDENCE. A test that ran and failed is information; a
   test that could not run is not. "growth: unknown" is dropped -- it carries
   nothing and makes the problem look harder than it is.

The ranges below are the ones the reconnaissance layers actually probe, so the
scopes are exact rather than approximate.
"""

from __future__ import annotations

import math

GROWTH_RANGE = "x in [2.2, 9.7e3]"          # 1.3**3 .. 1.3**33
PERIOD_RANGE = "x in [0, 60]"
CURV_RANGE = "x in [0.05, 8]"
ODE_POINTS = "x = 50, 200, 800, 3200"
HOMOG_POINTS = "x = 2, 3, 5, 7 with k = 2 and 3"
POLY_POINTS = "x = 1, 3, 5, ... 17"
POLY_FAR = "x = 81"


def _agreement_digits(a, b):
    """How many significant figures two values share -- the honest way to say
    'equal' about floating point."""
    if a is None or b is None:
        return 0
    if a == b:
        return 17
    scale = max(abs(a), abs(b), 1e-300)
    rel = abs(a - b) / scale
    return 0 if rel >= 1 else min(17, int(-math.log10(rel)))


def build_evidence(oracle, profile) -> str:
    """Returns the NUMERICAL EVIDENCE block, or '' when nothing was measured."""
    n = profile.arity
    lines = []

    # ---- domain ----
    if profile.errors:
        detail = ", ".join(f"{k} at {v} of the probed inputs"
                           for k, v in profile.errors.items())
        lines.append(f"- Domain: some probes raised -- {detail}. All other probed "
                     f"inputs returned a finite value.")
    else:
        lines.append("- Domain: every probed input returned a finite value, "
                     "including 0, negatives, fractions, 1e-6 and 1e6.")
    if profile.integer_only:
        lines.append("- f(0.5) raises while f(2) does not: the function accepts "
                     "integers only.")

    # ---- complex ----
    if profile.accepts_complex is True:
        lines.append("- f(1j) evaluates without error, so the implementation is "
                     "pure arithmetic: no math-module function is involved.")
    elif profile.accepts_complex is False:
        lines.append("- f(1j) raises TypeError, so the implementation calls at "
                     "least one math-module function (log, sqrt, sin, ...).")

    # ---- parity ----
    if n == 1 and profile.parity in ("even", "odd"):
        digits, pts = [], []
        for x in (0.7, 1.3, 2.9, 7.1, 13.3):
            a, b = oracle.val(x), oracle.val(-x)
            if a is None or b is None:
                continue
            digits.append(_agreement_digits(a, -b if profile.parity == "odd" else b))
            pts.append(x)
        if digits:
            rel = "f(-x) == f(x)" if profile.parity == "even" else "f(-x) == -f(x)"
            lines.append(f"- Symmetry: {rel} to at least {min(digits)} significant "
                         f"figures at x = {', '.join(f'{p:g}' for p in pts)}.")
    elif n == 1 and profile.parity == "neither":
        lines.append("- Symmetry: f(-x) equals neither f(x) nor -f(x) at the "
                     "probed symmetric pairs; the function is neither even nor odd.")

    # ---- polynomial ----
    if isinstance(profile.poly_degree, int):
        lines.append(f"- Polynomial degree: finite differences over {POLY_POINTS} "
                     f"become constant at order {profile.poly_degree}, and the "
                     f"degree-{profile.poly_degree} interpolant through those points "
                     f"reproduces f({POLY_FAR}) exactly. The function IS a polynomial "
                     f"of degree {profile.poly_degree}.")
    elif profile.poly_is_none:
        lines.append(f"- Polynomial EXCLUDED: no finite-difference order up to 6 "
                     f"became constant over {POLY_POINTS}, or the interpolant that "
                     f"did failed to reproduce f({POLY_FAR}). Note that every smooth "
                     f"function looks polynomial locally, so the extrapolation test "
                     f"is what settles it.")

    # ---- curvature and inflections ----
    if profile.second_derivative == "zero":
        lines.append(f"- Second derivative is identically zero over {CURV_RANGE}.")
    elif profile.second_derivative == "constant":
        lines.append(f"- Second derivative is constant at "
                     f"{profile.second_derivative_value:.10g} over {CURV_RANGE}.")
    elif profile.second_derivative in ("convex", "concave"):
        lines.append(f"- Second derivative keeps one sign over {CURV_RANGE} "
                     f"({profile.second_derivative}); there is no inflection point "
                     f"in that range.")
    elif profile.second_derivative == "sign-changing" and profile.inflections:
        pts = ", ".join(f"{r:.6f}" for r in profile.inflections)
        lines.append(f"- Second derivative changes sign over {CURV_RANGE}. "
                     f"f''(x) = 0 at x = {pts}, located by bisection to 1e-9.")

    # ---- growth order ----
    if profile.growth:
        names = {"linear": "f vs x", "log": "f vs log x",
                 "power": "log f vs log x", "exp": "log f vs x"}
        lines.append(f"- Growth: over {GROWTH_RANGE}, {names[profile.growth]} is "
                     f"straight with R2 = {profile.growth_r2:.5f}, clear of the "
                     f"runner-up by more than 0.01.")
    else:
        lines.append(f"- Growth EXCLUDED: over {GROWTH_RANGE}, none of f vs x, "
                     f"f vs log x, log f vs log x or log f vs x reached R2 > 0.999 "
                     f"with a margin over the runner-up.")

    # ---- ODE ratios ----
    if profile.ode_family:
        what = {"exponential": "f'/f", "power": "x*f'/f", "logarithmic": "x*f'"}
        lines.append(f"- Differential ratio: at {ODE_POINTS}, "
                     f"{what[profile.ode_family]} is constant at "
                     f"{profile.ode_parameter:.6g} (relative spread below 0.2%).")
    elif n == 1 and not profile.integer_only:
        lines.append(f"- Differential ratios EXCLUDED: at {ODE_POINTS}, none of "
                     f"f'/f, x*f'/f or x*f' settled to a constant.")

    # ---- homogeneity ----
    if profile.homogeneity is not None:
        lines.append(f"- Homogeneity: at {HOMOG_POINTS}, f(k*x) / f(x) gives a "
                     f"consistent degree of {profile.homogeneity:.4g}.")
    else:
        lines.append(f"- Homogeneity EXCLUDED: at {HOMOG_POINTS}, f(k*x) / f(x) "
                     f"gave no consistent degree.")

    # ---- periodicity ----
    if profile.period:
        lines.append(f"- Periodicity: over {PERIOD_RANGE}, f(x + T) - f(x) is "
                     f"constant for T = {profile.period:.6f} (spread below 2% of "
                     f"the detrended amplitude). A constant, not zero, difference "
                     f"means a linear trend may be superimposed.")
    elif profile.period_tested:
        lines.append(f"- Periodicity EXCLUDED: over {PERIOD_RANGE}, no candidate "
                     f"period from the FFT spectrum produced a constant "
                     f"f(x + T) - f(x) after refinement.")

    # ---- overflow ----
    if profile.overflow_decade:
        lines.append(f"- Magnitude: the result stops being representable at "
                     f"x = 1e{profile.overflow_decade}. Where it breaks bounds the "
                     f"growth order from above.")

    # ---- two-variable ----
    if n == 2:
        if profile.symmetric is True:
            lines.append("- Symmetry in the arguments: f(x, y) == f(y, x) at all "
                         "probed swapped pairs.")
        elif profile.symmetric is False:
            lines.append("- Symmetry in the arguments EXCLUDED: f(x, y) != f(y, x) "
                         "at the probed swapped pairs, so x and y are NOT "
                         "interchangeable.")
        if profile.separable is True:
            lines.append("- No interaction term: the mixed second difference "
                         "f(x,y) - f(x,0) - f(0,y) + f(0,0) is zero at every probed "
                         "point, so f is a sum of a function of x and a function of y.")
        elif profile.separable is False:
            lines.append("- Interaction term PRESENT: the mixed second difference "
                         "f(x,y) - f(x,0) - f(0,y) + f(0,0) is non-zero, so f does "
                         "not split into a function of x plus a function of y.")
        for axis, d in profile.slices.items():
            bits = []
            if isinstance(d.get("degree"), int):
                bits.append(f"is a degree-{d['degree']} polynomial in that variable")
            elif d.get("degree") == "not polynomial":
                bits.append("is not polynomial in that variable")
            if d.get("ode"):
                bits.append(f"differential ratio gives {d['ode']}")
            if d.get("parity") in ("even", "odd"):
                bits.append(f"is {d['parity']} in that variable")
            if bits:
                other = "y" if axis == "x" else "x"
                lines.append(f"- Slice with {other} fixed at 2: f {'; '.join(bits)}.")

    if not lines:
        return ""
    return ("NUMERICAL EVIDENCE (measured on the black box; no guesses, and every "
            "exclusion states the range it was tested over):\n\n" + "\n".join(lines))
