"""
Stage 0 -- Find useful information for the LLM.

Before asking the LLM to guess the function, the agent first explores the
black box using different inputs and outputs.

The purpose of this stage is to find useful information that can help the LLM
form better hypotheses. For example, the agent may look for patterns such as
symmetry, growth, polynomial behaviour, or interactions between variables.

Each test can return CONFIRMED, DENIED, or UNKNOWN. If there is not enough
evidence, the agent keeps the result as UNKNOWN instead of making an uncertain
claim.

The discovered information is stored in a Profile. This profile is then used
to decide what information and which input/output examples should be provided
to the LLM.

In short:

    black box
        -> explore different inputs and outputs
        -> find useful patterns
        -> store them in a Profile
        -> use them to prepare better information for the LLM
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import numpy as np

UNKNOWN = None          # explicit abstention


# safe calling

class Oracle:
    """Wraps the black box: counts calls, converts every failure into data."""

    def __init__(self, fn, n_args):
        self.fn = fn
        self.n_args = n_args
        self.calls = 0
        self.errors: dict[str, int] = {}

    def __call__(self, *args):
        """Returns (value, error_name). Exactly one of them is None."""
        self.calls += 1
        try:
            v = self.fn(*args)
        except Exception as exc:                      # a failure is information
            name = type(exc).__name__
            self.errors[name] = self.errors.get(name, 0) + 1
            return None, name
        if isinstance(v, complex):
            return v, None
        if not isinstance(v, (int, float)):
            return None, "NonNumeric"
        try:
            v = float(v)                     # a huge int (factorial) can fail here
        except OverflowError:
            self.errors["OverflowError"] = self.errors.get("OverflowError", 0) + 1
            return None, "OverflowError"
        if not math.isfinite(v):
            return None, "NonFinite"
        return v, None

    def val(self, *args):
        """Value only, or None."""
        return self(*args)[0]


def discover_arity(fn, max_args=3):
    """Ask the black box how many arguments it takes, by trial."""
    for n in range(1, max_args + 1):
        try:
            fn(*([1.0] * n))
        except TypeError:
            continue
        except Exception:
            return n                                   # it ran, then failed on 1.0
        else:
            return n
    return UNKNOWN


# profile

@dataclass
class Profile:
    """Independently established facts. Any field may be UNKNOWN."""
    arity: int | None = None
    legal_points: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    domain_note: str = ""
    constant: bool | None = None
    integer_only: bool | None = None
    accepts_complex: bool | None = None
    overflow_decade: int | None = None
    parity: str | None = None            # "even" | "odd" | "neither"
    poly_degree: int | None = None       # int, or "not polynomial" via poly_is_none
    poly_is_none: bool = False
    homogeneity: float | None = None
    growth: str | None = None            # "linear"|"log"|"power"|"exp"
    growth_r2: float | None = None
    second_derivative: str | None = None  # "zero"|"constant"|"convex"|"concave"|"sign-changing"
    second_derivative_value: float | None = None
    inflections: list = field(default_factory=list)
    ode_family: str | None = None        # "exponential"|"power"|"logarithmic"
    ode_parameter: float | None = None
    period: float | None = None
    period_tested: bool = False
    symmetric: bool | None = None        # 2-var
    separable: bool | None = None        # 2-var: no cross term
    slices: dict = field(default_factory=dict)   # 2-var: per-axis 1-var reading
    conflicts: list = field(default_factory=list)
    oracle_calls: int = 0

    # -------- rendering --------
    def rows(self):
        def s(v, fmt="{}"):
            return "?" if v is None else fmt.format(v)
        out = [
            ("arity", s(self.arity)),
            ("domain", self.domain_note or "?"),
            ("constant", s(self.constant)),
            ("integer only", s(self.integer_only)),
            ("accepts complex", s(self.accepts_complex)),
            ("overflow at", "none" if self.overflow_decade is None and self.arity else
                            (f"1e{self.overflow_decade}" if self.overflow_decade else "?")),
            ("parity", s(self.parity)),
            ("polynomial degree",
             "not polynomial" if self.poly_is_none else s(self.poly_degree)),
            ("homogeneity", s(self.homogeneity, "{:.4g}")),
            ("growth", s(self.growth) + (f" (R2={self.growth_r2:.5f})"
                                         if self.growth_r2 else "")),
            ("f''", s(self.second_derivative)
                    + (f" = {self.second_derivative_value:.6g}"
                       if self.second_derivative_value is not None else "")),
            ("inflections", ", ".join(f"{r:.6f}" for r in self.inflections)
                            if self.inflections else ("none" if self.second_derivative else "?")),
            ("ODE family", (f"{self.ode_family} (param {self.ode_parameter:.4f})"
                            if self.ode_family else ("none" if self.period_tested else "?"))),
            ("period", (f"{self.period:.6f}" if self.period
                        else ("none" if self.period_tested else "?"))),
        ]
        if self.arity == 2:
            out += [("symmetric", s(self.symmetric)),
                    ("separable", s(self.separable))]
            for ax, d in self.slices.items():
                out.append((f"slice f(.,{ax}) fixed",
                            "  ".join(f"{k}={v}" for k, v in d.items() if v is not None)))
        out.append(("black-box calls", str(self.oracle_calls)))
        if self.conflicts:
            out.append(("CONFLICTS", "; ".join(self.conflicts)))
        return out

    def render(self, title=""):
        head = f"--- profile {title} ---" if title else "--- profile ---"
        lines = [head]
        for k, v in self.rows():
            lines.append(f"  {k:<20}: {v}")
        return "\n".join(lines)


# layer 1: the ground

DOMAIN_PROBES_1D = [
    0.0,
    1e-6, -1e-6,
    1e-3, -1e-3,
    0.5, -0.5,
    1.0, -1.0,
    2.0, -2.0,
    10.0, -10.0,
    100.0, -100.0,
    1e3, -1e3,
    1e6, -1e6,
]
DOMAIN_PROBES_2D = [
    # origin / axes
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),

    # same-sign / opposite-sign
    (1.0, 1.0),
    (-1.0, -1.0),
    (1.0, -1.0),
    (-1.0, 1.0),

    # asymmetric inputs
    (2.0, 3.0),
    (3.0, 2.0),
    (-2.0, 3.0),
    (2.0, -3.0),

    # fractional scale
    (0.5, 0.5),
    (0.5, 2.0),
    (2.0, 0.5),

    # medium / large scales
    (10.0, 10.0),
    (100.0, 100.0),
    (1e3, 1e3),

    # different scales between variables
    (1.0, 100.0),
    (100.0, 1.0),

    # very small / very large
    (1e-6, 1e-6),
    (1e6, 1e6),
]


def layer_domain(o: Oracle, p: Profile):
    """Map where the function is defined. Everything downstream probes only here."""
    probes = DOMAIN_PROBES_1D if o.n_args == 1 else DOMAIN_PROBES_2D
    legal, seen_errors = [], {}
    for pt in probes:
        args = pt if isinstance(pt, tuple) else (pt,)
        v, e = o(*args)
        if v is None:
            seen_errors.setdefault(e, []).append(pt)
        else:
            legal.append(pt)
    p.legal_points = legal
    p.errors = {k: len(v) for k, v in seen_errors.items()}
    if not legal:
        p.domain_note = "no legal probe found"
        return False
    p.domain_note = ("all probes legal" if not seen_errors
                     else "restricted: " + ", ".join(f"{k}x{len(v)}"
                                                     for k, v in seen_errors.items()))
    # constant?
    vals = [o.val(*(pt if isinstance(pt, tuple) else (pt,))) for pt in legal]
    vals = [v for v in vals if v is not None]
    p.constant = bool(vals) and max(vals) - min(vals) < 1e-12 * max(1.0, abs(vals[0]))
    # integer-only?
    if o.n_args == 1:
        half, _ = o(0.5)
        two, _ = o(2.0)
        p.integer_only = (half is None and two is not None)
    else:
        half, _ = o(0.5, 0.5)
        two, _ = o(2.0, 2.0)
        p.integer_only = (half is None and two is not None)
    return True


# ------------------------------------------ layer 2: one probe, huge payoff

def layer_complex(o: Oracle, p: Profile):
    """Pure arithmetic accepts complex input; every math.* function rejects it.

    One call splits the whole hypothesis space in two."""
    args = (1j,) if o.n_args == 1 else (1j, 1.0)
    try:
        r = o.fn(*args)
    except TypeError:
        p.accepts_complex = False        # math.* present
        return
    except Exception:
        p.accepts_complex = UNKNOWN      # failed for some other reason
        return
    p.accepts_complex = isinstance(r, (int, float, complex))


def layer_overflow(o: Oracle, p: Profile):
    """Climb decades until the result stops being representable.

    WHERE it breaks is the growth order: exp dies near 1e3, x! near 1e2,
    x**2 only past 1e154, a logarithm never."""
    for k in range(1, 301):
        try:
            x = 10.0 ** k
        except OverflowError:
            break
        args = (x,) if o.n_args == 1 else (x, x)
        v, e = o(*args)
        if v is None:
            if e in ("OverflowError", "NonFinite"):
                p.overflow_decade = k
            else:
                p.overflow_decade = UNKNOWN     # domain error, not growth
            return
    p.overflow_decade = UNKNOWN


#  layer 3: structure

def layer_parity(o: Oracle, p: Profile):
    if o.n_args != 1:
        return
    pairs = [x for x in (0.7, 1.3, 2.9, 7.1, 13.3) if o.val(x) is not None
             and o.val(-x) is not None]
    if len(pairs) < 3:
        p.parity = UNKNOWN
        return
    even = all(abs(o.val(x) - o.val(-x)) <= 1e-9 * max(1.0, abs(o.val(x))) for x in pairs)
    odd = all(abs(o.val(x) + o.val(-x)) <= 1e-9 * max(1.0, abs(o.val(x))) for x in pairs)
    p.parity = "even" if even else ("odd" if odd else "neither")


def layer_polynomial(o: Oracle, p: Profile, base=1.0, h=2.0, maxdeg=6):
    """Finite differences PROPOSE a degree; extrapolation CONFIRMS it.

    Every smooth function is locally polynomial (Taylor), so a converging
    difference table on its own proves nothing. The proposed degree is used to
    interpolate through deg+1 points, then asked to predict a point far outside
    that range. exp collapses immediately; a real polynomial is exact."""
    if o.n_args != 1:
        return
    vals = []
    for i in range(maxdeg + 3):
        v = o.val(base + i * h)
        if v is None:
            p.poly_degree = UNKNOWN
            return
        vals.append(v)

    d, deg = list(vals), None
    scale = max(1.0, max(abs(v) for v in vals))
    for n in range(maxdeg + 1):
        if len(d) < 2:
            break
        if max(d) - min(d) < 1e-9 * scale:
            deg = n
            break
        d = [d[i + 1] - d[i] for i in range(len(d) - 1)]
    if deg is None:
        p.poly_is_none = True
        return

    xs = [base + i * h for i in range(deg + 1)]
    ys = vals[: deg + 1]
    if deg == 0:
        pred = lambda t: ys[0]
    else:
        c = np.polyfit(xs, ys, deg)
        pred = lambda t: float(np.polyval(c, t))

    checked = False
    for far in (base + 40 * h, base - 30 * h, base + 200 * h):
        truth = o.val(far)
        if truth is None:
            continue
        checked = True
        if abs(pred(far) - truth) > 1e-6 * max(1.0, abs(truth)):
            p.poly_is_none = True          # only locally polynomial
            return
    p.poly_degree = deg if checked else UNKNOWN


def layer_homogeneity(o: Oracle, p: Profile):
    """f(kx) = k^d f(x) -- one number that pins down sqrt/norm/monomial forms."""
    degs = []
    for base in (2.0, 3.0, 5.0, 7.0):
        for k in (2.0, 3.0):
            if o.n_args == 1:
                a, b = o.val(base), o.val(k * base)
            else:
                a, b = o.val(base, base), o.val(k * base, k * base)
            if a is None or b is None or a <= 0 or b <= 0:
                p.homogeneity = UNKNOWN
                return
            degs.append(math.log(b / a) / math.log(k))
    p.homogeneity = (round(float(np.mean(degs)), 4)
                     if float(np.std(degs)) < 1e-6 else UNKNOWN)


def _r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return -1.0
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            k, m = np.polyfit(a, b, 1)
            ss = float(np.sum((b - b.mean()) ** 2))
            res = float(np.sum((b - (k * a + m)) ** 2))
        except Exception:
            return -1.0
    if not (math.isfinite(ss) and math.isfinite(res)) or ss <= 0:
        return -1.0
    return 1.0 - res / ss


def layer_growth(o: Oracle, p: Profile):
    """Four linearisations. Whichever straightens the curve names the family.

    Requires a clear winner: R2 > 0.999 AND a margin over the runner-up.
    Otherwise abstain -- a near-tie means the data cannot tell them apart."""
    xs = [1.3 ** i for i in range(3, 34)]
    pts = []
    for x in xs:
        v = o.val(x) if o.n_args == 1 else o.val(x, x)
        if v is not None and v > 0:
            pts.append((x, v))
    if len(pts) < 10:
        p.growth = UNKNOWN
        return
    X = [q[0] for q in pts]
    Y = [q[1] for q in pts]
    lx = [math.log(x) for x in X]
    ly = [math.log(y) for y in Y]
    cand = {"linear": _r2(X, Y), "log": _r2(lx, Y),
            "power": _r2(lx, ly), "exp": _r2(X, ly)}
    best = max(cand, key=cand.get)
    ordered = sorted(cand.values(), reverse=True)
    if ordered[0] > 0.999 and ordered[0] - ordered[1] > 0.01:
        p.growth, p.growth_r2 = best, ordered[0]
    else:
        p.growth = UNKNOWN


# ------------------------------------------------- layer 4: the derivative

def _d1(o: Oracle, x):
    h = abs(x) * 1e-6 + 1e-8
    a, b = o.val(x + h), o.val(x - h)
    return None if a is None or b is None else (a - b) / (2 * h)


def _d2(o: Oracle, x):
    h = max(abs(x) * 1e-4, 1e-4)
    a, b, c = o.val(x + h), o.val(x), o.val(x - h)
    return None if None in (a, b, c) else (a - 2 * b + c) / (h * h)


def layer_curvature(o: Oracle, p: Profile, lo=0.05, hi=8.0, n=240):
    """f'' classifies shape; where it changes sign is an inflection, and an
    inflection's LOCATION often encodes a parameter directly.

    log(x**2 + a) inflects at x = sqrt(a): finding the inflection reads off a."""
    if o.n_args != 1 or p.integer_only:
        return
    xs = list(np.linspace(lo, hi, n))
    vals = [(x, _d2(o, x)) for x in xs]
    vals = [(x, v) for x, v in vals if v is not None]
    if len(vals) < n // 2:
        return
    a = np.array([v for _, v in vals])
    scale = max(float(np.abs(a).max()), 1e-30)

    if float(np.abs(a).max()) < 1e-6:
        p.second_derivative = "zero"
        return
    if float(a.std()) / scale < 1e-6:
        p.second_derivative = "constant"
        p.second_derivative_value = float(a.mean())
        return

    signs = np.sign(a)
    flips = [i for i in range(1, len(signs)) if signs[i] and signs[i] != signs[i - 1]]
    if not flips:
        p.second_derivative = "convex" if a.mean() > 0 else "concave"
        return

    p.second_derivative = "sign-changing"
    roots = []
    for i in flips:
        left, right = vals[i - 1][0], vals[i][0]
        fl = _d2(o, left)
        for _ in range(60):
            mid = (left + right) / 2
            fm = _d2(o, mid)
            if fm is None:
                break
            if fl * fm <= 0:
                right = mid
            else:
                left, fl = mid, fm
        roots.append((left + right) / 2)
    p.inflections = [round(float(r), 6) for r in roots]


def layer_ode(o: Oracle, p: Profile, xs=(50.0, 200.0, 800.0, 3200.0)):
    """Which ratio settles to a constant names the family AND its parameter:
         f'/f    -> exponential, constant is the rate
         x f'/f  -> power law,   constant is the exponent
         x f'    -> logarithm,   constant is the coefficient
    These are asymptotic, so low-order terms slow convergence; abstain rather
    than force a reading."""
    if o.n_args != 1 or p.integer_only:
        return
    fpf, xfpf, xfp = [], [], []
    for x in xs:
        fx = o.val(x)
        fp = _d1(o, x)
        if fx is None or fp is None or fx == 0:
            p.ode_family = UNKNOWN
            return
        if abs(fp) < 1e-12 * max(1.0, abs(fx)):
            p.ode_family = UNKNOWN          # flat: every ratio is trivially 0
            return
        fpf.append(fp / fx)
        xfpf.append(x * fp / fx)
        xfp.append(x * fp)

    def steady(v, tol=2e-3):
        arr = np.array(v)
        return float(np.std(arr)) / max(abs(float(np.mean(arr))), 1e-12) < tol

    for name, series in (("exponential", fpf), ("power", xfpf), ("logarithmic", xfp)):
        if steady(series):
            p.ode_family = name
            p.ode_parameter = float(np.mean(series))
            return
    p.ode_family = UNKNOWN


#  layer 5: periodicity

def layer_period(o: Oracle, p: Profile, lo=0.0, hi=60.0, n=1536):
    """FFT only PROPOSES a period; the proposal is confirmed by requiring
    f(x+T) - f(x) to be constant across the window (constant, not zero, so a
    superimposed linear trend is tolerated).

    Two failure modes this guards against, both seen while building it:
      * leftover trend puts the FFT peak at the lowest bin, giving a bogus
        "period" equal to the window -- rejected by demanding T << window;
      * FFT bin resolution is too coarse to hit the true period, so the
        constancy test fails on a genuinely periodic function -- fixed by
        refining T around the proposal before testing."""
    if o.n_args != 1 or p.integer_only:
        return
    xs = np.linspace(lo, hi, n)
    ys = [o.val(float(x)) for x in xs]
    if any(v is None for v in ys):
        return
    p.period_tested = True
    ys = np.asarray(ys, float)
    k, b = np.polyfit(xs, ys, 1)
    resid = ys - (k * xs + b)
    amp = float(resid.std())
    if amp < 1e-12:
        p.period = None
        return

    w = (resid - resid.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(w))[1:]
    freqs = np.fft.rfftfreq(n, d=(hi - lo) / (n - 1))[1:]
    span = hi - lo

    def spread(t, m=32):
        """Std of f(x+t) - f(x). For a periodic function (plus any linear
        trend) this is zero at the true period."""
        probe = np.linspace(lo, hi - t, m)
        diffs = []
        for x in probe:
            a1, a2 = o.val(float(x + t)), o.val(float(x))
            if a1 is None or a2 is None:
                return float("inf")
            diffs.append(a1 - a2)
        return float(np.std(diffs))

    for idx in np.argsort(spec)[::-1][:3]:
        if freqs[idx] <= 0:
            continue
        t0 = 1.0 / freqs[idx]
        if t0 > span / 4 or t0 < 1e-2:
            continue
        # two-stage refinement: coarse sweep, then a tight sweep around the best
        best_t, best_s = t0, spread(t0)
        for width, steps in ((0.15, 24), (0.01, 24)):
            grid = np.linspace(best_t * (1 - width), best_t * (1 + width), steps)
            for t in grid:
                s_ = spread(float(t))
                if s_ < best_s:
                    best_s, best_t = s_, float(t)
        if best_s < 0.02 * amp:
            p.period = round(best_t, 6)
            return
    p.period = None


# --------------------------------------------------- two-variable structure

def layer_two_var(o: Oracle, p: Profile):
    if o.n_args != 2:
        return
    pairs = [(2.0, 3.0), (1.0, 4.0), (-2.0, 5.0), (0.5, 2.5), (7.0, 1.0)]
    usable = [(x, y) for x, y in pairs
              if o.val(x, y) is not None and o.val(y, x) is not None]
    if len(usable) >= 3:
        p.symmetric = all(
            abs(o.val(x, y) - o.val(y, x)) <= 1e-9 * max(1.0, abs(o.val(x, y)))
            for x, y in usable)

    # mixed second difference: zero <=> no interaction term
    base = o.val(0.0, 0.0)
    if base is None:
        return
    mixed = []
    for x, y in ((1.0, 1.0), (2.0, 3.0), (0.5, 4.0), (-1.0, 2.0)):
        fxy, fx0, f0y = o.val(x, y), o.val(x, 0.0), o.val(0.0, y)
        if None in (fxy, fx0, f0y):
            return
        mixed.append(fxy - fx0 - f0y + base)
    scale = max(1.0, max(abs(o.val(x, y)) for x, y in ((1.0, 1.0), (2.0, 3.0))))
    p.separable = all(abs(m) <= 1e-9 * scale for m in mixed)


def layer_slices(o: Oracle, p: Profile):
    """Freeze one variable, run the single-variable battery on the slice.

    A two-variable problem collapses into two one-variable problems this way,
    which is the only route to a degree or growth reading for arity 2."""
    if o.n_args != 2:
        return
    for axis, y0 in (("x", 2.0), ("y", 2.0)):
        fn = ((lambda t, y0=y0: o.fn(t, y0)) if axis == "x"
              else (lambda t, y0=y0: o.fn(y0, t)))
        sub = Oracle(fn, 1)
        sp = Profile(arity=1)
        try:
            if not layer_domain(sub, sp):
                continue
            layer_polynomial(sub, sp)
            layer_growth(sub, sp)
            layer_ode(sub, sp)
            layer_parity(sub, sp)
        except Exception:
            continue
        p.slices[axis] = {
            "degree": ("not polynomial" if sp.poly_is_none else sp.poly_degree),
            "growth": sp.growth,
            "ode": (f"{sp.ode_family}({sp.ode_parameter:.4f})"
                    if sp.ode_family else None),
            "parity": sp.parity,
        }
        p.oracle_calls += sub.calls


# ------------------------------------------------------- conflict detection

def detect_conflicts(p: Profile):
    """Disagreement between independent tests is itself a signal -- usually a
    composite structure. Recording it beats silently trusting one of them."""
    c = []
    if isinstance(p.poly_degree, int) and p.ode_family == "power" and p.ode_parameter:
        if abs(p.ode_parameter - p.poly_degree) > 0.05:
            c.append(f"finite differences say degree {p.poly_degree}, "
                     f"ODE ratio says exponent {p.ode_parameter:.3f}")
    if p.poly_is_none and p.second_derivative == "constant":
        c.append("f'' is constant (implies quadratic) but extrapolation "
                 "rejected the polynomial hypothesis")
    if p.growth == "log" and isinstance(p.poly_degree, int) and p.poly_degree > 0:
        c.append(f"growth reads logarithmic yet degree {p.poly_degree} was confirmed")
    if p.period and p.growth == "exp":
        c.append("periodic and exponential were both accepted")
    p.conflicts = c


#  entry

def investigate(fn, n_args=None) -> Profile:
    """Run the full battery. Never raises: an unusable function yields a
    profile that is mostly UNKNOWN, which is a legitimate answer."""
    if n_args is None:
        n_args = discover_arity(fn) or 1
    o = Oracle(fn, n_args)
    p = Profile(arity=n_args)

    if not layer_domain(o, p):
        p.oracle_calls = o.calls
        return p
    if p.constant:
        p.oracle_calls = o.calls
        return p

    for step in (layer_complex, layer_overflow, layer_parity, layer_polynomial,
                 layer_homogeneity, layer_growth, layer_curvature, layer_ode,
                 layer_period, layer_two_var, layer_slices):
        try:
            step(o, p)
        except Exception:
            pass                    # a failed test abstains; it never aborts the run

    detect_conflicts(p)
    p.oracle_calls = o.calls
    return p
