"""
Visual fallback.

This module provides an additional source of evidence for difficult functions.

If a function cannot be solved within its normal LLM-call budget, the agent
can pause that case and continue solving the other functions. If budget is
still available later, the agent can return to the difficult case using the
visual approach.

The agent generates plots from different black-box input/output samples to
show the overall behaviour of the function. The plots are then sent to the
LLM together with the existing textual and numerical evidence.

The image does not replace the numerical observations. It provides another
representation of the same black-box behaviour that may reveal patterns which
were difficult to recognize from numbers alone.

In short:

    difficult function
        -> normal budget exhausted
        -> pause and solve other functions
        -> return if budget remains
        -> generate plots
        -> send text + numerical evidence + image to the LLM
"""
from __future__ import annotations

import base64
import io
import math

import matplotlib
matplotlib.use("Agg")                       # no display in this environment
import matplotlib.pyplot as plt
import numpy as np

FAMILY_UNRESOLVED_BELOW = 0.60              # loose agreement under this = family wrong


# ------------------------------------------------------------------- trigger

def family_established(profile):
    """Did ANY test name the family? Returns the reason, or None.

    The right question is not "did some particular test abstain" -- several
    always do -- but "is the family known by any route at all". Asking it the
    wrong way fires the visual channel on a confirmed linear function, which is
    both wasteful and indefensible."""
    if profile.constant:
        return "constant function"
    if isinstance(profile.poly_degree, int):
        return f"polynomial of degree {profile.poly_degree}, confirmed by extrapolation"
    if profile.second_derivative in ("zero", "constant"):
        return f"f'' is {profile.second_derivative}, which fixes the degree"
    if profile.period:
        return f"period {profile.period:.6f} confirmed"
    if profile.ode_family:
        return f"ODE ratio identifies it as {profile.ode_family}"
    if profile.growth:
        return f"linearisation identifies it as {profile.growth} (R2={profile.growth_r2:.5f})"
    if profile.homogeneity is not None:
        return f"homogeneous of degree {profile.homogeneity}"
    if profile.arity == 2:
        for ax, d in profile.slices.items():
            if isinstance(d.get("degree"), int) or d.get("ode") or d.get("growth"):
                return f"slice along {ax} is resolved: {d}"
    return None


def vision_warranted(profile, round_index, best_loose_rate):
    """Returns (should_attach, reason, trim_refuted)."""
    if round_index == 1:
        if profile.conflicts:
            return True, ("independent tests contradict each other before any "
                          "paid call: " + "; ".join(profile.conflicts)), False
        known = family_established(profile)
        if known:
            return False, f"router resolved the family ({known})", False
        return True, ("the numeric battery established no family at all -- "
                      "finite differences, growth order, ODE ratios, homogeneity "
                      "and periodicity all abstained"), False

    # later rounds: only when the FAMILY, not merely the detail, is still wrong
    if best_loose_rate < FAMILY_UNRESOLVED_BELOW:
        return True, (f"round {round_index - 1} failed with best loose agreement "
                      f"{best_loose_rate:.0%}: the family, not the detail, is wrong"), True
    return False, "", False


# ------------------------------------------------------------------ rendering

def _samples(oracle, n_args, lo, hi, n, y0=None, log_spaced=False,
             integers_only=False):
    """Sampling has to respect the domain the router mapped. An integer-only
    function sampled on a linspace returns nothing and renders a blank panel --
    a wasted image, which is worse than no image."""
    if integers_only:
        xs = np.unique(np.round(np.geomspace(1, max(hi, 2), min(n, 40))
                                if log_spaced else
                                np.linspace(max(lo, 0), hi, min(n, 40))))
    else:
        xs = (np.geomspace(max(lo, 1e-3), hi, n) if log_spaced
              else np.linspace(lo, hi, n))
    pts = []
    for x in xs:
        v = oracle.val(float(x)) if n_args == 1 else oracle.val(float(x), float(y0))
        if v is not None and math.isfinite(v):
            pts.append((float(x), v))
    return pts


MAX_LOG_DECADES = 14      # a log axis spanning more than this overflows the tick locator


def _trim_dynamic_range(xs, ys, logy):
    xs, ys = list(xs), list(ys)
    """A factorial spans 300 decades; matplotlib's log tick locator computes
    10**decades and overflows. Keep the top MAX_LOG_DECADES so the panel still
    shows the shape without asking the renderer for the impossible."""
    if not logy or not ys:
        return xs, ys
    pos = [(x, y) for x, y in zip(xs, ys) if y > 0]
    if not pos:
        return xs, ys
    top = max(y for _, y in pos)
    floor = top / (10.0 ** MAX_LOG_DECADES)
    kept = [(x, y) for x, y in pos if y >= floor]
    if len(kept) < 3:
        kept = pos[-3:]
    return [p[0] for p in kept], [p[1] for p in kept]


def _panel(ax, xs, ys, title, note, logx=False, logy=False, markers=False):
    xs, ys = list(xs), list(ys)          # callers pass numpy arrays too
    xs, ys = _trim_dynamic_range(xs, ys, logy)
    if len(ys) < 2:
        ax.axis("off")
        return
    if markers or len(xs) < 60:
        ax.plot(xs, ys, "o-", ms=3.5, lw=1.2, color="black")
    else:
        ax.plot(xs, ys, lw=1.8, color="black")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if not (logy or logx):
        # A single spike near a pole flattens everything else into a straight
        # line, hiding exactly the structure the panel exists to show. Clip to
        # robust percentiles so the bulk of the curve is legible; the outlier
        # is still visible running off the top.
        arr = np.asarray(ys, float)
        lo_p, hi_p = np.percentile(arr, [2, 98])
        if hi_p > lo_p and (arr.max() - arr.min()) > 8 * (hi_p - lo_p):
            pad = 0.15 * (hi_p - lo_p)
            ax.set_ylim(lo_p - pad, hi_p + pad)
            note += "\n(y-axis clipped: an outlier runs off-scale)"
    ax.set_title(title, fontsize=11, pad=6)
    ax.text(0.02, 0.94, note, transform=ax.transAxes, fontsize=8.5,
            va="top", color="#444")
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)


def render_1d(oracle, profile):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=100)
    fig.suptitle("Six views of the same black box. "
                 "A straight line in a panel identifies the family.",
                 fontsize=13)

    ints = bool(profile.integer_only)
    wide = _samples(oracle, 1, -50, 50, 400, integers_only=ints)
    near = _samples(oracle, 1, -6, 6, 400, integers_only=ints)
    pos = _samples(oracle, 1, 1e-2, 1e5, 400, log_spaced=True, integers_only=ints)
    pos = [(x, y) for x, y in pos if y > 0]

    if wide:
        _panel(axes[0][0], [p[0] for p in wide], [p[1] for p in wide],
               "f(x) vs x   (wide)", "straight => linear")
    if near:
        _panel(axes[0][1], [p[0] for p in near], [p[1] for p in near],
               "f(x) vs x   (near origin)", "repeating => periodic")
    if pos:
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        _panel(axes[0][2], X, Y, "f vs log x", "straight => LOGARITHMIC", logx=True)
        _panel(axes[1][0], X, Y, "log f vs log x",
               "straight => POWER LAW\nslope = the exponent", logx=True, logy=True)
        _panel(axes[1][1], X, Y, "log f vs x", "straight => EXPONENTIAL", logy=True)

    if near and len(near) > 3:
        xs = [p[0] for p in near]
        d = np.diff([p[1] for p in near])
        _panel(axes[1][2], xs[1:], d, "successive differences",
               "flat => linear\nrepeating => periodic")

    for ax in axes.flat:
        if not ax.lines:
            ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _to_png(fig)


def render_2d(oracle, profile):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=100)
    fig.suptitle("Slices of the same black box. A straight line in a panel "
                 "identifies the family along that axis.", fontsize=12)

    for ax, (axis, label) in zip(axes.flat[:2], (("x", "f(x, y0) vs x"),
                                                 ("y", "f(x0, y) vs y"))):
        drawn = False
        for y0 in (0.0, 1.0, 3.0):
            pts = []
            for t in np.linspace(-20, 20, 300):
                v = (oracle.val(float(t), y0) if axis == "x"
                     else oracle.val(y0, float(t)))
                if v is not None and math.isfinite(v):
                    pts.append((float(t), v))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.5,
                        label=f"other var = {y0:g}")
                drawn = True
        if drawn:
            ax.set_title(label, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25, lw=0.6)
            ax.text(0.02, 0.94, "parallel curves => no interaction term",
                    transform=ax.transAxes, fontsize=8.5, va="top", color="#444")
        else:
            ax.axis("off")

    diag = []
    for t in np.geomspace(1e-2, 1e4, 300):
        v = oracle.val(float(t), float(t))
        if v is not None and v > 0:
            diag.append((float(t), v))
    if diag:
        X = [p[0] for p in diag]
        Y = [p[1] for p in diag]
        _panel(axes[1][0], X, Y, "f(t, t) vs t   (log-log)",
               "straight => HOMOGENEOUS\nslope = the degree", logx=True, logy=True)
        _panel(axes[1][1], X, Y, "f(t, t) vs log t",
               "straight => LOGARITHMIC", logx=True)
    else:
        for ax in axes.flat[2:]:
            ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _to_png(fig)


def _to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render(oracle, profile, n_args):
    return render_1d(oracle, profile) if n_args == 1 else render_2d(oracle, profile)


def to_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


VISION_NOTE = """
A rendered view of the SAME observations is attached, as {panels} panels.
Each panel is labelled with what a straight line in it would mean. Use it to
decide the FAMILY, then use the numeric table above -- which carries full
float precision that the image does not -- to pin down the exact constants.
"""
