"""The visual channel: when it fires, and when it must not.

The trigger is the whole design decision. "When the run feels stuck" is
arbitrary and indefensible; "when the numeric battery established no family at
all" is checkable, and -- because the router runs before any paid call -- known
at round zero, so a function that needs the picture gets it on its FIRST call.

The channel is off by default. The evidence block carries the router's findings
as text: cheaper, lossless in precision, and needing no vision support.
"""

import math

import pytest

import reconnaissance as recon
import vision

RESOLVED = [
    ("linear",     lambda x: 3 * x - 4),
    ("quadratic",  lambda x: x * x / 2 + 5),
    ("log",        lambda x: math.log(x * x + 4)),
    ("exp",        lambda x: math.exp(x / 10)),
    ("sqrt",       lambda x: math.sqrt(abs(x))),
    ("periodic",   lambda x: math.sin(x) + x / 2),
]

UNRESOLVED = [
    ("amplitude-modulated", lambda x: math.sin(x) * x),
    ("oscillation + pole",  lambda x: math.sin(x) + math.cos(3 * x) / x if x else 0.0),
]


@pytest.mark.parametrize("name,fn", RESOLVED, ids=[c[0] for c in RESOLVED])
def test_silent_when_the_router_resolved_the_family(name, fn):
    p = recon.investigate(fn, 1)
    assert vision.family_established(p), f"{name}: router should have resolved this"
    fires, _, _ = vision.vision_warranted(p, 1, 0.0)
    assert not fires, f"{name}: an image is redundant when the family is known"


@pytest.mark.parametrize("name,fn", UNRESOLVED, ids=[c[0] for c in UNRESOLVED])
def test_fires_on_the_first_call_when_nothing_was_established(name, fn):
    p = recon.investigate(fn, 1)
    assert vision.family_established(p) is None
    fires, why, _ = vision.vision_warranted(p, 1, 0.0)
    assert fires and "no family" in why


def test_later_rounds_fire_only_when_the_family_is_wrong():
    """A first round that failed at 85% loose agreement means the family is
    right and only the detail is wrong -- that calls for residual analysis,
    not a picture."""
    p = recon.investigate(lambda x: math.log(x * x + 4), 1)
    wrong_family, _, trim = vision.vision_warranted(p, 2, 0.05)
    right_family, _, _ = vision.vision_warranted(p, 2, 0.85)
    assert wrong_family and trim, "must fire, and must trim the refuted list"
    assert not right_family


@pytest.mark.parametrize("name,fn", UNRESOLVED, ids=[c[0] for c in UNRESOLVED])
def test_render_produces_a_usable_chart(name, fn):
    p = recon.investigate(fn, 1)
    png = vision.render(recon.Oracle(fn, 1), p, 1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 20_000, "a near-empty chart would be a wasted image"


def test_render_survives_a_300_decade_dynamic_range():
    """A factorial overflows matplotlib's log tick locator, which computes
    10**decades. The range is trimmed rather than asking for the impossible."""
    fn = lambda x: math.factorial(int(x)) if x == int(x) and x >= 0 else 1 / 0
    p = recon.investigate(fn, 1)
    png = vision.render(recon.Oracle(fn, 1), p, 1)          # must not raise
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_two_variable_render():
    fn = lambda x, y: math.sqrt(x * x + 2 * y * y + 1)
    p = recon.investigate(fn, 2)
    png = vision.render(recon.Oracle(fn, 2), p, 2)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
