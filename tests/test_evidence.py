"""The evidence block: what the router measured, stated as text.

Three kinds of statement may reach the model: a measurement, an exclusion, and
the scope an exclusion was tested over. One kind may not: a positive guess
about what the function IS. A wrong measurement is impossible and a wrong
exclusion merely rules out too little, but a wrong suggestion steers every
later round with nothing to retract it.
"""

import math
import re

import pytest

import reconnaissance as recon
from evidence import build_evidence


def _evidence_for(fn, arity=1):
    p = recon.investigate(fn, arity)
    return build_evidence(recon.Oracle(fn, arity), p)


def test_measurements_are_stated_with_precision_not_adjectives():
    """'approximately even' is a vague claim the model can only use vaguely.
    'agrees to 17 significant figures at five pairs' is a hard constraint."""
    text = _evidence_for(lambda x: math.log(x * x + 4))
    assert "significant figures" in text
    assert re.search(r"at least \d+ significant figures", text)


def test_every_exclusion_carries_the_range_it_was_tested_over():
    """'no clear fit' may be a fact about the function or an artefact of too
    narrow a window; without the range there is no way to tell."""
    text = _evidence_for(lambda x: math.sin(x) * x)
    for line in text.splitlines():
        if "EXCLUDED" in line:
            assert re.search(r"(over|at) [a-z]* ?[=x]", line), \
                f"exclusion without a stated scope: {line}"


def test_no_positive_guess_about_the_family_ever_appears():
    """The block may say what was measured and what was ruled out. It may not
    say 'this looks like a logarithm'."""
    for fn in (lambda x: math.log(x * x + 4),
               lambda x: math.sin(x) * x,
               lambda x: -2.5 * x + 7):
        text = _evidence_for(fn).lower()
        for banned in ("looks like", "probably", "likely to be", "appears to be",
                       "suggests that", "we think", "my guess"):
            assert banned not in text, f"{banned!r} leaked into the evidence block"


def test_untested_things_are_omitted_entirely():
    """An abstention is not evidence. A test that ran and failed is
    information; a test that could not run is not."""
    text = _evidence_for(
        lambda x: math.factorial(int(x)) if x == int(x) and x >= 0 else 1 / 0)
    assert "unknown" not in text.lower()
    assert "?" not in text


def test_inflection_locations_reach_the_model_as_measurements():
    """log(x**2 + 4) inflects at exactly 2. That single number carries the
    constant term, and it is a measurement rather than an inference."""
    text = _evidence_for(lambda x: math.log(x * x + 4))
    assert "2.000000" in text


def test_two_variable_structure_is_reported():
    text = _evidence_for(lambda x, y: x * y + 2 * x - 3 * y, 2)
    assert "Interaction term PRESENT" in text
    assert "NOT" in text and "interchangeable" in text     # asymmetry stated
