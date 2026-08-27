"""The non-LLM baseline, and why the gap between it and the model is the result.

MockLLM is not a stub returning canned answers. It is a least-squares fitter
that sees exactly what the real model sees -- the observations parsed back out
of the prompt -- and never reads secret_functions.py. That makes it a fair
control: same information, no language model.

It fails in two distinct ways, and both are the point.
"""

from io import StringIO
from contextlib import redirect_stdout

import pytest

from agent import Detective
from llm_client import build_llm


@pytest.fixture(scope="module")
def baseline():
    llm = build_llm(mock=True)
    d = Detective(llm, k=5, max_calls_per_box=2, seed=7, verbose=False)
    with redirect_stdout(StringIO()):
        return d.run(), llm


def test_the_fitter_cannot_reach_non_polynomial_families(baseline):
    """It has no function library. Faced with a logarithm it can only emit
    higher-order polynomial approximations, which held-out verification refutes
    every time. This is the capability the model supplies."""
    cases, _ = baseline
    by_label = {c.label: c for c in cases}
    for label in ("BOX-C", "BOX-D", "BOX-F"):   # log, x*sin(x), sqrt
        assert by_label[label].answer is None, \
            f"{label} should be out of reach for a polynomial fitter"


def test_the_fitter_solves_strictly_fewer_than_the_model(baseline):
    cases, _ = baseline
    solved = sum(1 for c in cases if c.answer)
    assert 0 < solved < len(cases), \
        "the baseline must be neither useless nor complete for the gap to mean anything"


def test_fitted_coefficients_are_never_exact(baseline):
    """Even where it is numerically right, it is not exactly right:
    (3.4e-09) + 2*x + ... At strict tolerance that is a refutation. This is
    precisely the difference between FITTING and IDENTIFYING."""
    cases, _ = baseline
    refuted = [e for c in cases for e, _, _ in c.refuted]
    assert refuted, "the baseline should have produced refutable candidates"
    assert any("e-" in e or "e+" in e for e in refuted), \
        "fitted output should carry floating-point coefficients"
