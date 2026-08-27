"""The free filters, and the two decisions they rest on: what "equal" means
for floating point, and what the model is allowed to execute."""

import math

import pytest

import adjudicate as adj


# ------------------------------------------------------------------ sandbox

def test_only_a_single_expression_compiles():
    """compile(..., "eval") rejects statements before anything can run."""
    for hostile in ("x = 5", "import os", "print(1); print(2)",
                    "__import__('os').listdir('/')" if False else "for i in x: pass"):
        assert adj.compile_candidate(hostile, 1) is None, hostile


def test_builtins_are_unreachable():
    """Model output is untrusted input. A bare eval would hand it arbitrary
    code execution -- one line is enough to read the filesystem."""
    f = adj.compile_candidate("__import__('os').listdir('/')", 1)
    assert f is not None, "this one is a valid expression, so it must compile"
    with pytest.raises(Exception):
        f(1.0)                       # ... and must fail at evaluation


def test_math_helpers_are_available_with_and_without_prefix():
    for expr in ("math.log(x)", "log(x)", "math.sqrt(x) + sin(x)"):
        f = adj.compile_candidate(expr, 1)
        assert f is not None and isinstance(f(2.0), float), expr


# ---------------------------------------------------------------- tolerance

def test_comparison_is_relative_not_absolute():
    """Absolute tolerance is meaningless across scales: near 5e299 consecutive
    floats are 7e283 apart, so no correct implementation could satisfy an
    absolute 1e-9."""
    big = 5e299
    assert adj.close(big, big * (1 + 1e-12), 1e-9)
    assert not adj.close(big, big * 1.5, 1e-9)


def test_tolerance_has_an_absolute_floor_near_zero():
    """Relative error is undefined where the true value crosses zero."""
    assert adj.close(0.0, 1e-16, 1e-9)
    assert not adj.close(0.0, 1.0, 1e-9)


def test_strict_is_near_binary_and_loose_is_graded():
    """One number cannot do two jobs: strict decides accept/reject, loose
    diagnoses how close a wrong answer was."""
    truth = lambda x: math.log(x * x + 1)
    near = adj.compile_candidate("math.log(x**2)", 1)      # right family, wrong detail
    wrong = adj.compile_candidate("0.693*x**2", 1)         # wrong family entirely
    pts = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 500.0, 5000.0]

    def rate(f, tol):
        return sum(adj.close(f(x), truth(x), tol) for x in pts) / len(pts)

    # strict is near-binary: neither is exact, so both are rejected outright
    assert rate(near, adj.STRICT_TOL) == 0.0
    assert rate(wrong, adj.STRICT_TOL) == 0.0

    # loose is graded, and must ORDER them: that ordering is what tells the
    # controller whether the family is wrong or only the detail is
    assert rate(near, adj.LOOSE_TOL) > rate(wrong, adj.LOOSE_TOL)
    assert rate(wrong, adj.LOOSE_TOL) < 0.2


# ------------------------------------------------------------ version space

def test_discriminating_probe_separates_look_alike_candidates():
    """Candidates that agree on integers but differ elsewhere -- exactly what
    the live model returned -- must be separated by the probe search."""
    import random
    same = adj.compile_candidate("3*x - 4", 1)
    sneaky = adj.compile_candidate("3*x - 4 + math.sin(2*math.pi*x)", 1)
    args, spread = adj.most_discriminating(
        [("a", same), ("b", sneaky)], 1, random.Random(0))
    assert args is not None and spread > 1e-6
    assert abs(args[0] - round(args[0])) > 1e-3, "the probe must be non-integer"


def test_pac_bound_matches_the_stated_confidence():
    """300 samples surviving bounds the disagreement region below 1% at ~95%."""
    assert adj.pac_confidence(300, eps=0.01) > 0.94
    assert adj.pac_confidence(10, eps=0.01) < 0.11


# --------------------------------------------------------------- diagnosis

def test_residual_shape_identifies_a_missing_constant():
    """A constant residual means the candidate is short exactly that term --
    repairable locally, with the structure still coming from the model."""
    import random
    import reconnaissance as recon
    oracle = recon.Oracle(lambda x: x * x / 2 + 5, 1)
    cand = adj.compile_candidate("x*x/2", 1)
    kind, value, _ = adj.residual_shape(cand, oracle, 1, random.Random(0))
    assert kind == "constant" and abs(value - 5.0) < 1e-9


def test_residual_shape_identifies_a_missing_linear_term():
    import random
    import reconnaissance as recon
    oracle = recon.Oracle(lambda x: x * x + 3 * x, 1)
    cand = adj.compile_candidate("x*x", 1)
    kind, value, _ = adj.residual_shape(cand, oracle, 1, random.Random(0))
    assert kind == "linear" and abs(value[0] - 3.0) < 1e-6
