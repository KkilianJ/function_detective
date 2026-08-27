"""The router must never state a false fact. Abstaining is always acceptable.

That asymmetry is the whole point: a wrong structural claim propagates into
every later round and cannot be retracted, whereas an abstention merely falls
back to the generic path. So each case below asserts only what the router
should be *certain* of, and never asserts that it must reach a conclusion.

The functions here are deliberately awkward and deliberately NOT the six
targets -- restricted domains, poles, integer-only, fast overflow, a periodic
one, and an amplitude-modulated impostor that merely looks periodic. The six
targets can be swapped out at any time; these cannot be tuned against.
"""

import math

import pytest

import reconnaissance as recon

TAU = 2 * math.pi

CASES = [
    # (name, fn, arity, {attribute: expected})
    ("linear",        lambda x: 3 * x - 4,              1, {"poly_degree": 1,
                                                            "parity": "neither"}),
    ("quadratic",     lambda x: x * x / 2 + 5,          1, {"poly_degree": 2,
                                                            "parity": "even"}),
    ("cubic",         lambda x: x ** 3 - 2 * x,         1, {"poly_degree": 3,
                                                            "parity": "odd"}),
    ("constant",      lambda x: 7,                      1, {"constant": True}),
    ("log",           lambda x: math.log(x * x + 1),    1, {"poly_is_none": True,
                                                            "parity": "even",
                                                            "accepts_complex": False}),
    ("exp",           lambda x: math.exp(x / 10),       1, {"poly_is_none": True,
                                                            "ode_family": "exponential"}),
    ("sqrt_abs",      lambda x: math.sqrt(abs(x)),      1, {"homogeneity": 0.5,
                                                            "parity": "even"}),
    ("reciprocal",    lambda x: 1 / x,                  1, {"homogeneity": -1.0,
                                                            "parity": "odd"}),
    ("abs",           lambda x: abs(x),                 1, {"homogeneity": 1.0,
                                                            "parity": "even"}),
    ("periodic",      lambda x: math.sin(x) + x / 2,    1, {"parity": "odd"}),
    ("fake_periodic", lambda x: math.sin(x) * x,        1, {"period": None,
                                                            "parity": "even"}),
    ("additive_2d",   lambda x, y: 2 * x + y,           2, {"separable": True,
                                                            "symmetric": False}),
    ("interacting_2d", lambda x, y: x * y - x - y,      2, {"separable": False,
                                                            "symmetric": True}),
    ("norm_2d",       lambda x, y: math.sqrt(x * x + 3 * y * y), 2,
                                                           {"homogeneity": 1.0,
                                                            "symmetric": False}),
]


@pytest.mark.parametrize("name,fn,arity,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_router_states_only_true_facts(name, fn, arity, expected):
    p = recon.investigate(fn, arity)
    for key, want in expected.items():
        got = getattr(p, key)
        if isinstance(want, float):
            assert got is not None and abs(got - want) < 1e-3, \
                f"{name}: {key} was {got}, expected {want}"
        else:
            assert got == want, f"{name}: {key} was {got}, expected {want}"


def test_inflection_location_recovers_the_parameter():
    """log(x**2 + a) inflects at sqrt(a). Finding the inflection reads a off,
    which is a parameter rather than merely a family."""
    for a in (1, 4, 9, 2.5):
        p = recon.investigate(lambda x, a=a: math.log(x * x + a), 1)
        assert p.inflections, f"no inflection found for log(x**2 + {a})"
        assert abs(p.inflections[0] - math.sqrt(a)) < 1e-4


def test_period_is_confirmed_not_merely_proposed():
    """FFT proposes; the constancy of f(x+T) - f(x) confirms. sin(x)*x is
    amplitude-modulated and must be rejected despite looking periodic."""
    p = recon.investigate(lambda x: math.sin(x) + x / 2, 1)
    assert p.period is not None and abs(p.period - TAU) < 1e-2

    q = recon.investigate(lambda x: math.sin(x) * x, 1)
    assert q.period is None


def test_finite_differences_require_extrapolation():
    """Every smooth function is locally polynomial (Taylor), so a converging
    difference table proves nothing on its own. exp must be rejected."""
    p = recon.investigate(lambda x: math.exp(x / 10), 1)
    assert p.poly_is_none, "exp was accepted as a polynomial"
    assert not isinstance(p.poly_degree, int)


def test_integer_only_domain_is_detected_and_most_tests_abstain():
    p = recon.investigate(
        lambda x: math.factorial(int(x)) if x == int(x) and x >= 0 else 1 / 0, 1)
    assert p.integer_only is True
    assert p.accepts_complex is False
    # It must not invent a growth order or a period for a discrete function.
    assert p.growth is None and p.period is None


def test_router_never_raises_on_hostile_input():
    """A test that cannot run must abstain, not abort the whole profile."""
    hostile = [lambda x: 1 / 0, lambda x: math.sqrt(x - 1e9),
               lambda x: float("nan"), lambda x: "not a number"]
    for fn in hostile:
        p = recon.investigate(fn, 1)          # must not raise
        assert p is not None
