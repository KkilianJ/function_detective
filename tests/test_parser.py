"""The candidate parser.

This is the single point where a model that is "mostly obeying" the format
still breaks the whole run: a reply that cannot be parsed looks identical to a
model with nothing to say. Each case below is a real shape a model produces.
"""

import pytest

from llm_client import parse_candidates

WELL_FORMED = [
    ("labelled",        "CANDIDATE 1: math.log(x**2 + 1)\nCANDIDATE 2: math.asinh(x)"),
    ("markdown bold",   "**CANDIDATE 1:** math.log(x**2 + 1)\n**CANDIDATE 2:** 2*x"),
    ("bold outside",    "**CANDIDATE 1**: math.log(x**2 + 1)"),
    ("numbered",        "1. math.log(x**2 + 1)\n2. 2*math.log(abs(x)+1)"),
    ("bulleted",        "- math.log(x**2 + 1)\n- math.log1p(x**2)"),
    ("fenced",          "```python\nmath.log(x**2 + 1)\nmath.asinh(x)\n```"),
    ("prose around it", "Looking at the data, the growth is slow.\n\n"
                        "CANDIDATE 1: math.log(x**2 + 1)\nCANDIDATE 2: math.log1p(x*x)"),
    ("two variables",   "CANDIDATE 1: math.sqrt(x**2 + 3*y**2)\nCANDIDATE 2: x*y - x"),
]


@pytest.mark.parametrize("name,text", WELL_FORMED, ids=[c[0] for c in WELL_FORMED])
def test_recognised_shapes(name, text):
    assert parse_candidates(text, 5), f"{name} produced nothing"


def test_power_operator_survives_markdown_stripping():
    """Stripping '**' to remove markdown emphasis also deletes Python's power
    operator: x**2 becomes x2, which then fails the 'mentions x' check and is
    silently dropped. Emphasis must be stripped from the LABEL only."""
    got = parse_candidates("**CANDIDATE 1:** math.log(x**2 + 1)", 5)
    assert got == ["math.log(x**2 + 1)"]


def test_prose_yields_nothing_rather_than_garbage():
    assert parse_candidates("I need more data points to determine this.", 5) == []
    assert parse_candidates("", 5) == []


def test_syntactically_broken_lines_are_dropped_not_fatal():
    got = parse_candidates("CANDIDATE 1: math.log(x**2 + 1\n"
                           "CANDIDATE 2: math.log1p(x**2)", 5)
    assert got == ["math.log1p(x**2)"]


def test_duplicates_are_collapsed():
    got = parse_candidates("CANDIDATE 1: 2*x\nCANDIDATE 2: 2*x\nCANDIDATE 3: 3*x", 5)
    assert got == ["2*x", "3*x"]


def test_k_is_respected():
    text = "\n".join(f"CANDIDATE {i}: {i}*x" for i in range(1, 9))
    assert len(parse_candidates(text, 3)) == 3
