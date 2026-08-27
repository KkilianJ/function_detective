"""End to end, with a scripted model standing in for the API.

The offline fitter in MockLLM only fits polynomials, so it cannot exercise the
path that matters most: a correct non-polynomial form arriving among
structurally different wrong ones. This scripted committee can.

What is being asserted is not just "it gets the right answer" but the cost
shape the whole design exists to produce: ONE paid call per function, with
every elimination happening for free.
"""

import math

import pytest

from agent import Detective
from llm_client import _Base

SCRIPTS = {
    1: ["-2.5 * x + 7",
        "(x - 3) ** 2 + 2",
        "math.log(x ** 2 + 4)",
        "x * math.sin(x)",
        "-2.5 * x + 7 + math.sin(2 * math.pi * x)",
        "2 * math.log(abs(x) + 2)",
        "x ** 2 - 6 * x + 11",
        "math.sqrt(abs(x)) * 2",
        "math.asinh(x)"],
    2: ["x * y + 2 * x - 3 * y",
        "math.sqrt(x ** 2 + 2 * y ** 2 + 1)",
        "math.sqrt(x ** 2 + y ** 2)",
        "x * y - x - y",
        "2 * x + y",
        "x + 2 * y"],
}


class ScriptedLLM(_Base):
    """Returns a fixed, structurally diverse committee containing one correct
    candidate. Advancing by the number already refuted mimics a model that
    responds to the refutation list."""

    def propose(self, prompt, k, image_png=None):
        self.calls += 1
        self.prompt_tokens += len(prompt) // 4
        self.image_calls = getattr(self, "image_calls", 0) + (1 if image_png else 0)
        n = 2 if "f(x, y)" in prompt else 1
        pool = SCRIPTS[n]
        skip = prompt.count("  - ")
        out = pool[skip:skip + k] or pool[-k:]
        self.completion_tokens += sum(len(e) for e in out) // 4
        return out


@pytest.fixture(scope="module")
def run():
    llm = ScriptedLLM()
    d = Detective(llm, k=5, max_calls_per_box=3, seed=11, verbose=False)
    return d.run(), llm


def test_all_six_are_solved(run):
    cases, _ = run
    unsolved = [c.label for c in cases if not c.answer]
    assert not unsolved, f"unsolved: {unsolved}"


def test_one_paid_call_per_function(run):
    """The cost shape the architecture exists to produce. Every elimination --
    structural, consistency, version space, PAC -- happens for free."""
    cases, llm = run
    assert llm.calls == len(cases)
    assert all(c.llm_calls == 1 for c in cases)


def test_answers_match_ground_truth_numerically(run):
    """Symbolic equality is not claimed; agreement on fresh samples is."""
    import random

    import adjudicate as adj
    cases, _ = run
    rng = random.Random(5)
    for c in cases:
        f = adj.compile_candidate(c.answer, c.n_args)
        for _ in range(200):
            args = tuple(round(rng.uniform(-30, 30), 4) for _ in range(c.n_args))
            truth = c.oracle.val(*args)
            if truth is None:
                continue
            assert adj.close(adj.safe_eval(f, args), truth, adj.STRICT_TOL), \
                f"{c.label}: {c.answer} disagrees at {args}"


def test_no_image_was_needed(run):
    """The visual channel is off by default: the evidence block carries the
    router's findings as text, losslessly and without needing vision support."""
    _, llm = run
    assert getattr(llm, "image_calls", 0) == 0


def test_the_prompt_never_leaks_a_name(run):
    """The assignment forbids the model from learning the function's name. The
    labels, the identifiers and the exception messages must all stay out."""
    from evidence import build_evidence
    from planner import build_observations
    from synthesizer import build_prompt
    cases, _ = run
    for c in cases:
        prompt = build_prompt(c.n_args, c.observations, 5, evidence=c.evidence)
        for banned in (c.label, "func_", "secret_functions", "REGISTRY"):
            assert banned not in prompt, f"{banned!r} leaked into {c.label}'s prompt"
