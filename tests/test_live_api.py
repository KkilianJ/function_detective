"""Tests that call the real model. Deselected unless `pytest --live` is given.

Everything else in this suite is free and offline. These cost a few cents and
need credentials, so they never run by accident -- but they are the only place
the API path is exercised at all, and they are the ones worth running once
before submitting.
"""

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm():
    from llm_client import build_llm
    return build_llm(mock=False)


def test_the_model_returns_parsable_candidates(llm):
    """The one path no offline test can reach: whether a real reply survives
    the parser."""
    import reconnaissance as recon
    from evidence import build_evidence
    from planner import build_observations
    from secret_functions import REGISTRY
    from synthesizer import build_prompt

    fn, n = REGISTRY["BOX-A"]
    p = recon.investigate(fn, n)
    o = recon.Oracle(fn, n)
    obs, _ = build_observations(o, p)
    prompt = build_prompt(n, obs, 3, evidence=build_evidence(o, p))

    got = llm.propose(prompt, 3)
    assert got, "the reply parsed to zero candidates"

    import adjudicate as adj
    assert any(adj.compile_candidate(e, n) for e in got), \
        "no returned candidate was a valid Python expression"


def test_a_single_box_is_solved_end_to_end(llm):
    from agent import Detective
    d = Detective(llm, k=5, max_calls_per_box=2, seed=3, verbose=False)
    cases = d.run(labels=["BOX-A"])
    assert cases[0].answer, "BOX-A was not solved against the live model"
    assert llm.usd < 0.05, f"unexpectedly expensive: ${llm.usd:.4f}"
