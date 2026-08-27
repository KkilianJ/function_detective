"""Shared fixtures, and the gate that keeps paid tests out of the default run.

Everything in this suite is free and offline except the tests marked `live`.
Those are deselected unless `pytest --live` is given, so a reviewer can clone
the repo and run `pytest` without credentials, without cost, and without
surprises.
"""

import math

import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="also run tests that call the real API (costs money)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live (real API calls cost money)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def registry():
    from secret_functions import REGISTRY
    return REGISTRY


@pytest.fixture(scope="session")
def profiles(registry):
    """Reconnaissance is the expensive part; run it once for the whole session."""
    import reconnaissance as recon
    return {label: recon.investigate(fn, n) for label, (fn, n) in registry.items()}


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """What a green run actually proves. A list of dots says the code does what
    the tests say; this says what the tests were chosen to establish."""
    if exitstatus != 0:
        return
    live = " (including live API calls)" if config.getoption("--live") else \
           "\n  No API key was used and nothing was spent. `pytest --live` adds " \
           "the paid tests."
    terminalreporter.write_line("")
    terminalreporter.write_line(
        "  What a green run establishes:\n"
        "    * the structural router states only true facts, and abstains rather\n"
        "      than guess -- verified on 14 functions that are not the six targets\n"
        "    * the free filters isolate the correct candidate out of a diverse\n"
        "      committee, at ONE paid call per function\n"
        "    * a non-LLM baseline with identical information solves strictly fewer,\n"
        "      and its fitted coefficients are never exact -- that gap is what the\n"
        "      model supplies\n"
        "    * the evidence block carries measurements and scoped exclusions, and\n"
        "      never a guess about what the function is\n"
        "    * the visual channel engages only when no numeric criterion could\n"
        "      decide, and no function name ever reaches the prompt"
        + live)
