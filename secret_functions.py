"""
The six "secret functions" that the agent has to reverse-engineer.

Design constraints :
  * Function names must not leak anything about the implementation, so they are
    all called func_1 ... func_6.
  * The agent may only learn about them by calling them and observing outputs.
    It never reads this file.

Difficulty spread:
  single-variable: func_3 (easy)  func_1 (medium)  func_5 (hard)
  two-variable:    func_6 (easy)  func_2 (medium)  func_4 (hard)
  -- the numbering is deliberately shuffled so the index itself carries no hint.
"""

import inspect
import math

# ---------- single-variable ----------

def func_1(x):
    # Easy: linear
    return -2.5 * x + 7


def func_2(x):
    # Medium: shifted quadratic
    return (x - 3) ** 2 + 2


def func_3(x):
    # Hard: logarithmic growth with symmetry
    return math.log(x ** 2 + 4)


def func_4(x):
    # Visual challenge: oscillation with changing amplitude
    return x * math.sin(x)


# ---------- two-variable ----------

def func_5(x, y):
    # Medium: nonlinear interaction
    return x * y + 2 * x - 3 * y


def func_6(x, y):
    # Hard: nonlinear radial structure
    return math.sqrt(x ** 2 + 2 * y ** 2 + 1)


# ---------- registry ----------
REGISTRY = {
    "BOX-A": (func_1, 1),
    "BOX-B": (func_2, 1),
    "BOX-C": (func_3, 1),
    "BOX-D": (func_4, 1),  # x * sin(x) — intended to test visual channel
    "BOX-E": (func_5, 2),
    "BOX-F": (func_6, 2),
}
GROUND_TRUTH = {
    "BOX-A": "-2.5 * x + 7",
    "BOX-B": "(x - 3) ** 2 + 2",
    "BOX-C": "math.log(x ** 2 + 4)",
    "BOX-D": "x * math.sin(x)",
    "BOX-E": "x * y + 2 * x - 3 * y",
    "BOX-F": "math.sqrt(x ** 2 + 2 * y ** 2 + 1)",
}

# ---------- startup sanity check ----------
# It checks that the arity declared in the registry matches the actual number of function arguments.
for label, (fn, declared_arity) in REGISTRY.items():
    actual_arity = len(inspect.signature(fn).parameters)

    if actual_arity != declared_arity:
        raise AssertionError(
            f"{label}: REGISTRY declares {declared_arity} argument(s), "
            f"but {fn.__name__} takes {actual_arity}"
        )