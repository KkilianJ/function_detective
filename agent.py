"""
Function Detective

This program tries to discover the implementation of unknown mathematical
functions using only their inputs and outputs.

The main idea is simple:

    1. Explore the black box
       Try different inputs and observe the outputs to learn useful patterns.

    2. Choose useful evidence
       Instead of sending random examples to the LLM, select calculated input/output pairs
       that reveal useful patterns such as symmetry, growth, or interactions
       between variables.

    3. Ask the LLM for hypotheses
       The LLM receives only the selected evidence. It never receives the
       function name, source code, or implementation.

    4. Verify the hypotheses
       Each time, the LLM proposes five candidate functions. These candidates
       are then tested against the black box, and incorrect ones are rejected.

    5. Learn from failures
       If every candidate is wrong, the agent learns from the failures by identifying
       useful information, such as counterexamples or where the candidates fail.
       This information is then used to improve the next LLM request.

    6. Iterate step 2 to 5 and Stop when a candidate is reliable
       A surviving candidate is tested on fresh inputs before it is accepted.

The core questions behind the design are:

    1. What kind of information should the agent provide to the LLM?

    2. How confident can we be that our hypothesis matches the unknown function?

    3. When a hypothesis fails, what should the agent learn from the failure?

    4. What should the agent do when it keeps learning from failures but still
       cannot find the correct function?

        a) If the budget for a difficult function is exhausted, pause it and move on
        to the other functions. If the other functions are solved and there is
        still budget remaining, return to the difficult function.

        b) When returning to the difficult function, try different input values and
        plot their outputs to show the function's overall trend. Then send both
        the textual evidence and the plot to the LLM in the same request.

"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

import adjudicate as adj
import reconnaissance as recon
import vision
from llm_client import build_llm
from evidence import build_evidence
from planner import build_observations
from secret_functions import GROUND_TRUTH, REGISTRY
from synthesizer import build_prompt, summarise_common_failure


@dataclass
class Case:
    """Stores everything the agent currently knows about one black-box function."""
    # ---------- Basic information ----------
    label: str
    # Opaque name of the black box, e.g. "BOX-A".
    fn: object
    # The actual callable black-box function used internally for input/output queries.
    n_args: int
    # Number of arguments the function accepts, e.g. 1 for f(x), 2 for f(x, y).

    # ---------- Reconnaissance ----------
    oracle: recon.Oracle = None
    # Interface used to query the black box and record its input/output results.
    profile: recon.Profile = None
    # Structural information discovered during reconnaissance, such as symmetry, growth, polynomial behaviour, or interactions.
    observations: list = field(default_factory=list)
    # Selected input/output pairs collected from the black box.
    plan_rationale: list = field(default_factory=list)
    # Records why particular inputs were selected by the probe planner.

    # ---------- Failed hypotheses ----------
    refuted: list = field(default_factory=list)
    # Rejected LLM candidates stored as (expression, agreement rate, reason).
    refuted_fns: list = field(default_factory=list)
    # Compiled versions of rejected candidate functions for later comparison.
    llm_calls: int = 0
    # Number of LLM calls already used for this black box.

    # ---------- Current result ----------
    answer: str = None
    # Final accepted expression. Remains None until the function is solved.
    best_expr: str = None
    # Best candidate expression found so far, even if it has not passed verification.
    best_rate: float = 0.0
    # Agreement rate of the best candidate found so far.
    best_detail: str = ""
    # Additional information explaining the performance of the best candidate.

    # ---------- Failure diagnosis ----------
    common_failure: str = None
    # A failure pattern shared by multiple rejected candidates.
    residual_note: str = None
    # Notes about residual errors: true output - candidate output.

    # ---------- Escalation / visual fallback ----------
    escalations: list = field(default_factory=list)
    # Records when the agent switches to a more advanced strategy.
    vision_rounds: list = field(default_factory=list)
    # Records rounds where generated plots are sent to the LLM together with textual evidence.

    # ---------- Case status ----------
    done: bool = False
    # True when this black-box function has been successfully solved.
    exhausted: bool = False
    # True when the normal LLM-call budget for this box has been exhausted.


class Detective:
    def __init__(self, llm, k=5, max_calls_per_box=3, seed=7, verbose=True,
                 vision=False):
        self.llm = llm
        self.vision = vision
        self.k = k
        self.max_calls_per_box = max_calls_per_box
        self.rng = random.Random(seed)
        self.verbose = verbose

    def log(self, msg=""):
        if self.verbose:
            print(msg)

    # ---------------------------------------------------------- stage 0

    def reconnoitre(self, c: Case):
        c.profile = recon.investigate(c.fn, c.n_args)
        c.oracle = recon.Oracle(c.fn, c.n_args)
        c.observations, c.plan_rationale = build_observations(c.oracle, c.profile)
        c.evidence = build_evidence(c.oracle, c.profile)
        self.log(f"\n{'=' * 74}\n{c.label}  ({c.n_args} argument(s))\n{'=' * 74}")
        self.log(c.profile.render())
        self.log("\n  probe plan:")
        for reason, pts in c.plan_rationale:
            shown = ", ".join(f"{p}" for p in pts[:6]) + (" ..." if len(pts) > 6 else "")
            self.log(f"    - {reason}\n        {shown}")
        self.log(f"  -> evidence block {len(c.evidence) // 4} tokens, "
                 f"{len(c.observations)} observations, "
                 f"{c.profile.oracle_calls + c.oracle.calls} free black-box calls, "
                 f"0 paid calls so far")

    # ---------------------------------------------------------- one round

    def step(self, c: Case) -> bool:
        """One paid call plus all the free work around it. True if solved."""
        round_index = c.llm_calls + 1
        use_vision, why, trim = (False, "", False)
        if self.vision:
            use_vision, why, trim = vision.vision_warranted(
                c.profile, round_index, c.best_rate)

        image = None
        if use_vision:
            try:
                image = vision.render(c.oracle, c.profile, c.n_args)
                panels = 6 if c.n_args == 1 else 4
                c.vision_rounds.append((round_index, why))
                self.log(f"\n  [vision]      attaching {panels} rendered panels "
                         f"({len(image) // 1024} KB)\n"
                         f"                reason: {why}")
            except Exception as exc:
                self.log(f"  [vision]      render failed ({type(exc).__name__}); "
                         f"continuing text-only")
                image, use_vision = None, False

        prompt = build_prompt(c.n_args, c.observations, self.k,
                              refuted=c.refuted or None,
                              common_failure=c.common_failure,
                              residual_note=c.residual_note,
                              vision_panels=(6 if c.n_args == 1 else 4) if use_vision else 0,
                              trim_refuted=trim,
                              evidence=c.evidence)
        exprs = self.llm.propose(prompt, self.k, image_png=image)
        c.llm_calls += 1
        seen = {e for e, _, _ in c.refuted}
        fresh = [e for e in exprs if e not in seen]
        if exprs and not fresh:
            # The model is deterministic: an identical prompt yields an identical
            # answer, and re-asking spends money for nothing. If a round returns
            # only candidates already refuted, the loop has stopped learning and
            # the box is parked rather than ground on.
            c.exhausted = True
            self.log(f"\n  -- call {c.llm_calls}: every candidate was already "
                     f"refuted; no new information, parking this box --")
            return False
        exprs = fresh
        self.log(f"\n  -- call {c.llm_calls} -> {len(exprs)} candidate(s) --")
        for e in exprs:
            self.log(f"     {e}")
        if not exprs:
            return False

        alive = []
        for e in exprs:
            f = adj.compile_candidate(e, c.n_args)
            if f is None:
                self.record_refuted(c, e, None, "not a valid Python expression", None)
                self.log(f"     [compile]     refuted: {e}")
                continue
            alive.append((e, f))

        # (a) structural -- free, refutes at family level
        survivors = []
        for e, f in alive:
            why = adj.structural_conflict(e, f, c.profile, c.oracle)
            if why:
                self.record_refuted(c, e, None, why, f)
                self.log(f"     [structural]  refuted: {e}")
            else:
                survivors.append((e, f))
        alive = survivors

        # (b) consistency with the data it was shown -- free
        survivors = []
        for e, f in alive:
            why = adj.consistency_failure(f, c.observations)
            if why:
                rate, _, _ = adj.agreement(f, c.oracle, c.n_args, self.rng, adj.LOOSE_TOL)
                self.record_refuted(c, e, rate, why, f)
                self.log(f"     [consistency] refuted: {e}   (loose agreement {rate:.0%})")
            else:
                survivors.append((e, f))
        alive = survivors

        # (c) version space -- one free probe can kill most of the rest
        while len(alive) > 1:
            args, spread = adj.most_discriminating(alive, c.n_args, self.rng)
            if args is None or spread < 1e-9:
                self.log(f"     [version]     {len(alive)} candidates agree everywhere "
                         f"tested; taking the simplest")
                alive = [min(alive, key=lambda p: len(p[0]))]
                break
            truth = c.oracle.val(*args)
            sig = ", ".join(f"{a:g}" for a in args)
            kept = []
            for e, f in alive:
                if adj.close(adj.safe_eval(f, args), truth, adj.STRICT_TOL):
                    kept.append((e, f))
                else:
                    rate, _, _ = adj.agreement(f, c.oracle, c.n_args, self.rng, adj.LOOSE_TOL)
                    self.record_refuted(c, e, rate,
                                        f"at f({sig}) the black box gives {truth!r}, "
                                        f"this candidate gives "
                                        f"{adj.safe_eval(f, args)!r}", f)
            self.log(f"     [version]     probed f({sig}) = {truth!r} -> "
                     f"{len(kept)}/{len(alive)} survive")
            if not kept:
                alive = []
                break
            if len(kept) == len(alive):
                break
            alive = kept

        # (d) PAC absolute verification -- winning the committee is not being right
        for e, f in alive:
            rate, bad, tested = adj.agreement(f, c.oracle, c.n_args, self.rng, adj.STRICT_TOL)
            self.track_best(c, e, f, rate)
            if rate >= 1.0:
                conf = adj.pac_confidence(tested)
                c.answer = e
                c.done = True
                self.log(f"     [PAC]         {e}")
                self.log(f"                   agrees on all {tested} stratified samples "
                         f"-> disagreement region under 1% with confidence "
                         f"{conf:.2%}. CONFIDENT.")
                return True
            self.record_refuted(c, e, rate,
                                f"failed absolute verification "
                                f"({rate:.0%} of {tested} fresh samples)", f)
            self.log(f"     [PAC]         refuted: {e}   ({rate:.0%} of {tested})")

        self.diagnose(c)
        return False

    # ---------------------------------------------------------- bookkeeping

    def record_refuted(self, c: Case, expr, rate, why, fn):
        """Every candidate that compiles feeds the best-so-far ranking, no
        matter which filter killed it. Otherwise a box whose candidates all die
        at the structural filter has nothing to report at the end, and the
        anytime output is blank exactly when it is most needed."""
        if fn is not None and rate is None:
            rate, _, _ = adj.agreement(fn, c.oracle, c.n_args, self.rng, adj.LOOSE_TOL)
        c.refuted.append((expr, rate, why))
        if fn is not None:
            c.refuted_fns.append(fn)
            self.track_best(c, expr, fn, rate)

    def track_best(self, c: Case, expr, f, strict_rate):
        rate, bad, tested = adj.agreement(f, c.oracle, c.n_args, self.rng, adj.LOOSE_TOL)
        if rate > c.best_rate:
            c.best_rate, c.best_expr = rate, expr
            if bad:
                region = sorted(abs(a[0][0]) for a in bad)
                c.best_detail = (f"mismatches concentrated around |x| in "
                                 f"[{region[0]:.3g}, {region[len(region)//2]:.3g}]")

    def diagnose(self, c: Case):
        """Free analysis that decides what the next prompt should carry."""
        rates = [r for _, r, _ in c.refuted if r is not None]
        best = max(rates) if rates else 0.0
        c.common_failure = summarise_common_failure(c.oracle, c.n_args,
                                                    c.refuted_fns, self.rng)
        c.residual_note = None
        if best > 0.6 and c.best_expr:
            f = adj.compile_candidate(c.best_expr, c.n_args)
            if f:
                shape = adj.residual_shape(f, c.oracle, c.n_args, self.rng)
                if shape:
                    c.residual_note = f"For `{c.best_expr}`: {shape[2]}"
        verdict = ("the family looks right and only the detail is wrong"
                   if best > 0.6 else "the family itself is wrong")
        c.escalations.append(f"round {c.llm_calls}: best loose agreement "
                             f"{best:.0%} -> {verdict}")
        self.log(f"     [diagnosis]   best loose agreement {best:.0%} -> {verdict}")
        if c.residual_note:
            self.log(f"                   {c.residual_note}")

    # ---------------------------------------------------------- controller

    def run(self, labels=None):
        cases = [Case(l, fn, n) for l, (fn, n) in REGISTRY.items()
                 if not labels or l in labels]
        for c in cases:
            self.reconnoitre(c)

        # first pass: one call each, so an easy box is never starved by a hard one
        self.log(f"\n{'#' * 74}\n# round-robin pass 1: one call per box\n{'#' * 74}")
        for c in cases:
            if not c.done:
                self.step(c)

        # then hand the leftover budget to whichever unsolved box is closest,
        # measured by loose agreement -- progress, not merely "still open"
        while True:
            open_ = [c for c in cases if not c.done and not c.exhausted
                     and c.llm_calls < self.max_calls_per_box]
            if not open_:
                break
            c = max(open_, key=lambda c: c.best_rate)
            self.log(f"\n{'#' * 74}\n# escalating {c.label} "
                     f"(closest unsolved: {c.best_rate:.0%})\n{'#' * 74}")
            self.step(c)

        return cases


# ------------------------------------------------------------------- report

def report(cases, llm):
    print(f"\n\n{'=' * 74}\nFINAL REPORT\n{'=' * 74}")
    solved = 0
    for c in cases:
        v = "x" if c.n_args == 1 else "x, y"
        name = c.label.replace("-", "_").lower()
        if c.answer:
            solved += 1
            print(f"\n{c.label}  SOLVED")
            for r, why in c.vision_rounds:
                print(f"    visual channel  : engaged on round {r} -- {why}")
            print(f"    def revealed_{name}({v}):")
            print(f"        return {c.answer}")
        else:
            # anytime output: never blank. What was found, how close, where it fails.
            print(f"\n{c.label}  NOT SOLVED")
            if c.best_expr:
                print(f"    best candidate  : {c.best_expr}")
                print(f"    agreement       : {c.best_rate:.1%} of 300 stratified "
                      f"samples at 1% relative tolerance")
                if c.best_detail:
                    print(f"    where it fails  : {c.best_detail}")
            for e in c.escalations:
                print(f"    escalation      : {e}")
            for r, why in c.vision_rounds:
                print(f"    visual channel  : engaged on round {r} -- {why}")
            if c.exhausted:
                print(f"    parked          : the model stopped producing candidates "
                      f"it had not already had refuted")
        print(f"    ground truth (human check): {GROUND_TRUTH[c.label]}")
        print(f"    {c.profile.oracle_calls + c.oracle.calls} free black-box calls | "
              f"{c.llm_calls} paid calls | {len(c.refuted)} candidates refuted")

    print(f"\n{'-' * 74}")
    print(f"Solved {solved}/{len(cases)}")
    print(llm.cost_line())


def main():
    ap = argparse.ArgumentParser(description="Function Detective")
    ap.add_argument("--mock", action="store_true", help="offline, no API calls")
    ap.add_argument("--box", action="append", help="restrict to these labels")
    ap.add_argument("--k", type=int, default=5, help="candidates per call")
    ap.add_argument("--budget", type=int, default=3, help="max paid calls per box")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--vision", action="store_true",
                    help="enable the visual channel (off by default: the evidence "
                         "block carries the router's findings as text, which is "
                         "cheaper, lossless, and needs no vision support)")
    a = ap.parse_args()

    llm = build_llm(mock=a.mock, max_output_tokens=a.max_output_tokens)
    d = Detective(llm, k=a.k, max_calls_per_box=a.budget, seed=a.seed,
                  vision=a.vision)
    report(d.run(labels=a.box), llm)


if __name__ == "__main__":
    main()
