"""
LLM client with cost accounting.

Measured pricing for gpt-5.6-luna: $0.20 per million input tokens, $1.20 per
million output. A round of this agent costs roughly a ten-thousandth of a
euro, so the EUR 5 ceiling is not the binding constraint it first appears to
be. Calls are still minimised, but for engineering reasons rather than
survival ones: the marginal information in a counterexample decays sharply
while the cost of a call stays flat, and local analysis is exact where a paid
call is only probable.

Every call's token usage is recorded so the report can state what was actually
spent instead of estimating it.
"""

from __future__ import annotations

import os
import re

USD_PER_INPUT_TOKEN = 0.20 / 1_000_000
USD_PER_OUTPUT_TOKEN = 1.20 / 1_000_000

SYSTEM_PROMPT = (
    "You are a precise mathematical reverse-engineer. Given input-output "
    "observations of an unknown deterministic function, you infer the most "
    "likely closed-form implementation. You answer in the exact requested "
    "format, with no commentary."
)


def parse_candidates(text: str, k: int) -> list[str]:
    """Pull expressions out of the reply.

    Deliberately defensive: this is the one place where a model that is
    "mostly obeying" the format still breaks the whole run. Handles the
    labelled form, markdown emphasis around the label, numbered and bulleted
    lists, fenced code blocks, and a bare list of expressions.

    Note what is NOT done: markdown '**' is stripped only where it wraps the
    LABEL. A blanket text.replace('**', '') also deletes Python's power
    operator, turning x**2 into x2 -- which parses as a name, fails the
    'mentions x' test, and silently drops the candidate. That bug cost a live
    run before it was found.
    """
    if not text:
        return []
    t = re.sub(r"```(?:python)?", "", text).replace("`", "")

    label = re.compile(
        r"^\**\s*CANDIDATE\s*\d*\s*\**\s*[:.)\-]\s*\**\s*(.+?)\s*\**$", re.I)
    numbered = re.compile(r"^\**\s*\d+\s*[.):\-]\s*\**\s*(.+?)\s*\**$")
    bullet = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")

    out = []

    def keep(e):
        e = e.strip().rstrip(",;.").strip()
        if not e or len(e) > 300:
            return
        if not re.search(r"\bx\b", e):          # must actually use the variable
            return
        if re.search(r"[^\x00-\x7f]", e):       # stray prose / non-ascii
            return
        try:
            compile(e, "<probe>", "eval")       # must be a single expression
        except (SyntaxError, ValueError):
            return
        if e not in out:
            out.append(e)

    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = label.match(line) or numbered.match(line) or bullet.match(line)
        keep(m.group(1) if m else line)
    return out[:k]


class _Base:
    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def usd(self):
        return (self.prompt_tokens * USD_PER_INPUT_TOKEN
                + self.completion_tokens * USD_PER_OUTPUT_TOKEN)

    def cost_line(self):
        img = getattr(self, "image_calls", 0)
        extra = f" | {img} of them carried an image" if img else ""
        return (f"{self.calls} LLM calls | {self.prompt_tokens} in / "
                f"{self.completion_tokens} out tokens | ~${self.usd:.5f}{extra}")


class AzureLLM(_Base):
    """The model mandated by the assignment. The key is read from the
    environment and never hard-coded or committed."""

    def __init__(self, max_output_tokens=4096):
        super().__init__()
        from openai import AzureOpenAI
        _load_env()

        key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and paste the key there (.env is git-ignored).")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT",
                                         "gpt-5.6-luna-internship")
        self.max_output_tokens = max_output_tokens
        self.client = AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                       "2024-12-01-preview"),
            azure_endpoint=os.environ.get(
                "AZURE_OPENAI_ENDPOINT",
                "https://industry-x-demos-ai-foundry.cognitiveservices.azure.com/"),
            api_key=key)

    def propose(self, prompt: str, k: int, image_png: bytes = None) -> list[str]:
        """image_png, when present, is attached alongside the text -- never
        instead of it. The table carries full float precision that the render
        does not; both channels go together."""
        self.calls += 1
        content = [{"type": "text", "text": prompt}]
        if image_png:
            import base64
            uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": uri, "detail": "high"}})
            self.image_calls = getattr(self, "image_calls", 0) + 1
        r = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            max_completion_tokens=self.max_output_tokens)
        if getattr(r, "usage", None):
            self.prompt_tokens += r.usage.prompt_tokens or 0
            self.completion_tokens += r.usage.completion_tokens or 0
        raw = r.choices[0].message.content or ""
        parsed = parse_candidates(raw, k)
        if not parsed:
            # A silent "0 candidates" is the worst possible failure: it looks
            # like the model had nothing to say. Show what actually came back,
            # and why it stopped -- a truncated reply and a misformatted one
            # need opposite fixes.
            reason = getattr(r.choices[0], "finish_reason", "?")
            print(f"\n  !! parsed 0 candidates (finish_reason={reason!r}, "
                  f"{len(raw)} chars). Raw reply:\n"
                  f"  ---8<---\n{raw[:2000]}\n  --->8---\n")
        return parsed


class MockLLM(_Base):
    """Offline stand-in for development, so the loop, the filters and every
    edge case can be exercised at zero cost.

    It is NOT a stub returning canned answers: it is a least-squares fitter
    that sees exactly what the real model sees -- the observations parsed back
    out of the prompt -- and never reads secret_functions.py. That makes it a
    legitimate non-LLM baseline. Because its hypothesis space is polynomials
    plus a few fixed transforms, it necessarily fails on anything outside that
    space, which is the cleanest demonstration of what the LLM contributes:
    proposing a structural form, not tuning coefficients inside a fixed basis.
    """

    def propose(self, prompt: str, k: int, image_png: bytes = None) -> list[str]:
        import numpy as np
        self.calls += 1   # the offline fitter has no visual channel; image ignored
        self.prompt_tokens += len(prompt) // 4

        rows = re.findall(r"f\(([^)]*)\)\s*=\s*(-?[\d.eE+-]+)\s*$",
                          prompt, flags=re.MULTILINE)
        samples = []
        for a, b in rows:
            try:
                samples.append(([float(t) for t in a.split(",")], float(b)))
            except ValueError:
                continue
        if not samples:
            return ["x"]
        n = len(samples[0][0])
        out = []
        if n == 1:
            xs = np.array([s[0][0] for s in samples])
            ys = np.array([s[1] for s in samples])
            for deg in (1, 2, 3, 4):
                if len(xs) <= deg:
                    continue
                try:
                    c = np.polyfit(xs, ys, deg)
                except Exception:
                    continue
                terms = []
                for i, coef in enumerate(c):
                    pw = deg - i
                    terms.append(f"({coef:.10g})" if pw == 0 else
                                 f"({coef:.10g})*x" if pw == 1 else
                                 f"({coef:.10g})*x**{pw}")
                out.append(" + ".join(terms))
        else:
            basis = [("1", lambda x, y: 1.0), ("x", lambda x, y: x),
                     ("y", lambda x, y: y), ("x*y", lambda x, y: x * y),
                     ("x**2", lambda x, y: x * x), ("y**2", lambda x, y: y * y)]
            M = np.array([[g(*s[0]) for _, g in basis] for s in samples])
            t = np.array([s[1] for s in samples])
            if M.shape[0] >= M.shape[1]:
                c, *_ = np.linalg.lstsq(M, t, rcond=None)
                out.append(" + ".join(f"({v:.10g})*{nm}" if nm != "1" else f"({v:.10g})"
                                      for v, (nm, _) in zip(c, basis)))
            out += ["x + y", "x*y", "x**2 + y**2"]
        self.completion_tokens += sum(len(e) for e in out) // 4
        return out[:k]


def build_llm(mock: bool, max_output_tokens=4096):
    return MockLLM() if mock else AzureLLM(max_output_tokens)


def _load_env():
    """Load .env and say clearly WHICH thing is missing."""
    import pathlib
    import sys

    here = pathlib.Path(__file__).resolve().parent
    env_file = here / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            print("note: python-dotenv is not installed; .env was parsed directly.",
                  file=sys.stderr)
        else:
            print(f"note: python-dotenv is not installed AND no .env found at "
                  f"{env_file}", file=sys.stderr)
        return
    if not env_file.exists():
        print(f"note: no .env at {env_file} -- did you run "
              f"`cp .env.example .env`?", file=sys.stderr)
    load_dotenv(env_file)