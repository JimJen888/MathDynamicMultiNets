"""
The controller: what to do next.

Figure 1's controller sits between the two sets of read/write heads and is
"commanded/instructed cooperatively by both the contents on the tape in the
abstract domain and those in the specific domain". That is a decision problem
with no closed form -- which rule is worth forming, what data would settle it,
whether two rules are secretly the same rule -- so it is the one component here
that is a language model.

Two implementations of the same interface:

    LLMController      Claude decides, one tool call at a time, against the
                       instruction set in tools.py.
    ScriptedController a fixed plan runs through the identical tool table. Used
                       by the examples and the tests so the whole pipeline is
                       exercisable offline, and so an LLM run can be compared
                       against a known-good sequence.

The LLM never writes or executes code. It picks an operation and its arguments;
the machine performs it. Everything it can do is bounded by tools.py, which is
what makes an autonomous multi-hour run over a rule library a safe thing to
start.

The loop is written out rather than delegated to the SDK's tool runner because
every call has to be journaled onto the machine and counted against an explicit
step budget -- the run's cost is part of what the conciseness objective is
weighed against, so it is not a detail to leave implicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .machine import RenMachine
from .tools import Tool, anthropic_schemas, build_tools

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are the controller of a Ren machine: a non-Turing computer with two tapes.

  * The ABSTRACT tape holds symbols and digits. Exact calculation, rigid
    definitions and formal logic happen there.
  * The SPECIFIC tape holds images and rendered text. Structure is expressed as
    layout, and reading it back requires perception.

The two are connected only by MAPPING RULES. Reasoning on this machine is the
transfer of mapping rules: you form rules, verify them, chain them, and combine
them. Rules can be prior knowledge (exact, already installed), memorized
tables, learned neural rules, chains of other rules, or a base rule with
specialists overriding it on hard cases.

YOUR OBJECTIVE is a single number, reported by library_report:

    J = (bits to write the library down) + 1000 * (rule applications needed to
        solve the benchmark tasks)

Minimise J. A rule earns its place only if it shortens more derivation than it
costs to state. Concretely that means: form rules that are genuinely reusable,
collapse recurring chains into one rule, and delete rules that duplicate
another rule's behaviour or that no proof uses.

HOW TO WORK

1. Look before acting: inspect_machine, then show_catalogue.
2. Add the benchmark tasks you care about (add_task) BEFORE optimising -- the
   benchmark is the definition of "useful", and without it every rule looks
   like dead weight.
3. To form a rule: decide what mapping it performs and declare it, generate
   experiments, label them with an oracle, train, then VERIFY ON FRESH DATA
   (a different seed, ideally a harder tail). Training accuracy is not evidence.
4. Prefer strong evidence. An oracle's kind tells you what a check is worth:
   'definitional' bottoms out in a definition; 'derived' inherits the errors of
   the rules it used; 'constructed' only shows the machine can read its own
   handwriting. Where you can, also cross-check a learned rule against a chain
   of rules already trusted (verify_against_rules) -- agreement between two
   independent routes is the strongest thing available.
5. Only trusted rules may appear in a proof. If a rule verifies well but below
   threshold, grow_ensemble trains a specialist on its failures rather than
   retraining the whole thing.
6. Confidence multiplies along a chain. Ten steps at 0.99 each is a 0.90 proof.
   Short, well-verified chains beat long ones.
7. Re-run library_report after changes, and use simplify_library (dry run
   first) to remove what stopped paying for itself.

Work autonomously. Do not ask the user questions -- state an assumption and
continue. Call one or a few tools per turn, read what came back, and adjust.
Call finish when the goal is met or when you have stopped making progress,
with a summary of the rules you formed and what J ended at.
"""


@dataclass
class ControllerRun:
    """What a controller run did, for the caller and for the record."""

    steps: int = 0
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    finished: bool = False
    stop_reason: str = ""

    def summary(self) -> str:
        calls = ", ".join(name for name, _ in self.tool_calls[-12:])
        return (f"{self.steps} controller steps, {len(self.tool_calls)} tool calls "
                f"({'finished' if self.finished else self.stop_reason or 'budget spent'})"
                f"\n  recent: {calls}")


class BaseController:
    def __init__(self, machine: RenMachine, log: Callable[[str], None] = print):
        self.machine = machine
        self.tools: dict[str, Tool] = build_tools(machine)
        self.log = log

    def _invoke(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            return f"ERROR: no tool named {name!r}; available: {sorted(self.tools)}"
        self.log(f"> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        out = self.tools[name].call(**args)
        self.log("\n".join("  " + ln for ln in out.splitlines()[:24]))
        return out


class ScriptedController(BaseController):
    """A fixed plan through the same instruction set.

    Not a toy: it is how the examples run without an API key, and it is the
    reference an LLM run gets compared against. `stop_on_error` is off by
    default because a plan that hits an error should show what the controller
    would see and carry on, exactly as the LLM path does.
    """

    def run(self, plan: Sequence[tuple[str, dict[str, Any]]],
            stop_on_error: bool = False) -> ControllerRun:
        run = ControllerRun()
        for name, args in plan:
            run.steps += 1
            out = self._invoke(name, dict(args))
            run.tool_calls.append((name, dict(args)))
            run.transcript.append(out)
            if name == "finish":
                run.finished = True
                break
            if stop_on_error and out.startswith("ERROR"):
                run.stop_reason = f"error in {name}"
                break
        return run


class LLMController(BaseController):
    """Claude drives the machine through tools.py.

    Requires the `anthropic` package and credentials (ANTHROPIC_API_KEY, or an
    `ant auth login` profile -- the zero-argument client picks up either).
    """

    def __init__(self, machine: RenMachine, model: str = DEFAULT_MODEL,
                 max_steps: int = 40, max_tokens: int = 16000,
                 effort: str = "high", log: Callable[[str], None] = print,
                 client: Any = None):
        super().__init__(machine, log)
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as err:            # pragma: no cover
                raise RuntimeError(
                    "the LLM controller needs the anthropic package: "
                    "pip install anthropic (or run with ScriptedController)"
                ) from err
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, goal: str) -> ControllerRun:
        self.machine.goal = goal
        run = ControllerRun()
        system = [{"type": "text", "text": SYSTEM_PROMPT,
                   "cache_control": {"type": "ephemeral"}}]
        schemas = anthropic_schemas(self.tools)
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (f"Goal: {goal}\n\n"
                        f"Machine state:\n{self.machine.state()}\n\n"
                        "Begin. Minimise J."),
        }]

        while run.steps < self.max_steps:
            run.steps += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=schemas,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
            )

            if response.stop_reason == "refusal":
                run.stop_reason = "refusal"
                break
            if response.stop_reason == "pause_turn":
                # A server-side pause: hand the same turn back to continue it.
                messages.append({"role": "assistant", "content": response.content})
                continue

            # The full content goes back, thinking blocks included -- editing it
            # breaks the next turn on an adaptive-thinking model.
            messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self.log(f"\n[controller] {block.text.strip()}")

            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                run.stop_reason = response.stop_reason or "end_turn"
                break

            results = []
            for block in calls:
                args = dict(block.input or {})
                out = self._invoke(block.name, args)
                run.tool_calls.append((block.name, args))
                run.transcript.append(out)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out})
                if block.name == "finish":
                    run.finished = True
            # Every result for a turn goes back in ONE user message; splitting
            # them trains the model out of calling tools in parallel.
            messages.append({"role": "user", "content": results})
            if run.finished:
                break

        return run


def make_controller(machine: RenMachine, use_llm: bool | None = None,
                    **kwargs: Any) -> BaseController:
    """Pick a controller. `use_llm=None` means "LLM if credentials exist".

    The fallback is silent by design: an example should run for someone who has
    just cloned the repository, and print what it is doing either way.
    """
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return LLMController(machine, **kwargs) if use_llm else ScriptedController(
        machine, log=kwargs.get("log", print))
