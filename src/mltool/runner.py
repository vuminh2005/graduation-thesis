"""``mltool run``: the whole pipeline in one command, re-running only what is stale.

Every decision comes from the freshness ``status`` reports (``_phase_states``);
there is no separate list of staleness checks here. A real run re-reads that
state before each step, so a step sees what the steps before it just rewrote. A
dry run reads it once: everything downstream of a missing or stale step is
already shown stale by ``status``'s propagation, so the plan it prints is the
one a real run takes, except where a re-run step happens to rewrite
byte-identical artifacts and the real run can then skip what follows.

``finalize`` is never re-run automatically. It evaluates the test split, and
re-running it after every config edit would turn the test score into something
the user iterates on, i.e. select on test data. A stale final model stops the
run until the user passes ``--refinalize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

from mltool.config import MLToolConfig
from mltool.registry import list_versions
from mltool.reporting import _phase_states

STEPS = ["validate", "prepare", "features", "plan", "train", "tune", "finalize", "register"]
ALWAYS = {"validate", "plan"}

STOP_BEFORE_FINALIZE = (
    "the final model is stale ({reason}). finalize evaluates the test split, and "
    "re-running it after changes risks selecting a configuration on test data. "
    'Inspect the new results first ("mltool tuning-leaderboard"); if re-evaluating '
    "the test split is intentional, pass --refinalize"
)


@dataclass
class Decision:
    step: str
    action: str  # "run" | "skip" | "stop"
    reason: str
    force: bool = False  # finalize --force (only with --refinalize)


@dataclass
class StepOutcome:
    decision: Decision
    exit_code: int | None = None
    seconds: float | None = None


@dataclass
class RunReport:
    outcomes: list[StepOutcome] = field(default_factory=list)
    stopped_at: str | None = None
    failed_step: str | None = None
    exit_code: int = 0

    def details(self) -> dict[str, Any]:
        return {
            "steps": [
                {"step": o.decision.step, "action": o.decision.action, "reason": o.decision.reason,
                 "exit_code": o.exit_code, "seconds": o.seconds}
                for o in self.outcomes
            ],
            "stopped_at": self.stopped_at,
            "failed_step": self.failed_step,
        }


def _reason(state: str, note: str) -> str:
    return f"stale: {note}" if state == "stale" and note else state


def decide(step: str, config: MLToolConfig, states: dict[str, tuple[str, str]], *,
           refinalize: bool) -> Decision:
    """One step's decision from ``status``'s state of that step."""
    if step == "validate":
        if not config.validation.enabled:
            return Decision(step, "skip", "validation.enabled is false (prepare still validates the data)")
        return Decision(step, "run", "always runs")
    if step == "plan":
        return Decision(step, "run", "always runs")
    state, note = states[step]
    if step == "finalize":
        if state == "missing":
            return Decision(step, "run", "missing")
        if state == "fresh":
            return Decision(step, "skip", "fresh")
        if refinalize:
            return Decision(step, "run", f"{_reason(state, note)}; --refinalize: the test split is "
                                         "evaluated again", force=True)
        return Decision(step, "stop", STOP_BEFORE_FINALIZE.format(reason=note or state))
    if state == "fresh":
        return Decision(step, "skip", f"fresh ({note})" if step == "register" and note else "fresh")
    return Decision(step, "run", _reason(state, note))


def plan_decisions(config: MLToolConfig, *, until: str | None, refinalize: bool) -> list[Decision]:
    """``--dry-run``: every step's decision from one reading of the current state."""
    states = _phase_states(config)
    decisions = []
    for step in STEPS:
        decision = decide(step, config, states, refinalize=refinalize)
        decisions.append(decision)
        if decision.action == "stop" or step == until:
            break
    return decisions


def execute(
    config_path: Path,
    config: MLToolConfig,
    step_functions: dict[str, Callable[..., int]],
    *,
    until: str | None,
    refinalize: bool,
    announce: Callable[[str], None] = print,
) -> RunReport:
    report = RunReport()
    for index, step in enumerate(STEPS, start=1):
        decision = decide(step, config, _phase_states(config), refinalize=refinalize)
        outcome = StepOutcome(decision)
        report.outcomes.append(outcome)
        announce(f"\n[run {index}/{len(STEPS)}] {step}: {decision.action} ({decision.reason})")
        if decision.action == "stop":
            report.stopped_at, report.exit_code = step, 2
            return report
        if decision.action == "run":
            started = time.perf_counter()
            kwargs = {"force": True} if decision.force else {}
            outcome.exit_code = step_functions[step](config_path, **kwargs)
            outcome.seconds = float(time.perf_counter() - started)
            if outcome.exit_code != 0:
                report.failed_step, report.exit_code = step, outcome.exit_code
                return report
        if step == until:
            break
    return report


def render_decisions(decisions: list[Decision]) -> str:
    lines = ["MLTool run (dry run: nothing is executed)", ""]
    lines.extend(f"  {d.step:<9} {d.action:<5} {d.reason}" for d in decisions)
    if decisions and decisions[-1].action == "stop":
        lines.extend(["", f"A real run would stop before {decisions[-1].step}."])
    return "\n".join(lines)


def render_summary(config: MLToolConfig, report: RunReport, final: Any | None) -> str:
    lines = ["", "MLTool run summary", ""]
    for outcome in report.outcomes:
        d = outcome.decision
        took = f"{outcome.seconds:.1f}s" if outcome.seconds is not None else "-"
        status = "" if outcome.exit_code in (None, 0) else f"  exit {outcome.exit_code}"
        lines.append(f"  {d.step:<9} {d.action:<5} {took:>8}{status}  {d.reason if d.action != 'stop' else ''}")
    if report.failed_step:
        lines.extend(["", f"Result: FAILED at {report.failed_step} (exit {report.exit_code})"])
        return "\n".join(lines)
    if report.stopped_at:
        lines.extend(["", f"Stopped before {report.stopped_at}: {report.outcomes[-1].decision.reason}"])
    root = config.config_path.parent
    if final is not None:
        result = final.result
        lines.extend([
            "",
            f"Selected: {result['candidate_id']} ({result['feature_set']} x {result['model']['name']} "
            f"[{result['model']['family']}])",
            "Test metrics: " + ", ".join(f"{k}={v:.6f}" for k, v in result["metrics"].items()),
        ])
        if final.warning:
            lines.append(f"  ! the final model is stale: {final.warning}")
    versions = list_versions(root)
    lines.append(f"Registry version: {versions[-1] if versions else 'none'}")
    lines.extend([
        "",
        "Next",
        "  mltool best",
        "  mltool score --input <rows.csv> --output <predictions.csv>",
        "  mlflow ui --backend-store-uri .mltool/mlflow",
        "",
        f"Result: {'STOPPED' if report.stopped_at else 'DONE'}",
    ])
    return "\n".join(lines)
