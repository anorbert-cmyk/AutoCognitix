"""Behavioural tests for the `ci-success` gate in `.github/workflows/ci.yml`.

The gate decides whether a commit is mergeable, so it needs a test - but it is
YAML plus shell, not Python, and GitHub Actions cannot be run here. So this
module does the next best thing and tests the real artefacts:

  * it parses the actual workflow file (never a copy), and
  * it extracts the actual `run:` script of the "Check required jobs" step and
    executes it under ``/bin/sh`` with the job results injected as the same
    environment variables the workflow injects.

Nothing is re-implemented, so the truth table below is a property of the
shipped file, not of a paraphrase of it.

The bug being locked down
-------------------------
The job used to carry ``if: always()``. ``always()`` runs a job even when the
WORKFLOW RUN was cancelled - and this workflow sets ``cancel-in-progress: true``
on a ``github.ref`` concurrency group, so every push that lands while a run is
in flight cancels that run. Each superseded run still executed this gate, saw
six ``cancelled`` results, and posted a red X on a commit nobody had a verdict
about. Five such false alerts in one day is what prompted the fix.

The fix is ``if: ${{ !cancelled() }}``: run-scoped, so the gate is skipped -
claiming nothing - exactly when the run itself was cancelled, while still
running (and failing) when a required job fails. Both halves are asserted here,
because getting the second half wrong would let a failing build merge.
"""

import re
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import pytest

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard, not logic
    # Deliberately a hard error rather than pytest.importorskip: this gate
    # decides merges, and a guard that silently skips is not a guard (the same
    # mistake the frozen-vector test made). PyYAML arrives transitively via
    # transformers -> huggingface_hub, both pinned in backend/requirements.txt.
    # If that chain ever breaks, add PyYAML there - do not skip this module.
    raise RuntimeError(
        "PyYAML is required to test the ci-success gate; add it to "
        "backend/requirements.txt rather than skipping this module"
    ) from exc

WORKFLOW_PATH = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"

# Job -> the env var the workflow binds its `result` to.
REQUIRED_JOBS: Dict[str, str] = {
    "backend-lint": "BACKEND_LINT",
    "backend-test": "BACKEND_TEST",
    "frontend-lint": "FRONTEND_LINT",
    "frontend-build": "FRONTEND_BUILD",
    "frontend-test": "FRONTEND_TEST",
    "docker-build": "DOCKER_BUILD",
}

# Every value GitHub can put in `needs.<job>.result`.
JOB_RESULTS = ("success", "failure", "cancelled", "skipped")

GATE_STEP_NAME = "Check required jobs"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ci_success_job(workflow: dict) -> dict:
    return workflow["jobs"]["ci-success"]


@pytest.fixture(scope="module")
def gate_script(ci_success_job: dict) -> str:
    """The literal `run:` body of the gate step, as shipped."""
    for step in ci_success_job["steps"]:
        if step.get("name") == GATE_STEP_NAME:
            return str(step["run"])
    raise AssertionError(f"no {GATE_STEP_NAME!r} step in the ci-success job")


def run_gate(gate_script: str, results: Dict[str, str]) -> int:
    """Execute the shipped gate script under /bin/sh; return its exit code.

    /bin/sh is dash on this image while GitHub runs bash. Passing under the
    stricter POSIX shell is the stronger claim, so it is the one made here.
    """
    env = {"PATH": "/usr/bin:/bin"}
    env.update({REQUIRED_JOBS[job]: result for job, result in results.items()})
    completed = subprocess.run(
        ["/bin/sh", "-c", gate_script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return completed.returncode


def all_success() -> Dict[str, str]:
    return dict.fromkeys(REQUIRED_JOBS, "success")


# ---------------------------------------------------------------------------
# The job-level guard: this is what actually fixes the false red alerts
# ---------------------------------------------------------------------------


class TestJobLevelGuard:
    def test_gate_is_not_guarded_by_always(self, ci_success_job: dict) -> None:
        """`always()` is the bug: it runs the gate on a cancelled run."""
        condition = str(ci_success_job["if"])
        assert "always()" not in condition, (
            "ci-success must not use always(): it runs even when the workflow "
            "run was cancelled, which is what made superseded runs post a red X"
        )

    def test_gate_is_guarded_by_not_cancelled(self, ci_success_job: dict) -> None:
        """`!cancelled()` skips the gate only when the RUN was cancelled."""
        condition = str(ci_success_job["if"])
        normalized = re.sub(r"\s+", "", condition)
        assert normalized == "${{!cancelled()}}", (
            f"unexpected ci-success guard {condition!r}; expected "
            "${{ !cancelled() }} so the gate is skipped on a cancelled run but "
            "still runs - and fails - when a required job fails"
        )

    def test_guard_keeps_a_status_function_so_needs_default_is_overridden(
        self, ci_success_job: dict
    ) -> None:
        """Without a status-check function GitHub re-applies the implicit
        `success()` on `needs:`, so the gate would SKIP whenever a required job
        failed - and branch protection counts a skipped required check as
        satisfied. That failure mode is worse than the bug being fixed.
        """
        condition = str(ci_success_job["if"])
        assert re.search(r"\b(cancelled|failure|success|always)\s*\(\s*\)", condition), (
            f"the ci-success `if:` must contain a status-check function, got {condition!r}"
        )

    def test_concurrency_is_the_documented_cancellation_source(self, workflow: dict) -> None:
        """The comment on the guard points at this block; keep them in sync."""
        concurrency = workflow["concurrency"]
        assert concurrency["cancel-in-progress"] is True
        assert "github.ref" in concurrency["group"]

    def test_all_required_jobs_are_declared_as_needs(self, ci_success_job: dict) -> None:
        """A required job that is not in `needs:` yields an empty result string,
        which the gate would report as a failure with a blank value.
        """
        assert set(REQUIRED_JOBS).issubset(set(ci_success_job["needs"]))

    def test_every_required_result_is_bound_to_an_env_var(self, ci_success_job: dict) -> None:
        """The gate script reads env vars, so the bindings must exist."""
        for step in ci_success_job["steps"]:
            if step.get("name") == GATE_STEP_NAME:
                env = step["env"]
                for job, var in REQUIRED_JOBS.items():
                    assert env[var] == "${{ needs." + job + ".result }}"
                return
        raise AssertionError("gate step not found")


# ---------------------------------------------------------------------------
# The shell condition: executed, not paraphrased
# ---------------------------------------------------------------------------


class TestGateScriptTruthTable:
    def test_all_success_passes(self, gate_script: str) -> None:
        assert run_gate(gate_script, all_success()) == 0

    @pytest.mark.parametrize("job", sorted(REQUIRED_JOBS))
    @pytest.mark.parametrize("result", [r for r in JOB_RESULTS if r != "success"])
    def test_any_single_non_success_fails(self, gate_script: str, job: str, result: str) -> None:
        """Every required job, every non-success value: the gate must fail.

        Includes `cancelled`. Reaching this script means the RUN was not
        cancelled (the job-level `!cancelled()` guard), so a job-scoped
        `cancelled` here is a job that died on its own while the run carried
        on - unverified code, not a superseded run.
        """
        results = all_success()
        results[job] = result
        assert run_gate(gate_script, results) == 1, f"{job}={result} should fail the gate"

    def test_a_failure_is_not_masked_by_sibling_cancellations(self, gate_script: str) -> None:
        """The dangerous combination: a genuine failure plus cancelled siblings.

        If the shell ever started treating `cancelled` as acceptable, this is
        the case that would go green while a real lint failure was on record.
        """
        results = dict.fromkeys(REQUIRED_JOBS, "cancelled")
        results["backend-lint"] = "failure"
        assert run_gate(gate_script, results) == 1

    def test_a_dependency_failure_cascade_still_fails(self, gate_script: str) -> None:
        """backend-test fails -> docker-build (which needs it) reports skipped."""
        results = all_success()
        results["backend-test"] = "failure"
        results["docker-build"] = "skipped"
        assert run_gate(gate_script, results) == 1

    def test_missing_result_is_treated_as_failure(self, gate_script: str) -> None:
        """A `needs:` entry deleted by accident yields an empty string."""
        results = all_success()
        results["docker-build"] = ""
        assert run_gate(gate_script, results) == 1

    def test_the_script_is_posix_shell_clean(self, gate_script: str) -> None:
        """dash rejects bashisms; GitHub's bash would have accepted them."""
        completed = subprocess.run(
            ["/bin/sh", "-n", "-c", gate_script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_exhaustive_sweep(self, gate_script: str) -> None:
        """All 4**6 = 4096 combinations: the gate passes iff every result is
        'success'. Slow-ish (~4k tiny shells) but it is the whole claim.
        """
        failures: List[str] = []
        for combo in product(JOB_RESULTS, repeat=len(REQUIRED_JOBS)):
            results = dict(zip(sorted(REQUIRED_JOBS), combo))
            expected = 0 if set(combo) == {"success"} else 1
            actual = run_gate(gate_script, results)
            if actual != expected:
                failures.append(f"{results} -> {actual}, expected {expected}")
        assert not failures, failures[:10]


if __name__ == "__main__":  # pragma: no cover - manual truth-table printer
    script = None
    doc = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    for candidate in doc["jobs"]["ci-success"]["steps"]:
        if candidate.get("name") == GATE_STEP_NAME:
            script = str(candidate["run"])
    assert script is not None
    print(f"job-level if: {doc['jobs']['ci-success']['if']}")
    print(f"{'scenario':<58} {'exit':>4}  verdict")
    print("-" * 78)
    scenarios = [("all success", all_success())]
    for value in JOB_RESULTS:
        scenarios.append((f"all {value}", dict.fromkeys(REQUIRED_JOBS, value)))
    for name in sorted(REQUIRED_JOBS):
        for value in JOB_RESULTS:
            combo = all_success()
            combo[name] = value
            scenarios.append((f"{name}={value}, rest success", combo))
    mixed = dict.fromkeys(REQUIRED_JOBS, "cancelled")
    mixed["backend-lint"] = "failure"
    scenarios.append(("backend-lint=failure, rest cancelled", mixed))
    for label, combo in scenarios:
        code = run_gate(script, combo)
        print(f"{label:<58} {code:>4}  {'PASS' if code == 0 else 'FAIL'}")
    sys.exit(0)
