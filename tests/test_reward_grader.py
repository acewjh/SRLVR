"""Regression tests for the official-grader loading in tasks.math.reward.

Why this file exists (2026-09-18 audit): ``_load_oracle_functions`` AST-execs the
oracle's ``grade_answer`` with a namespace that only contained ``re``. Inside the
function body ``parse``/``verify`` therefore raised NameError, which the oracle's
own ``except Exception`` swallowed -- every grade in every training run and every
formal eval cell silently degraded to

    predicted.replace("$","").replace(" ","").lower() == gold...

Two consequences, both measured:
  * the whole 2026-09-17/18 formal eval campaign is invalid (2520/2520 recorded
    rewards reproduce that string equality exactly), and
  * AIME24 dev labels are LaTeX-wrapped (10/30 are ``\\textbf{...}``), which the
    string comparison can never match -- so a third of AIME24 was unwinnable for
    every checkpoint, which is what produced the "AIME24-specific regression".

Run:  /opt/conda/envs/cua-rl/bin/python tests/test_reward_grader.py
      (or: python -m pytest tests/test_reward_grader.py -q)
Needs the harness env (cua-rl): math_verify + latex2sympy2_extended.
"""

import sys
import threading
import traceback

sys.path.insert(0, "/workspace/cua/repos/opsd-slime")

from tasks.math import reward  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
    if not condition:
        FAILURES.append(name)


# --- A. the exact failure that invalidated the campaign ----------------------
def test_grader_performs_equivalence_not_string_equality() -> None:
    gold = "\\textbf{(540)}"
    check("A1 grade('540', '\\textbf{(540)}') is True", reward.grade_answer("540", gold) is True,
          "AIME24 dev labels are textbf-wrapped; only real math_verify accepts them")
    check("A2 grade('33', '033') is True", reward.grade_answer("33", "033") is True)
    check("A3 grade('1/576', '\\frac{1}{576}') is True",
          reward.grade_answer("1/576", "\\frac{1}{576}") is True)
    check("A4 grade('541', '540') is False", reward.grade_answer("541", "540") is False,
          "the grader must still reject genuinely wrong answers")
    check("A5 grade(None, '540') is False", reward.grade_answer(None, "540") is False)


# --- B. the grader must survive slime's threaded call sites ------------------
def test_grader_is_thread_safe() -> None:
    """math_verify's parse uses signal.alarm and raises in non-main threads.

    Without the wrapper that exception hits the oracle's broad except and the
    grader silently degrades again -- the very failure this file guards.
    """
    box: dict = {}

    def worker() -> None:
        try:
            box["r"] = reward.grade_answer("540", "\\textbf{(540)}")
        except Exception as exc:  # noqa: BLE001
            box["r"] = f"EXC {type(exc).__name__}: {exc}"

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    check("B1 grade_answer is correct inside a worker thread", box.get("r") is True,
          f"got {box.get('r')!r}")


# --- C. degraded graders must be loud, never silent --------------------------
def test_loader_refuses_to_degrade_silently() -> None:
    class _Empty:  # stands in for math_verify's parser entry points
        def __getattr__(self, item: str):
            raise AttributeError(item)

    def broken_import():
        raise ModuleNotFoundError("No module named 'latex2sympy2_extended'")

    try:
        reward._load_oracle_functions(math_verify_import=broken_import)
        check("C1 missing math_verify raises instead of degrading", False, "no exception raised")
    except RuntimeError as exc:
        check("C1 missing math_verify raises instead of degrading",
              "math_verify" in str(exc), f"raised {type(exc).__name__}")

    # A grader that cannot do equivalence (string-compare-only) must also be
    # rejected: it is exactly the silent-degradation failure mode.
    real_parse, real_verify = reward._import_math_verify()

    def string_only_import():
        return (lambda text, **kw: text), (
            lambda gold, target, **kw: str(gold).replace("$", "").lower().strip()
            == str(target).replace("$", "").lower().strip()
        )

    try:
        reward._load_oracle_functions(math_verify_import=string_only_import)
        check("C2 string-compare-only grader is rejected", False, "no exception raised")
    except RuntimeError:
        check("C2 string-compare-only grader is rejected", True)
    assert real_parse and real_verify  # the real symbols still load


def test_extract_boxed_answer_matches_oracle_semantics() -> None:
    check("D1 last boxed answer wins",
          reward.extract_boxed_answer("\\boxed{1} then \\boxed{2}") == "2")
    check("D2 brace matching handles nesting",
          reward.extract_boxed_answer("\\boxed{\\frac{25}{8}}") == "\\frac{25}{8}")
    check("D3 trailing display math is not consumed",
          reward.extract_boxed_answer("we get $$ \\boxed{540} $$") == "540")
    check("D4 no boxed -> None", reward.extract_boxed_answer("answer is 540") is None)


# --- E. fail closed: infrastructure failure must abort, never degrade --------
def test_production_path_is_fail_closed() -> None:
    """A broken grader must raise, not quietly answer with string comparison."""
    real_parse, real_verify = reward._import_math_verify()
    state = {"broken": False}

    def flaky_parse(text, **kwargs):
        if state["broken"]:
            raise RuntimeError("simulated infrastructure failure")
        return real_parse(text, **kwargs)

    grade = reward._make_grade_answer(flaky_parse, real_verify)
    check("E1 healthy grader answers correctly", grade("540", "\\textbf{(540)}") is True)

    state["broken"] = True
    try:
        result = grade("540", "\\textbf{(540)}")
        check("E2 infrastructure failure raises instead of downgrading", False,
              f"returned {result!r}")
    except reward.GraderUnavailable:
        check("E2 infrastructure failure raises instead of downgrading", True)

    # the old behaviour: the same broken call returned a *wrong* answer silently
    check("E3 the debug string comparator silently answers False here",
          reward.string_equal("540", "\\textbf{(540)}") is False,
          "this is precisely what the campaign was scored with")


def test_call_time_canary_detects_mid_run_degradation() -> None:
    """Reconstruct the 2026-09 grader: no real parsing, exact-string comparison."""

    def no_op_parse(text, **kwargs):
        return text

    def string_verify(gold, target, **kwargs):
        return (
            str(gold).replace("$", "").lower().strip()
            == str(target).replace("$", "").lower().strip()
        )

    grade = reward._make_grade_answer(no_op_parse, string_verify)
    try:
        reward._assert_canary(grade)
        check("F1 call-time canary rejects a grader that degraded mid-run", False,
              "no exception raised")
    except reward.GraderUnavailable:
        check("F1 call-time canary rejects a grader that degraded mid-run", True)


def test_unparseable_answers_are_false_not_errors() -> None:
    """Fail-closed must not over-trigger: bad model output is a legitimate 0."""
    check("G1 empty boxed content -> False", reward.grade_answer("", "540") is False)
    check("G2 free text -> False", reward.grade_answer("the answer is forty", "540") is False)
    check("G3 None -> False", reward.grade_answer(None, "540") is False)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except Exception:  # noqa: BLE001 - a raising test is a failing test
            traceback.print_exc()
            FAILURES.append(f"{fn.__name__} raised")
    print(f"\n{'=' * 60}\n{len(tests)} test functions, {len(FAILURES)} failure(s)")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
