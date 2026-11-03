"""Regression tests for the process-isolated grader.

Context (root-caused 2026-09-23 with py-spy): the RolloutManager thread was stuck
inside sympy's `solve`, computing a reward. `math_verify`'s timeout uses
`signal.alarm`, which only works on the main thread, so `reward.py` zeroed the
timeout off-main-thread -- disabling it exactly where the training rollout needs
it. One pathological answer then spins forever and the run looks like an SGLang
silent wedge. `tasks/math/grader_pool` runs the call in a worker process instead.

Run directly (`python tests/test_grader_pool.py`) or under pytest. Both work:
the pool uses the spawn context, whose worker entry point lives in an importable
module, so it does not depend on how this test file is invoked.
"""

import os
import sys
import threading
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

CANARY_PRED, CANARY_GOLD = "540", "\\textbf{(540)}"


def test_off_main_thread_grading_matches_main_thread():
    """The pool must reproduce the in-process grader's answers exactly."""
    from tasks.math.reward import grade_answer

    expected = grade_answer(CANARY_PRED, CANARY_GOLD)  # main thread

    result = {}
    started = time.time()

    def work():
        result["same"] = grade_answer(CANARY_PRED, CANARY_GOLD)
        result["different"] = grade_answer("541", CANARY_GOLD)

    th = threading.Thread(target=work)
    th.start()
    th.join(timeout=180)
    assert not th.is_alive(), "off-main-thread grading did not return"

    assert expected is True, "main-thread canary must be True"
    assert result["same"] is True, (
        "off-main-thread grading returned False for the canary -- the worker pool "
        "is not reproducing the grader (spawn/IPC failure would look like this)"
    )
    assert result["different"] is False, "a different answer must not grade as correct"
    print(f"  ok: off-main-thread canary=True, mismatch=False ({time.time()-started:.1f}s)")


def test_worker_is_killed_when_it_overruns():
    """An overrun must raise (and replace the worker), never hang the caller.

    This is the property that actually fixes the stall: a worker stuck in
    unbounded sympy work cannot be interrupted by any in-process timeout, so the
    PARENT has to kill it. The worker is made to sleep past its budget via the
    test-only OPSD_GRADER_TEST_SLEEP_S hook (spawn children inherit env).
    """
    os.environ["OPSD_GRADER_TEST_SLEEP_S"] = "30"
    os.environ["OPSD_GRADER_KILL_GRACE_S"] = "2"
    try:
        from tasks.math import grader_pool
        from tasks.math.grader_pool import GraderPool, GraderTimeout, reset_for_tests

        reset_for_tests()
        grader_pool._KILL_GRACE_S = 2.0  # read at call time
        pool = GraderPool(workers=1, timeout_s=1.0)
        started = time.time()
        try:
            pool.grade(CANARY_PRED, CANARY_GOLD, timeout_s=1.0)
            raise AssertionError("a 1s budget against a 30s worker should not complete")
        except GraderTimeout:
            pass
        elapsed = time.time() - started
        assert elapsed < 30, f"kill path took {elapsed:.0f}s -- it waited instead of killing"
        # the pool must still be usable afterwards (worker replaced, not poisoned)
        assert any(w.alive() for w in pool._workers)
        pool.close()
        print(f"  ok: overrun killed the worker after {elapsed:.1f}s and the pool recovered")
    finally:
        os.environ.pop("OPSD_GRADER_TEST_SLEEP_S", None)
        os.environ.pop("OPSD_GRADER_KILL_GRACE_S", None)


if __name__ == "__main__":
    print("test_grader_pool:")
    test_off_main_thread_grading_matches_main_thread()
    test_worker_is_killed_when_it_overruns()
    print("ALL PASS")
