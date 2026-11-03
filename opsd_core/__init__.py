"""L0 core primitives — task-agnostic AND method-agnostic.

This package is the single source for everything that was previously copied
across the method family:

    batch       right-padded microbatch packing + the shared-completion contract
    chunking    contiguous near-equal span partitioning
    alignment   teacher/student completion-span equality assertions
    text        chat-template user-turn tokenisation
    weights     per-token credit-weight normalisation (mean 1, clamped, stopgrad)
    teacher     frozen initial-policy teacher: load / forward / slice
    scoring     target-span-only lm_head log-prob scoring
    writer      one-rank writer election for shared per-run state files

Dependency rule (enforced by convention, checked in tests/test_opsd_core.py):
    opsd_core MUST NOT import from methods / tasks / math_opsd / alfworld_hpi /
    alfworld_hpi_v2 / code_grpo / frameworks. The dependency arrow points
    strictly downward: entry (registry/launcher) -> tasks -> methods -> opsd_core.

Old import paths are preserved as re-export shims so that vendored
``frameworks/slime-ascend`` and the running task lines keep resolving:
    math_opsd.opsd_batch.*        -> opsd_core.batch.*
    math_opsd.writer_election.*   -> opsd_core.writer.*
    math_opsd.opsd_loss.{_get_teacher,teacher_forward_completion_logits,
                          slice_response_logits}   -> opsd_core.teacher.*
    math_opsd.hpi.{split_chunks,score_view_logprobs,_forward_last_hidden}
                                                  -> opsd_core.{chunking,scoring}

The 2026-10-15 layout refactor moved the task and method
modules into ``tasks/`` and ``methods/`` and turned ``math_opsd`` /
``alfworld_hpi`` / ``alfworld_hpi_v2`` / ``code_grpo`` into alias shims
(``opsd_core/shim.py``). The arrows above are unchanged by that move.
"""
