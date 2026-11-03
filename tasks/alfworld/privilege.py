"""Future-HPI: Full-vs-Control privileged context, and the action span.

The scientific definition (v2)
-----------------------------
Future-HPI does **not** add a distillation objective, a KL term, a value head or a
shaping hyper-parameter. It only *re-allocates* credit that GRPO already assigned,
using information the student did not have.

For turn ``t`` the student saw ``h_t`` (task + bounded history + current
observation + admissible actions) and produced ``y_t = think_t + action_t``.

After the episode ends, a **frozen** teacher (the initial policy) is run twice on
the *same* ``y_t``:

* **Control** -- ``h_t`` + the terminal outcome ``R`` + an explicit
  ``(not available)`` future trace.
* **Full** -- byte-identical to Control except the Future interaction trace body.

Then, over the ``K`` tokens of the **action span only**::

    e_t = (1/K) * sum_k [ log q_full(action_k) - log q_control(action_k) ]

A MEAN, not a sum: otherwise a long action would have a larger ``|e|`` purely by
virtue of being long. ``r_t = exp(e_t)`` is then the geometric-mean token density
ratio ``q_full(action_t) / q_control(action_t)`` -- it needs no tanh, no sigmoid,
no scale, no clip.

Index convention (unchanged from v1)
------------------------------------
``transition i = (action_i, observation_{i+1})``. With ``gap = 1`` the Full trace
for turn ``t`` contains transitions ``i >= t + 1``: the immediate result of ``a_t``
(transition ``t``) is withheld, because ``o_{t+1}`` often narrates the action
verbatim ("You pick up the mug ..."). ``R`` is present in BOTH views and identical.

What the future trace may contain
---------------------------------
Only ``executed action`` and ``environment observation`` of later turns. Never a
gold/expert/reference action or trajectory, and never the current turn's own
action. The builder physically cannot read ``generated_text`` or ``think_text``
of later turns. Reasoning markers that are part of the *recorded* action or
observation text -- for example a rejected-action notice that echoes the model's
raw output back -- are preserved verbatim: this module enforces transition
provenance, not the content of the record.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import os

from . import config

__all__ = [
    "HPI_BLOCK_TEMPLATE",
    "assert_transition_provenance",
    "matched_sibling",
    "action_token_span_from_ids",
    "NO_FUTURE_PLACEHOLDER",
    "action_token_span",
    "build_full_control_ids",
    "render_hpi_block",
    "render_trajectory_block",
    "TRAJECTORY_BLOCK_TEMPLATE",
    "span_from_ids_for",
    "tag_token_span",
    "tag_token_span_from_ids",
    "terminal_outcome_word",
    "think_token_span_from_ids",
]

NO_FUTURE_PLACEHOLDER = "(not available)"

#: SRLVR-v4 template. Unlike the per-turn HPI block this one describes the WHOLE
#: executed episode -- every action and its resulting observation, plus the
#: outcome. It is therefore rendered once per trajectory and reused by every
#: turn's teacher hidden state, instead of rebuilding a future-only view per
#: token. The "not part of the conversation" clause is load-bearing: the trace
#: contains the actions themselves, so it must never be mistaken for student
#: context.
TRAJECTORY_BLOCK_TEMPLATE = (
    "<privileged_hindsight>\n"
    "This information is available only during training. It is the complete executed "
    "episode, including every action, its resulting observation, and the final outcome; "
    "none of it is part of the conversation.\n"
    "\n"
    "Final episode outcome:\n"
    "{outcome}\n"
    "\n"
    "Complete interaction trace:\n"
    "{future}\n"
    "</privileged_hindsight>"
)

#: One template for both views. Only ``{future}`` differs -- and for the control
#: view it is the placeholder, so an empty future can never leak a scaffold
#: asymmetry either.
HPI_BLOCK_TEMPLATE = (
    "<privileged_hindsight>\n"
    "This information is available only during training. It was observed AFTER the "
    "action being evaluated and is not part of the conversation.\n"
    "\n"
    "Final episode outcome:\n"
    "{outcome}\n"
    "\n"
    "Future interaction trace:\n"
    "{future}\n"
    "</privileged_hindsight>"
)


def terminal_outcome_word(success: bool) -> str:
    return "SUCCESS" if success else "FAILURE"


#: Chat-template artefacts that can land inside a trajectory's stored action or
#: observation text -- e.g. a turn cut off mid-generation, or a rejected-action
#: notice that echoed the model's raw output back. They are template noise, NOT
#: reasoning, so they are stripped from the trace body before rendering.
#:
#: Reasoning markers (`<think>`, `</think>`, `reasoning_content`) are NOT
#: stripped and are NOT rejected. A recorded observation can legitimately echo
#: the model's raw output (the rejected-action notice does exactly that), and the
#: future trace must preserve the recorded trajectory verbatim. What this module
#: still enforces is provenance: each rendered action/observation must match the
#: turn it is labelled with (see ``assert_transition_provenance``).
_TEMPLATE_ARTIFACTS = ("<|im_start|>", "<|im_end|>")


def _sanitize_trace_text(text: str) -> str:
    """Remove chat-template artefacts from one trace field (see above)."""
    if not text:
        return text
    out = text
    for token in _TEMPLATE_ARTIFACTS:
        out = out.replace(token, "")
    return out


def _render_future(trajectory, turn_idx: int, future_gap: int) -> str:
    """Transitions ``i >= turn_idx + future_gap``, as text. Empty when none left.

    Serialised as the transition pair, NOT as "state then action"::

        [Future transition i]
        Executed action:                action_i
        Resulting environment observation:  next_observation_i   (= o_{i+1})

    The distinction is load-bearing. An earlier version rendered
    ``trajectory.observation(i)`` -- the observation the agent was looking at
    *before* acting -- which for ``i = t + 1`` is ``o_{t+1}``: exactly the direct
    result of ``a_t`` that ``future_gap = 1`` is supposed to withhold. Rendering the
    *result* instead means the first thing the trace shows for turn ``t`` is
    ``(action_{t+1}, o_{t+2})``, so ``o_{t+1}`` never appears.
    """
    lines: list[str] = []
    order = 0
    for index, action, next_observation in trajectory.iter_tail_transitions(
        turn_idx + future_gap
    ):
        order += 1
        lines.append(f"[Future transition {index}]")
        lines.append(f"Executed action: {_sanitize_trace_text(action)}")
        lines.append(
            "Resulting environment observation: "
            f"{_sanitize_trace_text(next_observation).strip()}"
        )
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    # No separate "final observation" line.
    #
    # It used to be appended unconditionally, which broke two things at once: it
    # duplicated the last transition's own result, and for the FINAL turn (where
    # gap=1 leaves no allowed transition) it printed o_{t+1} -- the direct result of
    # the evaluated action -- making Full differ from Control on exactly the turn
    # that is supposed to be unscorable. Every transition already prints its
    # resulting observation, and the last one is the terminal observation, so
    # nothing is lost by dropping it.
    return "\n".join(lines)


def render_compact_future(transitions) -> str:
    """Render already HSI-eligible transitions, losslessly and deterministically.

    Deltas reference the preceding emitted observation, within this fragment.
    Line terminators are retained; no semantic normalization is performed.
    """
    source = [(i, _sanitize_trace_text(a), _sanitize_trace_text(o)) for i, a, o in transitions]
    groups = []
    for i, action, observation in source:
        if groups and i == groups[-1][1] + 1 and (action, observation) == groups[-1][2:]:
            first, _, a, o = groups[-1]
            groups[-1] = (first, i, a, o)
        else:
            groups.append((i, i, action, observation))
    output, expanded = [], []
    reference = reference_turn = None
    for first, last, action, observation in groups:
        label = str(first) if first == last else f"{first}-{last}"
        lines = [f"[Future transition {label}]", f"Executed action: {action}"]
        if first != last:
            lines.append(f"This action and feedback repeated consecutively {last-first+1} times.")
        full = "Resulting environment observation: " + observation
        rendered = full
        if reference is not None:
            before, after = reference.splitlines(keepends=True), observation.splitlines(keepends=True)
            edits, restored, cursor = [], [], 0
            for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
                if tag != "equal":
                    edits.append(dict(operation=tag, reference_lines=[i, j], removed=before[i:j], added=after[k:l]))
                    restored.extend(before[cursor:i])
                    restored.extend(after[k:l])
                    cursor = j
            restored.extend(before[cursor:])
            assert "".join(restored) == observation, "observation delta is not reversible"
            delta = (f"Resulting environment observation: line delta from turn {reference_turn}; "
                     "zero-based half-open reference line ranges; retain all unchanged lines.\n"
                     + json.dumps(edits, ensure_ascii=False, separators=(",", ":")))
            if len(delta) < len(full):
                rendered = delta
        lines.append(rendered)
        output.append("\n".join(lines))
        reference, reference_turn = observation, last
        expanded.extend((i, action, observation) for i in range(first, last+1))
    assert expanded == source, "compression changed transition coverage/order"
    return "\n\n".join(output)


def assert_transition_provenance(trajectory, turn_idx: int, future_gap: int, block: str) -> None:
    """Structural gate: the trace may only contain correctly-labelled transitions.

    Parsed by the ``[Future transition i]`` labels rather than by substring search.
    Substring matching is unusable here for the same reason it was unusable for the
    v1 leak check: ALFWorld repeats text constantly (``"Nothing happens."`` appears
    after every failed action), so a later observation legitimately contains an
    earlier one and a naive ``in`` test fires on correct output.

    Two things are verified:

    * **window** -- every rendered index is ``>= turn_idx + future_gap``;
    * **provenance** -- for each rendered ``i``, the observation printed is
      ``trajectory.turns[i].next_observation`` (the RESULT of ``a_i``), not
      ``trajectory.observation(i)`` (the state before it). That second check is
      what catches the off-by-one: the buggy rendering printed
      ``observation(i) == next_observation(i-1)``, which this rejects.
    """
    first_allowed = turn_idx + future_gap
    allowed = {i for i, _a, _o in trajectory.iter_tail_transitions(first_allowed)}

    # Observations are multi-line (the inadmissible-action notice embeds the room
    # text), so continuation lines must be folded back in. Parsing line-by-line and
    # keeping only the first line silently truncated every such observation and made
    # this gate fire on correct output.
    rendered: list[tuple[int, str | None, str | None]] = []
    current: list | None = None
    field: str | None = None
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("[Future transition ") and stripped.endswith("]"):
            if current is not None:
                rendered.append(tuple(current))
            index = int(stripped[len("[Future transition ") : -1])
            current = [index, None, None]
            field = None
        elif current is not None and stripped.startswith("Executed action:"):
            current[1] = stripped[len("Executed action:") :].strip()
            field = "action"
        elif current is not None and stripped.startswith("Resulting environment observation:"):
            current[2] = stripped[len("Resulting environment observation:") :].strip()
            field = "observation"
        elif stripped.startswith("</privileged_hindsight>"):
            # end of the block: stop folding continuation lines
            field = None
        elif current is not None and field is not None and stripped:
            current[1 if field == "action" else 2] = (
                (current[1 if field == "action" else 2] or "") + "\n" + stripped
            )
    if current is not None:
        rendered.append(tuple(current))

    for index, action, observation in rendered:
        if index < first_allowed:
            raise AssertionError(
                f"privilege for turn {turn_idx} renders transition {index}, inside the gap "
                f"window (gap={future_gap})"
            )
        turn = trajectory.turns[index]
        # Compare the rendered representation: the trace renderer removes chat
        # template artefacts from both fields without changing the trajectory.
        expected_action = _sanitize_trace_text(turn.action_text).strip()
        if action != expected_action:
            raise AssertionError(
                f"transition {index} prints action {action!r} but the trajectory recorded "
                f"{turn.action_text!r}"
            )
        expected_obs = "\n".join(
            part.strip()
            for part in _sanitize_trace_text(turn.next_observation or "").strip().splitlines()
            if part.strip()
        )
        if observation != expected_obs:
            raise AssertionError(
                f"transition {index} prints observation {observation!r}, which is not its "
                f"resulting observation {turn.next_observation!r} -- the trace is showing the "
                "state BEFORE the action, which for the first rendered step is the withheld "
                "result of the evaluated action"
            )

    missing = allowed - {index for index, _a, _o in rendered}
    if missing:
        raise AssertionError(
            f"privilege for turn {turn_idx} is missing transitions {sorted(missing)}"
        )


def render_hpi_block(
    trajectory,
    turn_idx: int,
    *,
    with_future: bool,
    future_gap: int,
    trace_source=None,
    outcome_text: str | None = None,
    compact_future: bool = False,
) -> str:
    """Render the privileged block for one view.

    ``with_future=False`` is the Control view: the identical block with the trace
    body replaced by the placeholder. Everything else -- disclaimer, outcome,
    headers, whitespace -- is byte-identical between the two views.

    ``outcome_text`` overrides the single-word outcome with a verifier-derived
    descriptor (task book section 4). It defaults to ``None``, which reproduces the
    historical ``SUCCESS``/``FAILURE`` word bit-for-bit, so every arm predating the
    HSI V2 protocol is unchanged. The descriptor is still plain language in the
    prompt: no reward number is ever rendered here.
    """
    # `trace_source` (contrast diagnostic only) renders the SAME scaffold with a
    # trace body taken from a sibling episode. Outcome, h_t, admissible actions and
    # the target all stay the evaluated trajectory's, so
    # `|e_shuffled|` vs `|e_full|` isolates "the teacher reacts to appended text at
    # all" from "the teacher reacts to THIS future".
    source = trace_source if trace_source is not None else trajectory
    if with_future and compact_future:
        future = render_compact_future(source.iter_tail_transitions(turn_idx + future_gap))
    else:
        future = _render_future(source, turn_idx, future_gap) if with_future else ""
    text = HPI_BLOCK_TEMPLATE.format(
        outcome=(
            outcome_text
            if outcome_text is not None
            else terminal_outcome_word(trajectory.success)
        ),
        future=future.strip() if future.strip() else NO_FUTURE_PLACEHOLDER,
    )
    # Reasoning content is allowed in the trace by design: it may be part of the
    # recorded action/observation text (e.g. the rejected-action notice), and the
    # block must not rewrite the original trajectory. Only chat-template control
    # tokens are rejected, as a seam/regression guard -- they are template noise,
    # not record content, and ``_sanitize_trace_text`` strips them from the body.
    lowered = text.lower()
    for token in _TEMPLATE_ARTIFACTS:
        if token in lowered:
            raise AssertionError(
                f"privileged block contains chat-template control token {token!r}"
            )
    if with_future and not compact_future:
        # Validate against the trajectory the TRACE came from. When `trace_source`
        # is set (shuffled contrast) the body is a sibling episode's, so checking it
        # against the evaluated trajectory would compare two different rollouts and
        # always fail. The scaffold and outcome still come from `trajectory`.
        assert_transition_provenance(source, turn_idx, future_gap, text)
    return text


def render_trajectory_block(
    trajectory,
    *,
    outcome_text: str | None = None,
    compact_future: bool = False,
    transitions: list[tuple[int, str, str]] | None = None,
) -> str:
    """SRLVR-v4 privilege: the WHOLE episode, rendered once per trajectory.

    This is deliberately not ``render_hpi_block(turn_idx=0, future_gap=0)``:
    that function's template says the trace was observed *after* the evaluated
    action, which is false here -- the trace is the episode itself. The trace
    body is still produced by the same renderers, so provenance, sanitisation
    and the compact (lossless) representation are identical to v2/v3.

    ``compact_future`` uses :func:`render_compact_future`, which folds repeated
    transitions and emits reversible line deltas for observations. It changes
    only the text the teacher reads, never the trajectory or the student rows.

    ``transitions`` lets the caller pass an already-filtered transition list --
    SRLVR-v4 uses it to withhold a random few turns from the hindsight trace so
    the teacher cannot simply recite the episode it is meant to teach from. The
    indices in the rendered text keep their original values, so a withheld turn
    leaves a visible gap rather than silently renumbering the trace.
    """
    if outcome_text is None:
        from .hsi_shared import render_outcome_descriptor

        outcome_text = render_outcome_descriptor(trajectory.success)

    if transitions is None:
        transitions = list(trajectory.iter_tail_transitions(0))
    if compact_future:
        future = render_compact_future(transitions)
    elif transitions is not None and len(transitions) != len(list(trajectory.iter_tail_transitions(0))):
        raise ValueError(
            "render_trajectory_block cannot render a filtered transition list in "
            "the non-compact form; use compact_future=True"
        )
    else:
        future = _render_future(trajectory, 0, 0)
    text = TRAJECTORY_BLOCK_TEMPLATE.format(
        outcome=outcome_text,
        future=future.strip() if future.strip() else NO_FUTURE_PLACEHOLDER,
    )
    lowered = text.lower()
    for token in _TEMPLATE_ARTIFACTS:
        if token in lowered:
            raise AssertionError(
                f"privileged trajectory block contains chat-template control token {token!r}"
            )
    if not compact_future:
        # Same structural gate as the per-turn Full view: every rendered index
        # must be a real transition and must print its RESULTING observation.
        assert_transition_provenance(trajectory, 0, 0, text)
    return text


def matched_sibling(trajectory, pool, *, remaining: int):
    """Deterministic matched sibling for the Shuffled view (spec section 2).

    Priority: (1) same task category, (2) same terminal outcome, (3) closest
    remaining horizon. If nothing satisfies both category and outcome, relax to
    same outcome + closest length. Returns ``None`` when no match exists -- the
    caller then leaves that turn's weight at 1 rather than crashing.

    Deterministic on purpose: the sibling must not change between runs, or the
    same turn would get a different control each time.
    """
    def best(candidates):
        pick = None
        for candidate in candidates:
            if candidate is trajectory or not candidate.turns:
                continue
            key = (abs(len(candidate.turns) - remaining), candidate.task_id)
            if pick is None or key < pick[0]:
                pick = (key, candidate)
        return pick[1] if pick else None

    # NOTE: the trajectory itself must be excluded when deciding which tier is
    # non-empty. It trivially satisfies both predicates, so including it made
    # `strict` always non-empty, `best()` then found nothing (it skips self) and
    # returned None -- the relaxed fallback was unreachable. Caught by the test.
    others = [c for c in pool if c is not trajectory]
    strict = [c for c in others if c.task_type == trajectory.task_type and c.success == trajectory.success]
    if strict:
        return best(strict)
    relaxed = [c for c in others if c.success == trajectory.success]
    return best(relaxed)


def has_future(trajectory, turn_idx: int, future_gap: int) -> bool:
    """False when Full and Control would be identical, i.e. nothing to score.

    At ``gap = 1`` the last turn has no remaining transition, so its privilege is
    outcome-only in both views and ``e_t`` is exactly 0. ``scorer.py`` skips those
    turns instead of paying for a teacher forward that cannot change anything.
    """
    return any(True for _ in trajectory.iter_tail_transitions(turn_idx + future_gap))


# ---------------------------------------------------------------------------
# Action span
# ---------------------------------------------------------------------------


def tag_token_span(tokenizer, generated_text: str, generated_ids: list[int],
                   tag: str) -> tuple[int, int] | None:
    """Token indices ``[start, end)`` of a tag's CONTENT inside ``generated_ids``.

    The content is what sits between ``<tag>`` and ``</tag>`` -- not the tags
    themselves. Returns ``None`` when the tags are missing or the mapping cannot be
    trusted.

    Mapping strategy: re-tokenise the decoded text with an offset map and require
    that it reproduces ``generated_ids`` exactly. Only then are the offsets used.
    Falling back to a cumulative decode would be O(n^2) and easy to get subtly
    wrong, so a mismatch returns ``None`` and the caller treats the turn as
    unscored rather than scoring the wrong tokens.
    """
    if not generated_text or not generated_ids:
        return None
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    open_idx = generated_text.find(open_tag)
    close_idx = generated_text.rfind(close_tag)
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return None
    content_start = open_idx + len(open_tag)
    content_end = close_idx
    if content_end <= content_start:
        return None

    try:
        encoded = tokenizer(generated_text, return_offsets_mapping=True, add_special_tokens=False)
    except (TypeError, ValueError, NotImplementedError):
        return None
    ids = list(encoded["input_ids"])
    if ids != list(generated_ids):
        return None
    offsets = list(encoded["offset_mapping"])

    span = [i for i, (s, e) in enumerate(offsets) if s < content_end and e > content_start]
    if not span:
        return None
    return span[0], span[-1] + 1


@dataclass(frozen=True)
class ViewPair:
    """One (Full, Control) pair, pre-tokenised and ready for the teacher."""

    full_ids: list[int]
    control_ids: list[int]
    full_prompt_len: int
    control_prompt_len: int
    target_len: int
    #: Token indices of the action content, relative to the START of the target
    #: (i.e. within ``y_t``). Identical in both views by construction.
    action_span: tuple[int, int] | None


def action_token_span(tokenizer, generated_text: str, generated_ids: list[int]) -> tuple[int, int] | None:
    """Action-content span. See ``tag_token_span``."""
    return tag_token_span(tokenizer, generated_text, generated_ids, "action")


def tag_token_span_from_ids(tokenizer, generated_ids: list[int], tag: str) -> tuple[int, int] | None:
    """A tag's content span recovered from the TARGET IDS alone.

    The scorer runs in the training process and only ever sees token ids, so it
    cannot be handed the generated text. Decoding the target and re-mapping it is
    safe here because the target ids ARE the generated ids, and the mapping is
    verified against them before use (see ``tag_token_span``).

    This also avoids inventing a new train-data key: spans are derived, not
    carried, so nothing has to be added to slime's forwarding whitelists.
    """
    if not generated_ids:
        return None
    text = tokenizer.decode(list(generated_ids), skip_special_tokens=False)
    return tag_token_span(tokenizer, text, list(generated_ids), tag)


def action_token_span_from_ids(tokenizer, generated_ids: list[int]) -> tuple[int, int] | None:
    """Action-content span from the target ids. See ``tag_token_span_from_ids``."""
    return tag_token_span_from_ids(tokenizer, generated_ids, "action")


def think_token_span_from_ids(tokenizer, generated_ids: list[int]) -> tuple[int, int] | None:
    """``<think>`` content span from the target ids. See ``tag_token_span_from_ids``."""
    return tag_token_span_from_ids(tokenizer, generated_ids, "think")


def full_generation_span_from_ids(tokenizer, generated_ids: list[int]) -> tuple[int, int] | None:
    """The WHOLE generated turn -- think and action together.

    The historical span was ``action`` (credit only over the action's tokens). The
    full-token form removes the reason the wider span was avoided: `think` was
    rejected because the teacher's response to it was dominated by a
    condition-independent magnitude shift (`reports/think_span_hpi_diagnostic.md`),
    and the full-token credit centres exactly that out within the trajectory.
    Credit over the whole generation also matches what the method claims to
    redistribute -- the turn's own work -- not just its final span.

    No tag decoding is needed: the eligible set is the entire target.
    """
    if not generated_ids:
        return None
    return 0, len(generated_ids)


#: Spans the HPI scorer can assign credit over, selected by ALFWORLD_V2_HPI_SPAN.
SPAN_CHOICES = {
    "action": action_token_span_from_ids,
    "think": think_token_span_from_ids,
    "full": full_generation_span_from_ids,
}


def span_from_ids_for(tokenizer, generated_ids: list[int], which: str) -> tuple[int, int] | None:
    """Dispatch on the span name carried by ``ALFWORLD_V2_HPI_SPAN``."""
    try:
        fn = SPAN_CHOICES[which]
    except KeyError:
        raise ValueError(
            f"unknown ALFWORLD_V2_HPI_SPAN={which!r}; expected one of {sorted(SPAN_CHOICES)}"
        ) from None
    return fn(tokenizer, generated_ids)


def build_full_control_ids(tokenizer, view, trajectory, turn_idx: int, generated_ids: list[int],
                           generated_text: str, future_gap: int) -> ViewPair:
    """Tokenise the Control and Full prompts for one turn.

    The target is the student's raw ``generated_ids`` appended verbatim; it is never
    decoded and re-encoded. Both views therefore teacher-force exactly the same
    target, and their prompts differ only inside the privileged block's trace body.
    """
    from .prompts import build_turn_user_message

    base_user = build_turn_user_message(view)
    control_block = render_hpi_block(trajectory, turn_idx, with_future=False, future_gap=future_gap)
    full_block = render_hpi_block(trajectory, turn_idx, with_future=True, future_gap=future_gap)

    control_prompt = _prompt_ids(tokenizer, base_user + "\n\n" + control_block)
    full_prompt = _prompt_ids(tokenizer, base_user + "\n\n" + full_block)

    return ViewPair(
        full_ids=list(full_prompt) + list(generated_ids),
        control_ids=list(control_prompt) + list(generated_ids),
        full_prompt_len=len(full_prompt),
        control_prompt_len=len(control_prompt),
        target_len=len(generated_ids),
        action_span=action_token_span(tokenizer, generated_text, generated_ids),
    )


def _prompt_ids(tokenizer, user_content: str) -> list[int]:
    from .prompts import build_system_prompt

    messages = [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=os.environ.get("ALFWORLD_THINKING_STYLE", "brief") != "off"
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def view_for_turn(trajectory, turn_idx: int, history_length: int):
    """Rebuild the student's TaskView for a turn, from the trajectory alone.

    The diagnostic must reconstruct the exact conditioning the student had without
    re-rolling out. Everything it needs is on the record: the turn's observation
    (including any inadmissible-action notice), its admissible actions, and the
    preceding (observation, action) pairs.
    """
    from .prompts import TaskView

    pairs = []
    for index in range(turn_idx):
        turn = trajectory.turns[index]
        pairs.append((turn.observation, turn.action_text))
    turn = trajectory.turns[turn_idx]
    return TaskView(
        task_description=trajectory.task_description,
        current_observation=turn.observation,
        admissible_actions=list(turn.admissible_actions),
        history=pairs,
    )
