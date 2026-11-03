"""Credit span selection: the scorer must be able to score <think> instead of <action>.

Motivation (reports/think_span_hpi_diagnostic.md): on the ACTION span the teacher's
Full-vs-Control log-density ratio is ~0, because the action is a near-deterministic
continuation of the turn's own <think>. The think span carries a far larger
response. This module pins the mechanics of selecting it -- which span is returned,
that the two are disjoint, and that a missing tag is a quiet None rather than an
exception or a silently wrong span.
"""

import pytest

from tasks.alfworld.privilege import (
    SPAN_CHOICES,
    action_token_span,
    span_from_ids_for,
    tag_token_span,
    tag_token_span_from_ids,
    think_token_span_from_ids,
)


def _text(think="plan A", action="go to cabinet 1"):
    return f"<think>{think}</think>\n<action>{action}</action>"


class _RoundTripTokenizer:
    """One char per token, and decode() inverts encode() exactly.

    The shared FakeTokenizer deliberately does not round-trip (its decode uses a
    different modulus), which is fine for the text path but would make every
    ``*_from_ids`` call return None here. The real tokenizer does round-trip --
    otherwise the production scorer could never derive a span from target ids --
    so this stub models that property rather than the other one.
    """

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):  # noqa: ARG002
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def _ids(tokenizer, text):
    return tokenizer.encode(text)


@pytest.fixture
def tokenizer():
    """Shadows conftest's FakeTokenizer: these tests need decode() to invert encode()."""
    return _RoundTripTokenizer()


# --- 1: each tag yields exactly its own content ------------------------------


def test_think_span_covers_only_the_think_content(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    span = tag_token_span(tokenizer, text, ids, "think")
    start, end = span
    # FakeTokenizer is one char per token, so the span is exactly the chars inside.
    assert (start, end) == (
        text.index("<think>") + len("<think>"),
        text.index("</think>"),
    )
    assert text[start:end] == "plan A"


def test_action_span_covers_only_the_action_content(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    span = tag_token_span(tokenizer, text, ids, "action")
    start, end = span
    assert text[start:end] == "go to cabinet 1"


def test_the_two_spans_are_disjoint_and_ordered(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    think = tag_token_span(tokenizer, text, ids, "think")
    action = tag_token_span(tokenizer, text, ids, "action")
    assert think[1] <= action[0], "think must end before the action content begins"
    assert think != action


# --- 2: the ids-only path agrees with the text path --------------------------


def test_from_ids_matches_the_text_path_for_both_tags(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    assert think_token_span_from_ids(tokenizer, ids) == tag_token_span(tokenizer, text, ids, "think")
    assert action_token_span(tokenizer, text, ids) == tag_token_span(tokenizer, text, ids, "action")


# --- 3: dispatcher -----------------------------------------------------------


def test_span_from_ids_for_dispatches_each_supported_name(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    for name in SPAN_CHOICES:
        assert span_from_ids_for(tokenizer, ids, name) is not None


def test_span_from_ids_for_rejects_an_unknown_name(tokenizer):
    ids = _ids(tokenizer, _text())
    with pytest.raises(ValueError, match="ALFWORLD_V2_HPI_SPAN"):
        span_from_ids_for(tokenizer, ids, "reasoning")


# --- 4: failure modes are quiet None, never a wrong span ---------------------


def test_missing_think_tag_returns_none_not_the_action_span(tokenizer):
    text = "<action>go to cabinet 1</action>"  # no <think> at all
    ids = _ids(tokenizer, text)
    assert tag_token_span(tokenizer, text, ids, "think") is None
    assert think_token_span_from_ids(tokenizer, ids) is None


def test_mismatched_ids_return_none(tokenizer):
    text = _text()
    assert tag_token_span(tokenizer, text, [1, 2, 3], "think") is None


def test_empty_inputs_return_none(tokenizer):
    assert tag_token_span(tokenizer, "", [], "think") is None
    assert tag_token_span_from_ids(tokenizer, [], "think") is None


def test_span_never_exceeds_the_target_length(tokenizer):
    text = _text()
    ids = _ids(tokenizer, text)
    for name in SPAN_CHOICES:
        start, end = span_from_ids_for(tokenizer, ids, name)
        assert 0 <= start < end <= len(ids)
