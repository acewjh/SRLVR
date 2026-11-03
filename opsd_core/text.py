"""Chat-template user-turn tokenisation. R6: three equivalent helpers existed
(``hpi._chat_prompt_ids``, ``hpi_v2.build_user_ids``, ``hsi_rollout._chat_user_ids``).
One implementation now; the old names remain as thin wrappers at their original
module paths so existing call sites are unchanged.
"""

from __future__ import annotations


def chat_user_ids(tokenizer, user_content: str, *, enable_thinking: bool = False) -> list[int]:
    """Token ids of a chat-templated user turn with a generation prompt.

    ``enable_thinking`` defaults to False because every replay/hindsight view is a
    teacher-forced continuation: the scaffold must not open a fresh think block.
    """
    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(text, add_special_tokens=False)
