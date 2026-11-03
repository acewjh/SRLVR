"""Bounded exact-text tokenization reuse within one rollout batch.

No model outputs or environment state live here. A fresh context isolates each
batch, concurrent tasks inherit it, and callers always receive a list copy.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar


class TokenCache:
    def __init__(self, max_entries=128, max_tokens=262144, max_chars=1048576):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.entries = OrderedDict()
        self.tokens = self.chars = self.hits = self.misses = self.saved_tokens = 0

    def encode(self, tokenizer, text):
        # The rendered text includes the chat template and thinking settings.
        # Retain tokenizer identity in each entry so its id cannot be recycled.
        key = (id(tokenizer), text)
        if key in self.entries:
            _, ids = self.entries.pop(key)
            self.entries[key] = (tokenizer, ids)
            self.hits += 1
            self.saved_tokens += len(ids)
            return list(ids)
        self.misses += 1
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if (self.max_entries > 0 and len(ids) <= self.max_tokens
                and len(text) <= self.max_chars):
            while self.entries and (len(self.entries) >= self.max_entries
                    or self.tokens + len(ids) > self.max_tokens
                    or self.chars + len(text) > self.max_chars):
                (_, old_text), (_, old_ids) = self.entries.popitem(last=False)
                self.tokens -= len(old_ids)
                self.chars -= len(old_text)
            self.entries[key] = (tokenizer, ids)
            self.tokens += len(ids)
            self.chars += len(text)
        return list(ids)


_active_cache = ContextVar("rollout_token_cache", default=None)


@contextmanager
def token_cache_scope(**limits):
    cache = TokenCache(**limits)
    marker = _active_cache.set(cache)
    try:
        yield cache
    finally:
        _active_cache.reset(marker)
        cache.entries.clear()
        cache.tokens = cache.chars = 0


def encode_prompt(tokenizer, text):
    cache = _active_cache.get()
    if cache is None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return cache.encode(tokenizer, text)
