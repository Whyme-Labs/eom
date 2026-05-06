"""Pinned tokeniser for EOM compactness checks.

The harness fixes cl100k_base so that B_A / B_AB budgets are
reproducible across implementations, regardless of the LLM that
ultimately consumes the context pack.
"""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` under the pinned encoding.

    Special-token strings such as ``<|endoftext|>`` are tokenised as
    ordinary text rather than raising, because EOM source text may
    legitimately quote LLM control tokens (e.g. articles about LLMs).
    """
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))
