import pytest

from eom.tokens import count_tokens, ENCODING_NAME


def test_encoding_pinned():
    assert ENCODING_NAME == "cl100k_base"


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_simple():
    assert count_tokens("hello") == 1


def test_count_tokens_sentence():
    n = count_tokens("The quick brown fox jumps over the lazy dog.")
    # cl100k_base tokenises this into 9-10 tokens; assert a stable range.
    assert 8 <= n <= 11


def test_count_tokens_multiline():
    a = count_tokens("hello\nworld")
    b = count_tokens("hello world")
    # Newline introduces at least one extra token.
    assert a >= b


def test_count_tokens_handles_special_token_strings():
    # Real EOM source may quote LLM control tokens; counter must not raise.
    n = count_tokens("the model emits <|endoftext|> when finished")
    assert n > 0
