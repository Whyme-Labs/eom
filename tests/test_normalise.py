from eom.normalise import normalise


def test_normalise_returns_string():
    assert normalise("hello") == "hello"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise("hello   \nworld  \n") == "hello\nworld\n"


def test_normalise_converts_crlf_to_lf():
    assert normalise("a\r\nb\r\nc") == "a\nb\nc"


def test_normalise_converts_lone_cr_to_lf():
    assert normalise("a\rb\rc") == "a\nb\nc"


def test_normalise_strips_bom():
    assert normalise("﻿hello") == "hello"


def test_normalise_applies_nfc():
    # NFC: precomposed é (U+00E9) preferred over decomposed e + combining acute (U+0065 U+0301)
    decomposed = "café"
    assert normalise(decomposed) == "café"
    assert len(normalise(decomposed)) == 4


def test_normalise_idempotent():
    text = "hello\nworld with é and 中文\n"
    assert normalise(normalise(text)) == normalise(text)
