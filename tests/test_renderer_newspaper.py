# tests/test_renderer_newspaper.py
import pytest  # noqa: F401
from bs4 import BeautifulSoup

from eom.renderers.newspaper import render_newspaper
from tests.fixtures.loader import load_pair


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_returns_html_string():
    _, eom = load_pair("freight_memo")
    html = render_newspaper(eom)
    assert html.startswith("<!DOCTYPE html>") or html.startswith("<html")


def test_contains_headline_in_h1():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    h1 = soup.find("h1")
    assert h1 is not None
    assert "Q1 Freight Cost Update" in h1.get_text()


def test_lead_marked_separately():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    lead = soup.find(class_="eom-lead")
    assert lead is not None
    assert "On-time delivery" in lead.get_text()


def test_tier_a_blocks_in_hero():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    hero = soup.find(class_="eom-hero")
    assert hero is not None
    for b in eom.blocks:
        if b.attention_tier == "A":
            assert b.content[:30] in hero.get_text()


def test_tier_d_in_archive():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    # Archive section should exist (even if empty for fixture without D blocks).
    archive = soup.find(class_="eom-archive")
    assert archive is not None


def test_includes_block_id_attributes():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    head = next(b for b in eom.blocks if b.type == "headline")
    el = soup.find(attrs={"data-block-id": head.id})
    assert el is not None


def test_inline_css_present():
    _, eom = load_pair("freight_memo")
    html = render_newspaper(eom)
    assert "<style>" in html
