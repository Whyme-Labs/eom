"""Fetch a small raw-document pool from license-clean sources.

Saves to data/raw/<doc_type>/<slug>.md plus a manifest at data/raw/MANIFEST.json.

Sources (target ~100 total):
- Wikipedia random article API (~50 articles, 300-3000 word range)
- arXiv abstract+intro for recent CS papers (~30 papers, CC-BY)
- IETF RFCs from rfc-editor.org (~20 RFCs)

Usage:
    uv run python scripts/fetch_raw_pool.py --target 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import httpx

RAW_DIR = Path("data/raw")
GOLD_DIR = Path("data/gold")
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(title: str) -> str:
    """Convert a title to a URL-safe slug."""
    s = unicodedata.normalize("NFKD", title)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s[:80]


def _count_words(text: str) -> int:
    return len(text.split())


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"version": "0.1", "documents": []}


def save_manifest(manifest: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def existing_slugs(manifest: dict) -> set[str]:
    return {d["slug"] for d in manifest["documents"]}


def gold_slugs() -> set[str]:
    """All slugs in data/gold/."""
    out = set()
    if GOLD_DIR.exists():
        for type_dir in GOLD_DIR.iterdir():
            if type_dir.is_dir():
                for md in type_dir.glob("*.md"):
                    out.add(md.stem)
    return out


def save_doc_and_manifest(
    manifest: dict,
    doc_type: str,
    slug: str,
    title: str,
    content: str,
    source_url: str,
    license: str,
    n_words: int,
) -> None:
    """Save document to disk and update manifest incrementally."""
    out_dir = RAW_DIR / doc_type
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{content.strip()}\n"
    path = out_dir / f"{slug}.md"
    path.write_text(text, encoding="utf-8")
    manifest["documents"].append({
        "slug": slug,
        "doc_type": doc_type,
        "source_url": source_url,
        "license": license,
        "lang": "en",
        "n_words": n_words,
    })
    save_manifest(manifest)


# ---------------------------------------------------------------------------
# Wikipedia fetcher
# ---------------------------------------------------------------------------

def _strip_wiki_markup(text: str) -> str:
    """Strip wikitext markup with regex (no mwparserfromhell needed)."""
    # Remove {{...}} templates (non-greedy, handle nesting somewhat)
    for _ in range(5):
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if text == prev:
            break
    # Remove [[File:...]] and [[Image:...]] links
    text = re.sub(r"\[\[(?:File|Image|Media):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Extract display text from [[link|display]] or [[link]]
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove external links [http://... text] -> text
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # Remove reference tags <ref...>...</ref> and <ref ... />
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", "", text)
    # Remove other HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove citation markers like [1], [2], [N]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,3}", "", text)
    # Remove section headers (==...==, ===...===)
    text = re.sub(r"={2,}\s*[^=]+\s*={2,}", "", text)
    # Remove table markup
    text = re.sub(r"^\{\|.*?\|\}", "", text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\|[-!].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|-.*$", "", text, flags=re.MULTILINE)
    # Remove category lines
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_wikipedia(target_n: int, skip_slugs: set[str], manifest: dict) -> int:
    """Fetch random Wikipedia articles via API. Returns count fetched."""
    fetched = 0
    attempts = 0
    max_attempts = target_n * 8  # allow many tries due to filtering
    consecutive_429 = 0

    client = httpx.Client(timeout=30.0, follow_redirects=True)
    headers = {
        "User-Agent": "EOM-synthetic-pipeline/0.1 (research; open-source dataset; github.com/eom-project)",
        "Accept": "application/json",
    }

    max_consecutive_429 = 5  # bail out if Wikipedia is persistently rate-limiting
    print(f"  Wikipedia: targeting {target_n} articles…")

    while fetched < target_n and attempts < max_attempts:
        if consecutive_429 >= max_consecutive_429:
            print(f"    Too many consecutive 429s ({consecutive_429}), bailing on Wikipedia")
            break
        attempts += 1
        try:
            # Use action API for random article (more lenient rate limit than REST v1)
            r = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": "1",
                    "format": "json",
                },
                headers=headers,
            )
            if r.status_code == 429:
                consecutive_429 += 1
                retry_after = float(r.headers.get("retry-after", 60))
                backoff = max(retry_after, min(60.0 * consecutive_429, 180.0))
                print(f"    429 rate limited — sleeping {backoff:.0f}s (#{consecutive_429})")
                time.sleep(backoff)
                continue
            r.raise_for_status()
            consecutive_429 = 0

            data = r.json()
            pages = data.get("query", {}).get("random", [])
            if not pages:
                time.sleep(3.0)
                continue
            title = pages[0].get("title", "")
            slug = _slugify(title)

            if not slug or slug in skip_slugs:
                time.sleep(2.0)
                continue

            # Fetch wikitext — longer delay to avoid 429
            time.sleep(4.0)  # conservative rate limit
            r2 = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "parse",
                    "page": title,
                    "prop": "wikitext",
                    "format": "json",
                    "redirects": "1",
                },
                headers=headers,
            )
            if r2.status_code == 429:
                consecutive_429 += 1
                retry_after = float(r2.headers.get("retry-after", 60))
                backoff = max(retry_after, min(60.0 * consecutive_429, 180.0))
                print(f"    429 rate limited (wikitext) — sleeping {backoff:.0f}s")
                time.sleep(backoff)
                continue
            r2.raise_for_status()
            consecutive_429 = 0

            data2 = r2.json()
            wikitext = data2.get("parse", {}).get("wikitext", {}).get("*", "")
            if not wikitext:
                continue

            content = _strip_wiki_markup(wikitext)
            n_words = _count_words(content)

            # Filter word count
            if n_words < 300 or n_words > 3000:
                continue

            # Skip stubs and disambiguation pages
            if re.search(r"\{\{(stub|disambig|disambiguation)[^}]*\}\}", wikitext, re.IGNORECASE):
                continue
            if "may refer to:" in content.lower()[:200]:
                continue

            # Sanitize slug collision
            base_slug = slug
            counter = 0
            while slug in skip_slugs:
                counter += 1
                slug = f"{base_slug}-{counter}"

            source_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            save_doc_and_manifest(
                manifest, "other", slug, title, content,
                source_url, "CC-BY-SA-3.0", n_words,
            )
            skip_slugs.add(slug)
            fetched += 1
            print(f"    [{fetched}/{target_n}] {slug} ({n_words}w)")
            time.sleep(3.0)  # between successful fetches

        except httpx.HTTPStatusError as e:
            print(f"    Wikipedia HTTP error (attempt {attempts}): {e}")
            time.sleep(5.0)
        except Exception as e:
            print(f"    Wikipedia error (attempt {attempts}): {type(e).__name__}: {e}")
            time.sleep(3.0)

    client.close()
    print(f"  Wikipedia: fetched {fetched} articles in {attempts} attempts")
    return fetched


# ---------------------------------------------------------------------------
# arXiv fetcher
# ---------------------------------------------------------------------------

def _parse_arxiv_atom(xml_text: str) -> list[dict]:
    """Parse arXiv Atom feed, return list of entry dicts."""
    import xml.etree.ElementTree as ET
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
        if title_el is None or summary_el is None or id_el is None:
            continue
        arxiv_id = id_el.text.strip().split("/")[-1] if id_el.text else ""
        entries.append({
            "title": title_el.text.strip().replace("\n", " "),
            "abstract": summary_el.text.strip() if summary_el.text else "",
            "arxiv_id": arxiv_id,
            "authors": authors[:5],
        })
    return entries


def fetch_arxiv(target_n: int, skip_slugs: set[str], manifest: dict) -> int:
    """Fetch arXiv CS paper abstracts. Returns count fetched."""
    fetched = 0
    client = httpx.Client(timeout=60.0, follow_redirects=True)
    headers = {"User-Agent": "EOM-synthetic-pipeline/0.1 (research)"}

    queries = [
        "cat:cs.LG AND all:transformer",
        "cat:cs.AI AND all:language model",
        "cat:cs.CL AND all:neural",
        "cat:cs.CV AND all:vision",
        "cat:cs.IR AND all:retrieval",
        "cat:cs.NE AND all:deep learning",
    ]

    print(f"  arXiv: targeting {target_n} papers…")

    for query in queries:
        if fetched >= target_n:
            break
        try:
            time.sleep(3.0)  # arXiv rate limit: 1 query per 3s
            r = client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": query,
                    "max_results": 15,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                headers=headers,
            )
            r.raise_for_status()
            entries = _parse_arxiv_atom(r.text)

            for entry in entries:
                if fetched >= target_n:
                    break
                arxiv_id = entry["arxiv_id"]
                slug = _slugify(f"arxiv-{arxiv_id}")
                if not slug or slug in skip_slugs:
                    continue

                title = entry["title"]
                abstract = entry["abstract"]
                authors = entry["authors"]

                if not abstract or len(abstract.split()) < 50:
                    continue

                # Build document: title + authors + abstract
                author_str = ", ".join(authors) if authors else "Unknown"
                content = f"Authors: {author_str}\n\n## Abstract\n\n{abstract}"
                n_words = _count_words(content)

                source_url = f"https://arxiv.org/abs/{arxiv_id}"
                save_doc_and_manifest(
                    manifest, "paper", slug, title, content,
                    source_url, "CC-BY-4.0", n_words,
                )
                skip_slugs.add(slug)
                fetched += 1
                print(f"    [{fetched}/{target_n}] {slug} ({n_words}w)")

        except Exception as e:
            print(f"    arXiv error (query={query}): {type(e).__name__}: {e}")

    client.close()
    print(f"  arXiv: fetched {fetched} papers")
    return fetched


# ---------------------------------------------------------------------------
# RFC fetcher
# ---------------------------------------------------------------------------

def _extract_rfc_sections(rfc_text: str) -> str:
    """Extract Abstract and Introduction from RFC text."""
    lines = rfc_text.split("\n")
    out_lines = []
    in_section = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect Abstract section
        if re.match(r"^Abstract\s*$", stripped, re.IGNORECASE):
            in_section = True
            out_lines.append(stripped)
            i += 1
            continue

        # Detect Introduction section (numbered or not)
        if re.match(r"^(?:1\.?\s+)?Introduction\s*$", stripped, re.IGNORECASE):
            in_section = True
            out_lines.append(stripped)
            i += 1
            continue

        # End of introduction: numbered section >= 2
        if in_section and re.match(r"^[2-9]\.\s+\w", stripped):
            break

        # Skip page headers/footers (RFC boilerplate)
        if re.match(r"^RFC \d+", stripped) or re.match(r"^\[Page \d+\]", stripped):
            i += 1
            continue

        if in_section:
            out_lines.append(line.rstrip())

        i += 1

    content = "\n".join(out_lines).strip()
    # Clean up excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content


def fetch_rfcs(target_n: int, skip_slugs: set[str], manifest: dict) -> int:
    """Fetch IETF RFCs. Returns count fetched."""
    fetched = 0
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    headers = {"User-Agent": "EOM-synthetic-pipeline/0.1 (research)"}

    # Pick RFCs in range 8000-9500 (recent, standard track)
    candidate_rfcs = list(range(8000, 9501, 10))
    random.shuffle(candidate_rfcs)

    print(f"  RFC: targeting {target_n} RFCs…")

    for rfc_num in candidate_rfcs:
        if fetched >= target_n:
            break
        slug = f"rfc-{rfc_num}"
        if slug in skip_slugs:
            continue
        try:
            time.sleep(1.5)
            r = client.get(
                f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt",
                headers=headers,
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            rfc_text = r.text

            # Extract title from first few lines
            title_match = re.search(r"^(.{20,80})\s*$", rfc_text[:2000], re.MULTILINE)
            rfc_title = title_match.group(1).strip() if title_match else f"RFC {rfc_num}"
            # Better: look for the prominent title line
            title_lines = [
                ln.strip() for ln in rfc_text[:500].split("\n")
                if len(ln.strip()) > 10 and not re.match(
                    r"^(Network|Internet|Request|RFC|IETF|Category|Status|Updates|Obsoletes|ISSN|DOI|Published|Errata)",
                    ln.strip(),
                )
            ]
            if title_lines:
                rfc_title = title_lines[0]

            content = _extract_rfc_sections(rfc_text)
            if not content or len(content.split()) < 100:
                continue

            n_words = _count_words(content)
            if n_words > 3000:
                content = " ".join(content.split()[:3000])
                n_words = 3000

            source_url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt"
            save_doc_and_manifest(
                manifest, "policy", slug, f"RFC {rfc_num}: {rfc_title}", content,
                source_url, "IETF-public-domain", n_words,
            )
            skip_slugs.add(slug)
            fetched += 1
            print(f"    [{fetched}/{target_n}] {slug} ({n_words}w)")

        except Exception as e:
            print(f"    RFC {rfc_num} error: {type(e).__name__}: {e}")

    client.close()
    print(f"  RFC: fetched {fetched} RFCs")
    return fetched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw document pool")
    parser.add_argument("--target", type=int, default=100, help="Total target documents")
    parser.add_argument("--wiki", type=int, default=None, help="Override Wikipedia count")
    parser.add_argument("--arxiv", type=int, default=None, help="Override arXiv count")
    parser.add_argument("--rfc", type=int, default=None, help="Override RFC count")
    args = parser.parse_args()

    # Distribute target — Wikipedia is rate-limited, so cap at 30 and use arXiv+RFC for the rest
    n_wiki = args.wiki if args.wiki is not None else min(30, int(args.target * 0.30))
    n_arxiv = args.arxiv if args.arxiv is not None else int(args.target * 0.40)
    n_rfc = args.rfc if args.rfc is not None else (args.target - n_wiki - n_arxiv)

    print(
        f"Fetch plan: {n_wiki} Wikipedia + {n_arxiv} arXiv + {n_rfc} RFC "
        f"= {n_wiki + n_arxiv + n_rfc} target"
    )

    manifest = load_manifest()
    skip = existing_slugs(manifest) | gold_slugs()
    print(f"Pre-existing slugs to skip: {len(skip)}")
    print(f"Already in manifest: {len(manifest['documents'])}")

    # Wikipedia
    wiki_fetched = 0
    try:
        wiki_fetched = fetch_wikipedia(n_wiki, skip, manifest)
    except Exception as e:
        print(f"WARNING: Wikipedia fetch completely failed: {e}")

    # arXiv
    arxiv_fetched = 0
    try:
        arxiv_fetched = fetch_arxiv(n_arxiv, skip, manifest)
    except Exception as e:
        print(f"WARNING: arXiv fetch completely failed: {e}")

    # RFC
    rfc_fetched = 0
    try:
        rfc_fetched = fetch_rfcs(n_rfc, skip, manifest)
    except Exception as e:
        print(f"WARNING: RFC fetch completely failed: {e}")

    total = len(manifest["documents"])
    print(f"\nDone. New: wiki={wiki_fetched} arxiv={arxiv_fetched} rfc={rfc_fetched}.")
    print(f"Manifest total: {total}")
    by_type: dict[str, int] = {}
    for d in manifest["documents"]:
        by_type[d["doc_type"]] = by_type.get(d["doc_type"], 0) + 1
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
