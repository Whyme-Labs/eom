from tests.fixtures.loader import load_pair


def test_load_freight_memo():
    source, eom = load_pair("freight_memo")
    assert eom.document_type == "memo"
    assert eom.version == "0.1"
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")
    # Sanity: each block's source_span.quote matches the slice
    for block in eom.blocks:
        if block.source_span:
            sub = source[block.source_span.start : block.source_span.end]
            assert sub == block.source_span.quote, (
                f"block {block.id}: expected {block.source_span.quote!r}, "
                f"got {sub!r}"
            )
