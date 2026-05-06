"""EOM — Editorial Object Model.

Public API:
    compile(source_text, hints) -> EOMDocument
    validate(eom, source_text) -> ValidationReport
    render_newspaper(eom) -> str
    render_context_pack(eom, token_budget) -> str
"""

__version__ = "0.1.0"
