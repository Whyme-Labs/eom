/**
 * Text normalisation for EOM source text (TS port of eom/normalise.py).
 *
 * All offset-comparing code (validators, compilers) must apply this before
 * computing or checking offsets, so byte-for-byte agreement is guaranteed
 * across implementations.
 */

// Match every trailing character where Python's str.isspace() is true.
// JS \s covers ASCII whitespace; explicit \u escapes cover NBSP (U+00A0),
// the Zs block (U+2000..U+200A), line/paragraph separators (U+2028,
// U+2029), and ideographic space (U+3000).
const TRAILING_WS = new RegExp("[\\s\\u00a0\\u2000-\\u200a\\u2028\\u2029\\u3000]+$");

/**
 * Return text in canonical EOM form.
 *
 * Order of operations:
 * 1. Strip a leading U+FEFF (BOM) if present. Mid-text U+FEFF is preserved.
 * 2. Normalise CRLF and lone CR to LF.
 * 3. Strip trailing whitespace from each line, preserving line breaks.
 * 4. Apply NFC unicode normalisation.
 *
 * Idempotent: `normalise(normalise(t)) === normalise(t)`.
 */
export function normalise(text: string): string {
  if (text.charCodeAt(0) === 0xFEFF) {
    text = text.slice(1);
  }
  text = text.replace(/\r\n?/g, "\n");
  text = text
    .split("\n")
    .map((line) => line.replace(TRAILING_WS, ""))
    .join("\n");
  return text.normalize("NFC");
}
