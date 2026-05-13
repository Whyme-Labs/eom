/**
 * Pinned tokeniser for EOM compactness checks.
 *
 * Mirrors eom/tokens.py — fixes cl100k_base so B_A / B_AB budgets are
 * reproducible across language implementations, regardless of the LLM that
 * ultimately consumes the context pack. The Python side uses tiktoken;
 * here we use gpt-tokenizer's cl100k bundle, which is byte-exact.
 *
 * Special-token strings such as `<|endoftext|>` are tokenised as ordinary
 * text rather than raising, because source text may legitimately quote
 * LLM control tokens.
 */

import { encode } from "gpt-tokenizer/encoding/cl100k_base";

export const ENCODING_NAME = "cl100k_base";

export function countTokens(text: string): number {
  if (!text) return 0;
  return encode(text).length;
}
