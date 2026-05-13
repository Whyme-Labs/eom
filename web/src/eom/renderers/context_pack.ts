/**
 * Token-budgeted LLM payload (TS port of eom/renderers/context_pack.py).
 *
 * Tier-A blocks are always included. Then tier B by priority desc, until
 * budget is tight. Then tier C as one-line summaries. Tier D omitted.
 *
 * The output is the inbound-dialect lowering target: dense, ordered,
 * grounded — what the consuming LLM reads in one shot.
 */

import type { Block, EOMDocument } from "../schema";
import { countTokens } from "../tokens";

const SECTION_ORDER: ReadonlyArray<readonly [Block["type"], string]> = [
  ["headline", "## headline"],
  ["lead", "## lead"],
  ["decision", "## decisions"],
  ["factbox", "## facts"],
  ["evidence", "## evidence"],
  ["claim", "## claims"],
  ["caveat", "## caveats"],
  ["appendix", "## appendix"],
];

function formatBlock(b: Block, withCitation: boolean): string {
  const citation = withCitation ? ` [src:${b.id}]` : "";
  return `- ${b.content}${citation}`;
}

function summaryHeader(eom: EOMDocument, bodyTokens: number): string {
  const compression = bodyTokens / Math.max(1, eom.source.chars);
  return (
    `<!-- eom_v${eom.version} | profile=${eom.render_profile} | ` +
    `document_type=${eom.document_type} | ` +
    `source_chars=${eom.source.chars} | ` +
    `context_tokens=${bodyTokens} | ` +
    `compression=${compression.toFixed(3)} -->\n` +
    `${eom.summary}\n`
  );
}

/**
 * Build a context pack respecting `tokenBudget` (cl100k_base tokens).
 *
 * Returns a string with a header comment, the document summary, and the
 * selected blocks grouped by canonical section order.
 */
export function renderContextPack(eom: EOMDocument, tokenBudget: number): string {
  const byTier: Record<"A" | "B" | "C" | "D", Block[]> = {
    A: [], B: [], C: [], D: [],
  };
  for (const b of eom.blocks) {
    byTier[b.attention_tier].push(b);
  }
  for (const tier of Object.values(byTier)) {
    tier.sort((a, b) =>
      b.priority - a.priority || a.reading_order - b.reading_order
    );
  }

  const chosen: Block[] = [];
  chosen.push(...byTier.A);  // always include tier A

  let used = chosen.reduce((s, b) => s + countTokens(b.content), 0);
  let headroom = Math.max(0, tokenBudget - used - 100);  // 100-tok safety

  // Greedy add tier B by priority desc.
  for (const b of byTier.B) {
    const cost = countTokens(b.content);
    if (cost <= headroom) {
      chosen.push(b);
      headroom -= cost;
    }
  }

  // Tier C as one-line summaries (first sentence, then ellipsis).
  for (const b of byTier.C) {
    const firstSent = b.content.split(". ", 1)[0]!;
    const truncated = firstSent.endsWith(".") ? firstSent : firstSent + "…";
    const cost = countTokens(truncated);
    if (cost <= headroom) {
      chosen.push({ ...b, content: truncated });
      headroom -= cost;
    }
  }

  // Group by section in canonical order.
  const byType = new Map<Block["type"], Block[]>();
  for (const [k] of SECTION_ORDER) byType.set(k, []);
  for (const b of chosen) {
    if (byType.has(b.type)) byType.get(b.type)!.push(b);
  }

  const bodyLines: string[] = [];
  for (const [typeKey, header] of SECTION_ORDER) {
    const blocks = byType.get(typeKey)!;
    if (blocks.length === 0) continue;
    bodyLines.push(header);
    for (const b of blocks) {
      const withCite =
        (b.type === "evidence" || b.type === "factbox") &&
        (b.attention_tier === "A" || b.attention_tier === "B");
      bodyLines.push(formatBlock(b, withCite));
    }
    bodyLines.push("");
  }

  const body = bodyLines.join("\n").replace(/\n+$/, "") + "\n";
  const bodyTokens = countTokens(body);

  return summaryHeader(eom, bodyTokens) + "\n" + body;
}
