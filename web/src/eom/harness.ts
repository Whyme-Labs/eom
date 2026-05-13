/**
 * EOM harness — per-document validators (H1–H12).
 * TS port of eom/harness.py.
 *
 * The harness is the standard. A document is EOM-conformant iff
 * `validate(eom, sourceText).passed === true`.
 */

import type { Block, EOMDocument } from "./schema";
import { countTokens } from "./tokens";

export interface FailureRecord {
  rule: string;
  message: string;
  block_id?: string;
  span?: readonly [number, number];
}

export interface WarningRecord {
  rule: string;
  message: string;
}

export interface ValidationReport {
  failures: FailureRecord[];
  warnings: WarningRecord[];
  metrics: Record<string, number>;
  passed: boolean;
}

/** H1: exactly one block of type=headline. */
export function checkH1(doc: EOMDocument): FailureRecord[] {
  const n = doc.blocks.filter((b) => b.type === "headline").length;
  return n === 1
    ? []
    : [{ rule: "H1", message: `expected 1 headline, found ${n}` }];
}

/** H2: exactly one lead, with reading_order <= 3. */
export function checkH2(doc: EOMDocument): FailureRecord[] {
  const leads = doc.blocks.filter((b) => b.type === "lead");
  if (leads.length !== 1) {
    return [{ rule: "H2", message: `expected 1 lead, found ${leads.length}` }];
  }
  const lead = leads[0]!;
  if (lead.reading_order > 3) {
    return [{
      rule: "H2",
      message: `lead reading_order=${lead.reading_order} > 3`,
      block_id: lead.id,
    }];
  }
  return [];
}

/**
 * H3: tier distribution caps. |A| <= max(1, 0.10*N); |B| <= max(2, 0.25*N).
 *
 * The fractional caps (10% A, 25% B) bind on long documents, while the
 * small-doc floors (1 A, 2 B) keep the harness usable on short briefs
 * where a single headline already exceeds the percentage.
 */
export function checkH3(doc: EOMDocument): FailureRecord[] {
  const n = doc.blocks.length;
  if (n === 0) return [];
  const counts = { A: 0, B: 0, C: 0, D: 0 };
  for (const b of doc.blocks) counts[b.attention_tier] += 1;
  const out: FailureRecord[] = [];
  const aCap = Math.max(1.0, n * 0.10);
  const bCap = Math.max(2.0, n * 0.25);
  const EPS = 1e-9;
  if (counts.A > aCap + EPS) {
    out.push({
      rule: "H3",
      message: `tier A count ${counts.A}/${n} exceeds cap max(1, 10%)=${aCap.toFixed(2)}`,
    });
  }
  if (counts.B > bCap + EPS) {
    out.push({
      rule: "H3",
      message: `tier B count ${counts.B}/${n} exceeds cap max(2, 25%)=${bCap.toFixed(2)}`,
    });
  }
  return out;
}

/** H4: reading_order is a total order in [0, N) with no duplicates or gaps. */
export function checkH4(doc: EOMDocument): FailureRecord[] {
  const n = doc.blocks.length;
  const orders = doc.blocks.map((b) => b.reading_order).sort((a, b) => a - b);
  const expected = Array.from({ length: n }, (_, i) => i);
  if (orders.length === expected.length && orders.every((v, i) => v === expected[i])) {
    return [];
  }
  const out: FailureRecord[] = [];
  const seen = new Set<number>();
  for (const b of doc.blocks) {
    if (seen.has(b.reading_order)) {
      out.push({
        rule: "H4",
        message: `duplicate reading_order ${b.reading_order}`,
        block_id: b.id,
      });
    }
    seen.add(b.reading_order);
  }
  if (out.length === 0) {
    out.push({
      rule: "H4",
      message: `reading_order is not [0, N); got [${orders.join(",")}], expected [${expected.join(",")}]`,
    });
  }
  return out;
}

/** H5: block IDs unique within document. */
export function checkH5(doc: EOMDocument): FailureRecord[] {
  const seen = new Set<string>();
  const out: FailureRecord[] = [];
  for (const b of doc.blocks) {
    if (seen.has(b.id)) {
      out.push({ rule: "H5", message: `duplicate id ${JSON.stringify(b.id)}`, block_id: b.id });
    }
    seen.add(b.id);
  }
  return out;
}

/** H6: every block has non-empty content (re-check at harness layer). */
export function checkH6(doc: EOMDocument): FailureRecord[] {
  const out: FailureRecord[] = [];
  for (const b of doc.blocks) {
    if (!b.content.trim()) {
      out.push({
        rule: "H6",
        message: "block content is empty or whitespace-only",
        block_id: b.id,
      });
    }
  }
  return out;
}

const CANONICAL_BLOCK_TYPES = new Set<string>([
  "headline", "lead", "claim", "evidence",
  "factbox", "caveat", "decision", "appendix",
]);

/** H7: every block.type is one of the eight canonical types. */
export function checkH7(doc: EOMDocument): FailureRecord[] {
  const out: FailureRecord[] = [];
  for (const b of doc.blocks) {
    if (!CANONICAL_BLOCK_TYPES.has(b.type)) {
      out.push({
        rule: "H7",
        message: `unknown block type ${JSON.stringify(b.type)}`,
        block_id: b.id,
      });
    }
  }
  return out;
}

/** H8: headline <= 100 chars; lead <= 60 words (English). */
export function checkH8(doc: EOMDocument): FailureRecord[] {
  const out: FailureRecord[] = [];
  for (const b of doc.blocks) {
    if (b.type === "headline" && b.content.length > 100) {
      out.push({
        rule: "H8",
        message: `headline length ${b.content.length} > 100`,
        block_id: b.id,
      });
    }
    if (b.type === "lead") {
      const nWords = b.content.split(/\s+/).filter((w) => w.length > 0).length;
      if (nWords > 60) {
        out.push({
          rule: "H8",
          message: `lead word count ${nWords} > 60`,
          block_id: b.id,
        });
      }
    }
  }
  return out;
}

/** H9: sum of tokens across tier A blocks <= attention_budget.B_A. */
export function checkH9(doc: EOMDocument): FailureRecord[] {
  const total = doc.blocks
    .filter((b) => b.attention_tier === "A")
    .reduce((s, b) => s + countTokens(b.content), 0);
  if (total > doc.attention_budget.B_A) {
    return [{
      rule: "H9",
      message: `tier A total tokens ${total} > B_A ${doc.attention_budget.B_A}`,
    }];
  }
  return [];
}

/** H10: sum of tokens across tier A and B blocks <= attention_budget.B_AB. */
export function checkH10(doc: EOMDocument): FailureRecord[] {
  const total = doc.blocks
    .filter((b) => b.attention_tier === "A" || b.attention_tier === "B")
    .reduce((s, b) => s + countTokens(b.content), 0);
  if (total > doc.attention_budget.B_AB) {
    return [{
      rule: "H10",
      message: `tier A+B total tokens ${total} > B_AB ${doc.attention_budget.B_AB}`,
    }];
  }
  return [];
}

/** H11: every evidence/factbox has a valid source_span (offsets, quote). */
export function checkH11(doc: EOMDocument, sourceText: string): FailureRecord[] {
  const out: FailureRecord[] = [];
  for (const b of doc.blocks) {
    if (b.type !== "evidence" && b.type !== "factbox") continue;
    if (!b.source_span) {
      out.push({
        rule: "H11",
        message: `${b.type} block missing source_span`,
        block_id: b.id,
      });
      continue;
    }
    const span = b.source_span;
    if (span.end > sourceText.length) {
      out.push({
        rule: "H11",
        message: `source_span [${span.start},${span.end}) out of range ` +
                 `(source has ${sourceText.length} chars)`,
        block_id: b.id,
        span: [span.start, span.end] as const,
      });
      continue;
    }
    const actual = sourceText.slice(span.start, span.end);
    if (actual !== span.quote) {
      out.push({
        rule: "H11",
        message: `source_span quote mismatch: expected ${JSON.stringify(span.quote)}, ` +
                 `got ${JSON.stringify(actual)}`,
        block_id: b.id,
        span: [span.start, span.end] as const,
      });
    }
  }
  return out;
}

/**
 * H12: claim/decision must have source_span or be is_inferred with valid basis.
 *
 * Inference basis must reference existing evidence/factbox blocks.
 */
export function checkH12(doc: EOMDocument): FailureRecord[] {
  const out: FailureRecord[] = [];
  const byId = new Map<string, Block>();
  for (const b of doc.blocks) byId.set(b.id, b);
  for (const b of doc.blocks) {
    if (b.type !== "claim" && b.type !== "decision") continue;
    if (b.is_inferred) {
      if (b.inference_basis.length === 0) {
        out.push({
          rule: "H12",
          message: `${b.type} is_inferred=True but empty inference_basis`,
          block_id: b.id,
        });
        continue;
      }
      for (const refId of b.inference_basis) {
        const ref = byId.get(refId);
        if (!ref) {
          out.push({
            rule: "H12",
            message: `inference_basis contains unknown id ${JSON.stringify(refId)}`,
            block_id: b.id,
          });
        } else if (ref.type !== "evidence" && ref.type !== "factbox") {
          out.push({
            rule: "H12",
            message: `inference_basis target ${JSON.stringify(refId)} is type ` +
                     `${JSON.stringify(ref.type)}; must be evidence or factbox`,
            block_id: b.id,
          });
        }
      }
    } else if (!b.source_span) {
      out.push({
        rule: "H12",
        message: `${b.type} lacks source_span and is not is_inferred`,
        block_id: b.id,
      });
    }
  }
  return out;
}

/** Run H1-H12 against the document and source text. */
export function validate(doc: EOMDocument, sourceText: string): ValidationReport {
  const failures: FailureRecord[] = [
    ...checkH1(doc),
    ...checkH2(doc),
    ...checkH3(doc),
    ...checkH4(doc),
    ...checkH5(doc),
    ...checkH6(doc),
    ...checkH7(doc),
    ...checkH8(doc),
    ...checkH9(doc),
    ...checkH10(doc),
    ...checkH11(doc, sourceText),
    ...checkH12(doc),
  ];

  const aBlocks = doc.blocks.filter((b) => b.attention_tier === "A");
  const bBlocks = doc.blocks.filter((b) => b.attention_tier === "B");
  const metrics: Record<string, number> = {
    n_blocks: doc.blocks.length,
    tier_a_count: aBlocks.length,
    tier_b_count: bBlocks.length,
    tier_c_count: doc.blocks.filter((b) => b.attention_tier === "C").length,
    tier_d_count: doc.blocks.filter((b) => b.attention_tier === "D").length,
    tier_a_tokens: aBlocks.reduce((s, b) => s + countTokens(b.content), 0),
    tier_ab_tokens:
      aBlocks.reduce((s, b) => s + countTokens(b.content), 0) +
      bBlocks.reduce((s, b) => s + countTokens(b.content), 0),
  };

  const warnings: WarningRecord[] = [
    { rule: "H13", message: "salience monotonicity is corpus-level; not checked here" },
    { rule: "H14", message: "lead centrality is corpus-level; not checked here" },
  ];

  return {
    failures,
    warnings,
    metrics,
    passed: failures.length === 0,
  };
}
