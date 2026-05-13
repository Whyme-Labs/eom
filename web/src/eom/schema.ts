/**
 * EOM schema (TypeScript port of eom/schema.py).
 *
 * v0.1 is the original; v0.2 is strictly additive — every v0.1 document is
 * a valid v0.2 document. New v0.2 fields are optional with v0.1-equivalent
 * defaults. See `docs/SPEC-v0.2.md` in the repo root.
 *
 * Zod is the runtime gate; static types come from `z.infer<>`.
 */

import { z } from "zod";

const SLUG_RE = /^[a-z][a-z0-9-]*$/;
const LANG_RE = /^[a-z]{2}$/;

export const SourceSpan = z.object({
  start: z.number().int().nonnegative(),
  end: z.number().int().nonnegative(),
  quote: z.string().min(1),
}).strict().refine(
  (s) => s.end > s.start,
  { error: "end must be > start" },
);
export type SourceSpan = z.infer<typeof SourceSpan>;

export const SourceMetadata = z.object({
  checksum: z.string().min(1),
  chars: z.number().int().nonnegative(),
  lang: z.string().regex(LANG_RE,
    "lang must be ISO-639-1 (2 lowercase ASCII letters)"),
}).strict();
export type SourceMetadata = z.infer<typeof SourceMetadata>;

export const BLOCK_TYPES = [
  "headline", "lead", "claim", "evidence",
  "factbox", "caveat", "decision", "appendix",
] as const;
export const BlockType = z.enum(BLOCK_TYPES);
export type BlockType = z.infer<typeof BlockType>;

export const ATTENTION_TIERS = ["A", "B", "C", "D"] as const;
export const AttentionTier = z.enum(ATTENTION_TIERS);
export type AttentionTier = z.infer<typeof AttentionTier>;

export const AttentionBudget = z.object({
  B_A: z.number().int().nonnegative(),
  B_AB: z.number().int().nonnegative(),
  token_budget: z.number().int().nonnegative().nullable().optional(),
}).strict().refine(
  (b) => b.B_AB >= b.B_A,
  { error: "B_AB must be >= B_A" },
);
export type AttentionBudget = z.infer<typeof AttentionBudget>;

export const RELATION_TYPES = [
  "supports", "qualifies", "contradicts",
  "derived_from", "cites", "refines",
] as const;
export const RelationType = z.enum(RELATION_TYPES);
export type RelationType = z.infer<typeof RelationType>;

export const Relation = z.object({
  type: RelationType,
  target: z.string(),
  confidence: z.number().min(0).max(1).default(1.0),
}).strict();
export type Relation = z.infer<typeof Relation>;

export const ROLE_TAGS = ["ground_truth", "claim", "speculation", "citation"] as const;
export const RoleTag = z.enum(ROLE_TAGS);
export type RoleTag = z.infer<typeof RoleTag>;

export const EVIDENCE_LAYERS = ["surface", "drill", "archive"] as const;
export const EvidenceLayer = z.enum(EVIDENCE_LAYERS);
export type EvidenceLayer = z.infer<typeof EvidenceLayer>;

export const SYSTEM_INTENTS = [
  "question", "summarize", "extract", "compare", "decide",
] as const;
export const SystemIntent = z.enum(SYSTEM_INTENTS);
export type SystemIntent = z.infer<typeof SystemIntent>;

export const DIALECTS = ["outbound", "inbound"] as const;
export const Dialect = z.enum(DIALECTS);
export type Dialect = z.infer<typeof Dialect>;

export const Block = z.object({
  id: z.string().regex(SLUG_RE,
    "id must be slug (lowercase, alphanumeric, hyphen)"),
  type: BlockType,
  content: z.string().refine(
    (s) => s.trim().length > 0,
    "content must be non-empty (and non-whitespace)",
  ),
  attention_tier: AttentionTier,
  priority: z.number().min(0).max(1),
  reading_order: z.number().int().nonnegative(),
  source_span: SourceSpan.nullable().optional(),
  is_inferred: z.boolean().default(false),
  inference_basis: z.array(z.string()).default([]),
  parent_id: z.string().nullable().optional(),
  relations: z.array(Relation).default([]),
  role_tag: RoleTag.nullable().optional(),
  evidence_layer: EvidenceLayer.nullable().optional(),
}).strict().superRefine((b, ctx) => {
  if (b.is_inferred && b.type !== "claim" && b.type !== "decision") {
    ctx.addIssue({
      code: "custom",
      message: `is_inferred=True only allowed for claim/decision, got ${b.type}`,
    });
  }
  if (b.inference_basis.length > 0 && !b.is_inferred) {
    ctx.addIssue({
      code: "custom",
      message: "inference_basis is only valid when is_inferred=True",
    });
  }
});
export type Block = z.infer<typeof Block>;

export const DOCUMENT_TYPES = [
  "memo", "report", "paper", "transcript", "news", "policy", "other",
] as const;
export const DocumentType = z.enum(DOCUMENT_TYPES);
export type DocumentType = z.infer<typeof DocumentType>;

export const RENDER_PROFILE_NAMES = ["executive_brief", "analytical_brief"] as const;
export const RenderProfileName = z.enum(RENDER_PROFILE_NAMES);
export type RenderProfileName = z.infer<typeof RenderProfileName>;

export const VERSIONS = ["0.1", "0.2"] as const;
export const SchemaVersion = z.enum(VERSIONS);
export type SchemaVersion = z.infer<typeof SchemaVersion>;

export const EOMDocument = z.object({
  version: SchemaVersion,
  document_type: DocumentType,
  summary: z.string().refine((s) => s.trim().length > 0,
    "summary must be non-empty (and non-whitespace)"),
  render_profile: RenderProfileName,
  attention_budget: AttentionBudget,
  blocks: z.array(Block).min(1),
  source: SourceMetadata,
  dialect: Dialect.default("outbound"),
  system_intent: SystemIntent.nullable().optional(),
}).strict();
export type EOMDocument = z.infer<typeof EOMDocument>;

/**
 * Canonical render-profile budgets (mirrors RENDER_PROFILES in Python).
 * Frozen via `as const satisfies`.
 */
export const RENDER_PROFILES = {
  executive_brief: { B_A: 200, B_AB: 800 },
  analytical_brief: { B_A: 400, B_AB: 2000 },
} as const satisfies Record<RenderProfileName, { B_A: number; B_AB: number }>;
