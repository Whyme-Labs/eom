/**
 * Printable single-page newspaper-style HTML view of an EOM document
 * (TS port of eom/renderers/newspaper.py).
 *
 * Outbound dialect lowering target: a static HTML page a human reads in
 * thirty seconds, with every claim grounded to a source span.
 *
 * The Jinja template + .css file are inlined here as template literals so
 * the renderer is self-contained on the edge runtime (no FS reads).
 */

import type { Block, EOMDocument } from "../schema";

const CSS = `* { box-sizing: border-box; }
body {
  font-family: 'Georgia', 'Times New Roman', serif;
  max-width: 1100px;
  margin: 2em auto;
  padding: 0 1em;
  color: #1a1a1a;
  line-height: 1.5;
}
.eom-meta { color: #666; font-size: 0.85em; border-bottom: 1px solid #999; padding-bottom: 0.5em; }
.eom-hero { border-bottom: 3px solid #000; padding-bottom: 1em; margin-bottom: 1.5em; }
.eom-hero h1 { font-size: 2.4em; margin: 0 0 0.3em; line-height: 1.1; }
.eom-lead { font-size: 1.2em; font-weight: 600; color: #333; margin: 0.5em 0 1em; }
.eom-factbox {
  background: #f5f0e6;
  border-left: 4px solid #b8860b;
  padding: 0.75em 1em;
  font-size: 0.95em;
  margin: 1em 0;
}
.eom-decision {
  background: #1a1a1a;
  color: #fff;
  padding: 0.75em 1em;
  margin: 1em 0;
  font-weight: 600;
}
.eom-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 2em; }
.eom-main { font-size: 1em; }
.eom-rail { font-size: 0.92em; color: #444; border-left: 1px dashed #ccc; padding-left: 1em; }
.eom-block { margin: 0.75em 0; }
.eom-block p { margin: 0.4em 0; }
.eom-caveat {
  font-style: italic;
  color: #555;
  border-top: 1px solid #ccc;
  margin-top: 1em;
  padding-top: 0.5em;
}
.eom-archive {
  margin-top: 2em;
  border-top: 2px solid #ccc;
  padding-top: 0.75em;
  font-size: 0.85em;
  color: #777;
}
.eom-archive summary { cursor: pointer; font-weight: 600; }
.eom-block sup { color: #b8860b; font-size: 0.75em; }
@media print {
  body { max-width: 100%; }
  .eom-archive[open] summary { font-weight: bold; }
}`;

const HTML_ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]!);
}

interface Partitions { A: Block[]; B: Block[]; C: Block[]; D: Block[]; }

function partition(blocks: ReadonlyArray<Block>): Partitions {
  const out: Partitions = { A: [], B: [], C: [], D: [] };
  for (const b of blocks) out[b.attention_tier].push(b);
  const cmp = (a: Block, b: Block) =>
    b.priority - a.priority || a.reading_order - b.reading_order;
  out.A.sort(cmp); out.B.sort(cmp); out.C.sort(cmp); out.D.sort(cmp);
  return out;
}

function blockHtml(b: Block, withCitation: boolean): string {
  const cite = withCitation && b.source_span ? `<sup>[${esc(b.id)}]</sup>` : "";
  return `    <div class="eom-block eom-${esc(b.type)}" data-block-id="${esc(b.id)}">
      <p>${esc(b.content)}</p>
      ${cite}
    </div>`;
}

/** Render an EOM document as a printable HTML page. */
export function renderNewspaper(eom: EOMDocument): string {
  const parts = partition(eom.blocks);

  let headline = parts.A.find((b) => b.type === "headline");
  if (!headline) {
    // Caller bypassed harness; render with placeholder.
    headline = {
      id: "headline-missing",
      type: "headline",
      content: "(missing headline)",
      attention_tier: "A",
      priority: 1.0,
      reading_order: 0,
      is_inferred: false,
      inference_basis: [],
      relations: [],
    } as Block;
  }

  const lead = eom.blocks.find((b) => b.type === "lead");
  const tierAExtras = parts.A.filter(
    (b) => b.type !== "headline" && b.type !== "lead",
  );
  const tierB = parts.B.filter((b) => b.type !== "lead");
  const tierC = parts.C.filter((b) => b.type !== "lead");
  const tierD = parts.D.filter((b) => b.type !== "lead");

  const leadHtml = lead
    ? `  <div class="eom-lead" data-block-id="${esc(lead.id)}">${esc(lead.content)}</div>`
    : "";

  return `<!DOCTYPE html>
<html lang="${esc(eom.source.lang)}">
<head>
<meta charset="utf-8">
<title>${esc(headline.content)}</title>
<style>${CSS}</style>
</head>
<body>
<div class="eom-meta">EOM v${esc(eom.version)} · ${esc(eom.document_type)} · profile=${esc(eom.render_profile)}</div>

<div class="eom-hero">
  <h1 data-block-id="${esc(headline.id)}">${esc(headline.content)}</h1>
${leadHtml}
${tierAExtras.map((b) => blockHtml(b, true)).join("\n")}
</div>

<div class="eom-grid">
  <div class="eom-main">
${tierB.map((b) => blockHtml(b, true)).join("\n")}
  </div>
  <aside class="eom-rail">
${tierC.map((b) => blockHtml(b, false)).join("\n")}
  </aside>
</div>

<details class="eom-archive">
  <summary>Archive (${tierD.length} blocks)</summary>
${tierD.map((b) => blockHtml(b, false)).join("\n")}
</details>

</body>
</html>
`;
}
