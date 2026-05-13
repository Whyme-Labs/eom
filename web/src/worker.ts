/**
 * EOM live-demo Worker.
 *
 * Static assets (public/) are served by the ASSETS binding.
 * API routes under /api/* are handled by this Worker.
 *
 *   POST /api/render/pack       { id, budget }                -> text
 *   POST /api/render/newspaper  { id }                         -> html
 *   POST /api/render/validate   { id }                         -> ValidationReport
 *   POST /api/ask               { id, mode, question, budget } -> ask-row
 *
 * `id` is a sample id like "policy/gdpr" (see public/samples/manifest.json).
 * `mode` is "raw" or "pack".
 */

import { Hono } from "hono";
import { z } from "zod";
import { validate } from "./eom/harness";
import { normalise } from "./eom/normalise";
import { renderContextPack } from "./eom/renderers/context_pack";
import { renderNewspaper } from "./eom/renderers/newspaper";
import { EOMDocument } from "./eom/schema";
import { countTokens } from "./eom/tokens";

// CF Workers env bindings — declared in wrangler.jsonc.
//
// /api/ask is BYO-key: users supply their own OpenRouter key in the
// request. The legacy OPENROUTER_API_KEY secret is *not* used for /api/ask
// and is kept only for forward compatibility (and so `wrangler secret put`
// during local dev is harmless).
type Env = {
  ASSETS: Fetcher;
  OPENROUTER_API_KEY?: string;  // unused by /api/ask; kept for back-compat
  CORPUS?: R2Bucket;
  DB?: D1Database;
  CACHE?: KVNamespace;
  AI?: Ai;
};

const app = new Hono<{ Bindings: Env }>();

const SAMPLE_ID_RE = /^[a-z0-9_-]+\/[a-z0-9_-]+$/;
const SampleId = z.string().regex(SAMPLE_ID_RE, "id must be 'type/slug'");

const ANSWERER_MODEL = "google/gemma-4-31b-it";
const ANSWER_SYSTEM = (
  "You answer questions about a document the user provides. " +
  "Quote the relevant text where helpful. If the answer is not in the " +
  "document, say 'not in document' and stop. Be concise."
);

// --- helpers --------------------------------------------------------------

async function fetchAsset(env: Env, request: Request, path: string): Promise<Response> {
  // Build a same-origin request and forward to the ASSETS binding.
  const url = new URL(request.url);
  url.pathname = path;
  return env.ASSETS.fetch(new Request(url.toString(), { method: "GET" }));
}

// R2 keys mirror the path layout: <type>/<slug>.md, <type>/<slug>.eom.json.
async function fetchSampleFromR2(env: Env, id: string)
  : Promise<{ raw: string; eom: unknown } | null> {
  if (!env.CORPUS) return null;
  const [md, eom] = await Promise.all([
    env.CORPUS.get(`${id}.md`),
    env.CORPUS.get(`${id}.eom.json`),
  ]);
  if (!md || !eom) return null;
  return { raw: await md.text(), eom: await eom.json() };
}

async function fetchSampleFromAssets(env: Env, request: Request, id: string)
  : Promise<{ raw: string; eom: unknown } | null> {
  const [mdResp, eomResp] = await Promise.all([
    fetchAsset(env, request, `/samples/${id}.md`),
    fetchAsset(env, request, `/samples/${id}.eom.json`),
  ]);
  if (mdResp.status === 404 || eomResp.status === 404) return null;
  return { raw: await mdResp.text(), eom: await eomResp.json() };
}

async function loadSample(env: Env, request: Request, id: string)
  : Promise<{ raw: string; norm: string; eom: EOMDocument; source: "r2" | "assets" } | Response> {
  if (!SAMPLE_ID_RE.test(id)) {
    return new Response(JSON.stringify({ error: "bad sample id" }),
      { status: 400, headers: { "content-type": "application/json" } });
  }
  // Prefer R2 (canonical store); fall back to bundled assets so wrangler
  // dev runs without R2 still work.
  let source: "r2" | "assets" = "r2";
  let raw_eom = await fetchSampleFromR2(env, id);
  if (!raw_eom) {
    raw_eom = await fetchSampleFromAssets(env, request, id);
    source = "assets";
  }
  if (!raw_eom) {
    return new Response(JSON.stringify({ error: `unknown sample ${id}` }),
      { status: 404, headers: { "content-type": "application/json" } });
  }
  const parsed = EOMDocument.safeParse(raw_eom.eom);
  if (!parsed.success) {
    return new Response(JSON.stringify({
      error: "eom schema invalid",
      details: parsed.error.issues,
    }), { status: 422, headers: { "content-type": "application/json" } });
  }
  return { raw: raw_eom.raw, norm: normalise(raw_eom.raw), eom: parsed.data, source };
}

// KV cache for renderContextPack output. Key = `${id}::${budget}`; value is
// the rendered text. EOM compilations are immutable per (id, budget) so a
// 24h TTL is safe; bumping the EOM schema bumps the key implicitly via id.
const PACK_CACHE_TTL_SECONDS = 24 * 60 * 60;
async function cachedPack(env: Env, id: string, budget: number,
                          eom: EOMDocument): Promise<{ text: string; cached: boolean }> {
  const key = `pack::${id}::${budget}`;
  if (env.CACHE) {
    const hit = await env.CACHE.get(key);
    if (hit) return { text: hit, cached: true };
  }
  const text = renderContextPack(eom, budget);
  if (env.CACHE) {
    await env.CACHE.put(key, text, { expirationTtl: PACK_CACHE_TTL_SECONDS });
  }
  return { text, cached: false };
}

async function callOpenRouter(apiKey: string, system: string, user: string,
                               maxTokens: number = 512)
  : Promise<{ text: string; latencyMs: number }> {
  const key = apiKey;
  if (!key) throw new Error("OpenRouter key required");
  const t0 = Date.now();
  const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: ANSWERER_MODEL,
      temperature: 0.0,
      max_tokens: maxTokens,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`openrouter ${resp.status}: ${body.slice(0, 200)}`);
  }
  const data: any = await resp.json();
  const text = data?.choices?.[0]?.message?.content ?? "";
  return { text, latencyMs: Date.now() - t0 };
}

// --- routes ---------------------------------------------------------------

app.get("/api/health", async (c) => {
  // Bind-aware health check so the demo can show which CF services are wired.
  const bindings = {
    assets: !!c.env.ASSETS,
    r2: !!c.env.CORPUS,
    d1: !!c.env.DB,
    kv: !!c.env.CACHE,
    ai: !!c.env.AI,
    openrouter: !!c.env.OPENROUTER_API_KEY,
  };
  let d1_docs: number | null = null;
  if (c.env.DB) {
    try {
      const row = await c.env.DB.prepare("SELECT COUNT(*) AS n FROM docs").first<{ n: number }>();
      d1_docs = row?.n ?? null;
    } catch {
      d1_docs = null;
    }
  }
  // openrouter binding presence doesn't grant /api/ask access — it's BYO-key.
  // Surface this explicitly so the frontend can show the right UI.
  return c.json({
    ok: true,
    ts: new Date().toISOString(),
    bindings,
    d1_docs,
    ask_auth: "byo-key",
  });
});

/** List all sample docs — D1-backed with static fallback. */
app.get("/api/samples", async (c) => {
  if (c.env.DB) {
    try {
      const { results } = await c.env.DB.prepare(
        "SELECT id, type, slug, title FROM docs ORDER BY type, slug",
      ).all<{ id: string; type: string; slug: string; title: string }>();
      if (results && results.length > 0) {
        return c.json({ source: "d1", samples: results });
      }
    } catch (e) {
      // fall through to static
    }
  }
  // Static fallback — bundled manifest under public/samples/.
  const r = await fetchAsset(c.env, c.req.raw, "/samples/manifest.json");
  const samples = await r.json();
  return c.json({ source: "assets", samples });
});

/** Questions for a doc — D1-backed with static fallback. */
app.get("/api/qsets/:type/:slug", async (c) => {
  const id = `${c.req.param("type")}/${c.req.param("slug")}`;
  if (c.env.DB) {
    try {
      const { results } = await c.env.DB.prepare(
        "SELECT q_id AS id, question AS q, reference AS ref " +
        "FROM qsets WHERE doc_id = ? ORDER BY position",
      ).bind(id).all<{ id: string; q: string; ref: string }>();
      return c.json({ source: "d1", doc_id: id, questions: results ?? [] });
    } catch (e) {
      // fall through
    }
  }
  // Static fallback: filter qsets.json by slug (matches the JSON layout).
  const r = await fetchAsset(c.env, c.req.raw, "/qsets.json");
  const data = await r.json() as { qsets?: Array<{ doc_id: string; questions: any[] }> };
  const slug = c.req.param("slug");
  const match = data.qsets?.find((q) => q.doc_id === slug);
  return c.json({
    source: "assets",
    doc_id: id,
    questions: match?.questions ?? [],
  });
});

/** Latest inbound-benchmark roll-up — D1-backed. */
app.get("/api/bench/results", async (c) => {
  if (!c.env.DB) return c.json({ source: "none", rows: [] });
  try {
    const { results } = await c.env.DB.prepare(
      "SELECT run_id, doc_id, mode, COUNT(*) AS n, " +
      "       SUM(input_tokens) AS sum_input_tokens, " +
      "       AVG(latency_ms) AS avg_latency_ms, " +
      "       AVG(judge_score) AS avg_score " +
      "FROM bench_results " +
      "GROUP BY run_id, doc_id, mode " +
      "ORDER BY run_id DESC, doc_id, mode",
    ).all();
    return c.json({ source: "d1", rows: results ?? [] });
  } catch (e: unknown) {
    return c.json({ error: e instanceof Error ? e.message : String(e) }, 500);
  }
});

const PackBody = z.object({
  id: SampleId,
  budget: z.number().int().positive().max(8000).default(1500),
});
app.post("/api/render/pack", async (c) => {
  const body = PackBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
  const loaded = await loadSample(c.env, c.req.raw, body.data.id);
  if (loaded instanceof Response) return loaded;
  const { text, cached } = await cachedPack(c.env, body.data.id, body.data.budget, loaded.eom);
  return new Response(text, {
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "x-eom-source": loaded.source,
      "x-eom-cache": cached ? "hit" : "miss",
    },
  });
});

const NewspaperBody = z.object({ id: SampleId });
app.post("/api/render/newspaper", async (c) => {
  const body = NewspaperBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
  const loaded = await loadSample(c.env, c.req.raw, body.data.id);
  if (loaded instanceof Response) return loaded;
  const html = renderNewspaper(loaded.eom);
  return new Response(html, {
    headers: { "content-type": "text/html; charset=utf-8" },
  });
});

app.post("/api/render/validate", async (c) => {
  const body = NewspaperBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
  const loaded = await loadSample(c.env, c.req.raw, body.data.id);
  if (loaded instanceof Response) return loaded;
  return c.json(validate(loaded.eom, loaded.norm));
});

// /api/ask is BYO-key: the user supplies their own OpenRouter API key in
// the request body (or in the Authorization header, sk-or-... pattern).
// This keeps the demo zero-cost regardless of traffic. The server-side
// OPENROUTER_API_KEY secret is intentionally NOT a fallback here.
const AskBody = z.object({
  id: SampleId,
  mode: z.enum(["raw", "pack"]),
  question: z.string().min(1).max(1000),
  budget: z.number().int().positive().max(8000).default(1500),
  apiKey: z.string().regex(/^sk-or-[A-Za-z0-9_-]+$/,
    "expected an OpenRouter key starting with sk-or-").optional(),
});

function extractApiKey(c: any): string | null {
  // Prefer Authorization header (Bearer sk-or-...), fall back to body.apiKey.
  const auth = c.req.header("authorization") || "";
  const m = auth.match(/^Bearer\s+(sk-or-[A-Za-z0-9_-]+)$/);
  return m ? m[1] : null;
}

app.post("/api/ask", async (c) => {
  const body = AskBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
  const apiKey = extractApiKey(c) || body.data.apiKey;
  if (!apiKey) {
    return c.json({
      error: "missing OpenRouter API key — paste yours in the sidebar " +
             "(stored only in your browser's localStorage). " +
             "Get one at https://openrouter.ai/keys",
    }, 401);
  }
  const loaded = await loadSample(c.env, c.req.raw, body.data.id);
  if (loaded instanceof Response) return loaded;

  const context = body.data.mode === "raw"
    ? loaded.norm
    : renderContextPack(loaded.eom, body.data.budget);
  const kind = body.data.mode === "raw" ? "raw markdown" : "EOM context-pack";
  const user =
    `### Document (${kind})\n\n${context}\n\n` +
    `### Question\n${body.data.question}\n\n` +
    "Answer the question using only the document above.";
  const inputTokens = countTokens(user);

  try {
    const { text, latencyMs } = await callOpenRouter(apiKey, ANSWER_SYSTEM, user);
    return c.json({
      mode: body.data.mode,
      answer: text,
      latencyMs,
      inputTokens,
      outputTokens: countTokens(text),
      contextPreview: context.length > 1200 ? context.slice(0, 1200) + "…" : context,
      contextTokens: countTokens(context),
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return c.json({ error: msg }, 502);
  }
});

// Catch-all 404 for /api/*; static assets fall through to ASSETS binding.
app.notFound((c) => {
  if (c.req.path.startsWith("/api/")) {
    return c.json({ error: "not found" }, 404);
  }
  // Defer to the static-asset router.
  return c.env.ASSETS.fetch(c.req.raw);
});

export default app;
