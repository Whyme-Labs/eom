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
type Env = {
  ASSETS: Fetcher;
  // OPENROUTER_API_KEY is a secret, not a binding. Accessed via env.
  OPENROUTER_API_KEY?: string;
  // Optional storage bindings — present in production, may be undefined in
  // local dev runs without `wrangler dev --persist-to`.
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

async function loadSample(env: Env, request: Request, id: string)
  : Promise<{ raw: string; norm: string; eom: EOMDocument } | Response> {
  if (!SAMPLE_ID_RE.test(id)) {
    return new Response(JSON.stringify({ error: "bad sample id" }),
      { status: 400, headers: { "content-type": "application/json" } });
  }
  const mdResp = await fetchAsset(env, request, `/samples/${id}.md`);
  const eomResp = await fetchAsset(env, request, `/samples/${id}.eom.json`);
  if (mdResp.status === 404 || eomResp.status === 404) {
    return new Response(JSON.stringify({ error: `unknown sample ${id}` }),
      { status: 404, headers: { "content-type": "application/json" } });
  }
  const raw = await mdResp.text();
  const norm = normalise(raw);
  const eomJson = await eomResp.json();
  const parsed = EOMDocument.safeParse(eomJson);
  if (!parsed.success) {
    return new Response(JSON.stringify({
      error: "eom schema invalid",
      details: parsed.error.issues,
    }), { status: 422, headers: { "content-type": "application/json" } });
  }
  return { raw, norm, eom: parsed.data };
}

async function callOpenRouter(env: Env, system: string, user: string,
                               maxTokens: number = 512)
  : Promise<{ text: string; latencyMs: number }> {
  const key = env.OPENROUTER_API_KEY;
  if (!key) throw new Error("OPENROUTER_API_KEY not set");
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

app.get("/api/health", (c) =>
  c.json({ ok: true, ts: new Date().toISOString() })
);

const PackBody = z.object({
  id: SampleId,
  budget: z.number().int().positive().max(8000).default(1500),
});
app.post("/api/render/pack", async (c) => {
  const body = PackBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
  const loaded = await loadSample(c.env, c.req.raw, body.data.id);
  if (loaded instanceof Response) return loaded;
  const text = renderContextPack(loaded.eom, body.data.budget);
  return new Response(text, {
    headers: { "content-type": "text/plain; charset=utf-8" },
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

const AskBody = z.object({
  id: SampleId,
  mode: z.enum(["raw", "pack"]),
  question: z.string().min(1).max(1000),
  budget: z.number().int().positive().max(8000).default(1500),
});
app.post("/api/ask", async (c) => {
  const body = AskBody.safeParse(await c.req.json().catch(() => ({})));
  if (!body.success) return c.json({ error: body.error.issues }, 400);
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
    const { text, latencyMs } = await callOpenRouter(c.env, ANSWER_SYSTEM, user);
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
