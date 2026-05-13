# Kaggle submission checklist — Gemma 4 Good Hackathon (Unsloth track)

Run this top-to-bottom on submission day.

## 0. Repo state

```bash
git log --oneline | head -10        # latest commit should be the polish commit
git status                          # nothing uncommitted (modulo .claude/)
uv run pytest -q --ignore=tests/test_renderer_newspaper.py
```

Expect 213 passed.

## 1. Public GitHub repo

✓ Done — repo at `git@github.com:Whyme-Labs/eom.git`, default branch
`main`. To re-sync after later commits:

```bash
git push origin main
```

## 2. Cloudflare Workers deploy

The judges should be able to click a live URL. The canonical demo lives
in `web/` (Hono + static assets + Workers AI binding). The Streamlit
demo at `demo/app.py` remains for local Python dev.

```bash
cd web
bun install                # if not done already
bun run scripts/sync.ts    # refresh public/ from data/gold + data/bench

# One-time: log in to Cloudflare and set the secret
bunx wrangler login
bunx wrangler secret put OPENROUTER_API_KEY   # paste the sk-or-... key

# (Optional, for the maximalist suite story) create the bindings
# referenced by wrangler.jsonc and paste the returned ids in:
bunx wrangler r2 bucket create eom-corpus
bunx wrangler d1 create eom-data
bunx wrangler kv namespace create CACHE

bun run deploy             # wrangler deploy
```

Capture the worker URL (e.g. `eom-demo.<your-subdomain>.workers.dev`)
— it goes in the Kaggle submission form.

For local dev: `bun run dev` (= `wrangler dev --local`), opens
`http://127.0.0.1:8787`. Reads `web/.dev.vars` for the OpenRouter key.

## 3. Record the video

`docs/VIDEO-SCRIPT.md` is the shot-by-shot. Pre-flight:

```bash
export OPENROUTER_API_KEY=...
uv run python -m bench.inbound --docs paris-2024 --no-judge   # warm caches
uv run streamlit run demo/app.py --server.port 8501
```

Record at 1080p, 30fps, 16:9. Upload to YouTube (unlisted is fine).
Capture the URL.

## 4. Modal adapter URL (Unsloth track requirement)

Confirm the v5 adapter is still on the volume:

```bash
modal volume ls eom-sft-out | grep gemma4-v5
```

Should list `eom-sft-adapter-gemma4-v5/` with `adapter_config.json`,
`adapter_model.safetensors`, plus tokenizer files.

For the submission, the adapter location reference is:
> `modal://eom-sft-out/eom-sft-adapter-gemma4-v5`

## 5. Kaggle submission form fields

Open the [submission page](https://www.kaggle.com/competitions/gemma-4-good-hackathon/submissions).
Paste the following.

### Title
> EOM — a two-way wire format between humans and models

### Tagline (≤140 chars)
> Markdown is for storage. EOM is for dialogue. One IR, two dialects, one validator. Inbound demo cuts LLM input by 52% on benchmark.

### Writeup
Paste contents of `docs/KAGGLE-WRITEUP.md` (1163 words).

### Links

| Field | Value |
|---|---|
| Code repository | `https://github.com/Whyme-Labs/eom` |
| Live demo | `https://eom-demo.swmengappdev.workers.dev` |
| Video | `https://youtu.be/<id>` |
| Adapter (Unsloth track) | `modal://eom-sft-out/eom-sft-adapter-gemma4-v5` |

### Track
> Unsloth ($10K)

## 6. Pre-submit sanity checks

- [ ] `git push origin main` succeeded; the public repo shows the latest commit
- [ ] CF Worker URL loads without an error banner; the sample picker is populated
- [ ] OPENROUTER_API_KEY secret is set on the Worker (test the **🔄 Ask AI** tab end-to-end)
- [ ] Video plays end-to-end on a fresh tab (incognito, logged out)
- [ ] `docs/SPEC-v0.2.md`, `docs/KAGGLE-WRITEUP.md`, `docs/VIDEO-SCRIPT.md`, `LICENSE` all visible on GitHub
- [ ] Modal volume `eom-sft-out` still has `eom-sft-adapter-gemma4-v5`
- [ ] No secrets in any committed file (`grep -r 'sk-or-' . --include='*.py' --include='*.ts' --include='*.md'` returns only placeholders)

## 7. Submit

Click submit on the Kaggle form. Capture the submission ID from the URL
post-submit and add it to this file under the "Submitted" section.

## 8. After submission

```bash
git tag -a v0.2.0 -m "Kaggle hackathon submission"
git push origin v0.2.0
```

## Submitted

| When | Submission ID | Notes |
|---|---|---|
| _(fill after submit)_ | | |
