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

This is a **user action** — the assistant has not pushed.

```bash
# If not yet created:
gh repo create <handle>/eom --public --source=. --push --description \
  "EOM — two-way wire format between humans and models"

# Otherwise:
git remote add origin git@github.com:<handle>/eom.git
git push -u origin master:main      # Kaggle expects 'main'
```

Then update the URL in three places if `<handle>` is not `soh`:

- `README.md` — the GitHub link in the License section
- `demo/app.py` footer
- `docs/KAGGLE-WRITEUP.md` if any link references it

## 2. Streamlit Cloud deploy

The judges should be able to click a live URL.

1. Visit https://streamlit.io/cloud → New app.
2. Repo `<handle>/eom`, branch `main`, main file `demo/app.py`.
3. Advanced → Secrets → paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```
4. Deploy. Cold start ~2 min. Confirm http URL works.

Capture the URL — it goes in the Kaggle submission form.

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
| Code repository | `https://github.com/<handle>/eom` |
| Live demo | `https://<your-app>.streamlit.app/` |
| Video | `https://youtu.be/<id>` |
| Adapter (Unsloth track) | `modal://eom-sft-out/eom-sft-adapter-gemma4-v5` |

### Track
> Unsloth ($10K)

## 6. Pre-submit sanity checks

- [ ] `git push origin main` succeeded; the public repo shows the latest commit
- [ ] Streamlit Cloud URL loads without an error banner
- [ ] Streamlit secret is set (test by clicking **🔄 Ask AI** with a sample loaded)
- [ ] Video plays end-to-end on a fresh tab (incognito, logged out)
- [ ] `docs/SPEC-v0.2.md`, `docs/KAGGLE-WRITEUP.md`, `docs/VIDEO-SCRIPT.md`, `LICENSE` all visible on GitHub
- [ ] Modal volume `eom-sft-out` still has `eom-sft-adapter-gemma4-v5`
- [ ] No secrets in any committed file (`grep -r 'sk-or-' . --include='*.py' --include='*.md'` returns nothing)

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
