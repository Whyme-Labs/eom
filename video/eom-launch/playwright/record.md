# Playwright shots — recorder notes

The 10 PNGs in `shots/` are the dynamic assets for the video. Each is a
real 1920×1080 screenshot (or DOM-region screenshot) of the live demo at
https://eom-demo.swmengappdev.workers.dev, taken with the Playwright MCP
adapter against the production Worker.

| File | Source | Captured state |
|---|---|---|
| `01-hero.png` | full viewport | landing page, no sample loaded, hero strip + empty 5-tab pane |
| `02-newspaper.png` | full viewport | paris-2024-olympics sample loaded, Newspaper tab active, lede + body rendered |
| `03-harness.png` | full viewport | Harness tab: green PASS badge + metrics table (n_blocks=10, tier_a=1, …, tier_ab_tokens=167) + H13/H14 notes |
| `04-pack.png` | full viewport | Context-pack tab: tokenised pack with `[src:…]` citation markers in gold |
| `05-json.png` | full viewport | JSON tab: pretty-printed EOM doc |
| `06-ask-empty.png` | full viewport | Ask AI tab, paris-2024-q3 selected ("Which countries topped…"), reference-answer caption visible, Run button armed |
| `07-ask-results.png` | full viewport | Both columns populated: raw 1,468 tok / pack 438 tok (−70%), full answers |
| `08-ask-headline.png` | element-only (#ask-headline) | "Compression: pack uses 0.30× the tokens of raw (438 vs 1,468, 70% reduction). Same model, same question." |
| `09-askgrid.png` | element-only (.askgrid) | The two side-by-side cards alone, no surrounding chrome |
| `10-hero-bindings.png` | element-only (.hero) | Top hero strip with the live binding tags `R2, D1 (32 docs), KV, AI, OpenRouter` |

## To re-record

The shots are deterministic given the same sample (`news/paris-2024-olympics`),
the same preset question (paris-2024-q3), and a working OpenRouter key.

```
# preflight
curl -s https://eom-demo.swmengappdev.workers.dev/api/health | jq

# warm the KV cache so the pack render is fast
curl -s -X POST https://eom-demo.swmengappdev.workers.dev/api/render/pack \
  -H 'content-type: application/json' \
  -d '{"id":"news/paris-2024-olympics","budget":1500}' > /dev/null
```

Then drive Playwright (via MCP, or via a vanilla `@playwright/test` script)
through the sequence:

1. Resize viewport to 1920×1080.
2. Navigate to `https://eom-demo.swmengappdev.workers.dev` and capture `01-hero.png`.
3. `localStorage.setItem('eom.openrouter_key', '<your-sk-or-key>')`, reload.
4. Patch the masked display: `document.getElementById('apikey-mask').textContent = 'sk-or-***'`
   so the screenshots don't include any chars of the real key suffix.
5. Select `news/paris-2024-olympics` from `#sample-picker`. Wait 3.5s for
   parallel API calls (newspaper, pack, validate) to settle.
6. Capture `02-newspaper.png`.
7. Click `.tab[data-tab='harness']`, capture `03-harness.png`.
8. Click `.tab[data-tab='pack']`, capture `04-pack.png`.
9. Click `.tab[data-tab='json']`, capture `05-json.png`.
10. Click `.tab[data-tab='ask']`. Select the q3 preset, capture `06-ask-empty.png`.
11. Click `#ask-run`. Wait ~7s for both OpenRouter calls.
12. Capture viewport `07-ask-results.png`. Then capture element-only
    `#ask-headline` → `08-ask-headline.png` and `.askgrid` → `09-askgrid.png`.
13. Capture element-only `.hero` → `10-hero-bindings.png`.
14. Clear localStorage: `localStorage.removeItem('eom.openrouter_key')`.

## Why these specific shots

The storyboard names every one of these files in the asset-audit table.
They feed Beats 2-4 of the video:

- Beat 2 (Outbound): 02, 03 (and the eom-mark + tagline already visible in 01)
- Beat 3 (Inbound): 06, 07, 08, 09 (+ the numeric callouts overlaid in
  the composition)
- Beat 4 (Architecture): 10 (the hero's `bindings:` meta line is the
  CF-suite story — magnified and used as the chip strip)
