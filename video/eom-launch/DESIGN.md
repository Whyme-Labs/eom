# Design System

## Overview

EOM's visual personality is editorial-newsroom-meets-typewriter: cream paper,
ink black, antique gold accent. The product positions itself as a *formal IR
for human-AI documents*, and the design leans into that thesis by borrowing
print-newspaper typography (Georgia serif for the brand mark) and a calm,
type-driven layout with almost no chrome. One-page app: a left sidebar holds
sample picker, direction explainer, and OpenRouter key form; the main column
is a 5-tab interface (Newspaper / Context pack / JSON / Harness / Ask AI).
The overall feel is "long-read journal" — print-warm, low-contrast, monospaced
where it counts (code, token counts, JSON), serif where the brand speaks.

## Colors

- **Surface Cream**: `#FAFAF7` — page background, soft paper feel.
- **Card White**: `#FFFFFF` — sidebar, raised content.
- **Ink Black**: `#111111` — primary text, hero h1, button fills.
- **Muted Slate**: `#5A5A5A` — secondary text, sidebar uppercase labels, captions.
- **Antique Gold**: `#B8860B` — accent: active-tab underline, links, source-span markers, headline rule.
- **Gold Dim**: `#D4AF3733` — accent-tint backgrounds (reference panels, key callouts).
- **Rule Bone**: `#E0DDD5` — 1px borders between sidebar and content, between sections.
- **Code Paper**: `#F3F1EC` — inline `<code>` background, codepane panel.
- **Pass Green**: `#2E7D32` — harness PASS badge.
- **Fail Red**: `#B71C1C` — harness FAIL badge, error banners.

## Typography

- **Serif (brand)**: Georgia 700. Hero `📰 EOM` headline (27px) and the
  newspaper render's `<h1>`. The serif is the visual cue for "this is editorial."
- **Sans (system)**: `-apple-system / BlinkMacSystemFont / Inter` at 400 / 600 /
  700. Body copy (15px / 1.55 line-height), buttons, sidebar labels.
- **Monospace (UI signal)**: `ui-monospace / SF Mono / Menlo`. JSON panes,
  context-pack output, token counts, `[src:block-id]` citation markers,
  binding tags in the hero meta (`R2, D1 (32 docs), KV, AI, OpenRouter`).
- **Hierarchy**: hero h1 27px / 700 · sidebar h2 13.8px / 700 / uppercase /
  letter-spaced · tab labels 0.95em / 600 · body 15px / 1.55 · caption 0.85em.

## Elevation

Flat. No shadows. Depth comes from:
- 1px hairline rules (`#E0DDD5`) between hero / sidebar / content / footer.
- A 3px gold underline on the active tab.
- A 4px gold left-bar on the compression-headline callout.
- The newspaper render is shown inside an inset iframe with a 1px rule and
  matched cream background — same paper, different page.

## Components

- **Editorial hero**: Georgia 700 brand mark, sans tagline, muted meta line
  surfacing the live CF binding tags (`bindings: R2, D1 (32 docs), KV, AI,
  OpenRouter`) — that meta line is the architecture diagram, embedded.
- **Sidebar sample picker**: native `<select>` with type-prefixed options
  ("paper — Attention Is All You Need"), grouped implicitly by alphabetical
  order. Direction explainer paragraph + OpenRouter key form.
- **Tab strip**: 5 buttons (Newspaper / Context pack / JSON / Harness / Ask AI),
  underline-only active state, no boxes.
- **Newspaper iframe**: outbound projection — hero / lede / body / archive,
  Georgia headline, Verdana-class body, source-span hover.
- **Context-pack codepane**: monospaced text with section headers (`## headline`,
  `## lead`, `## evidence`), `[src:block-id]` citation markers in gold.
- **JSON pane**: pretty-printed EOM, monospaced.
- **Harness report**: pass/fail badge + failures list + metrics table + notes.
- **Ask AI side-by-side**: two raised cards side-by-side. Left card "Raw
  markdown" shows input tokens (large monospaced number) / latency / answer.
  Right card "EOM context-pack" shows the same with a `-69%` delta on the
  input-tokens metric. Below: a single bold compression-headline line with
  a 4px gold left-bar.

## Do's and Don'ts

### Do's
- Lead with Georgia serif for any "editorial" moment — brand mark, scene
  titles, newspaper render.
- Quote the *exact* hero tagline ("Two-way wire format between humans and
  models, with attention budgets and source grounding built in.") at least once
  in the video.
- Treat numbers as visual hooks: `52% reduction`, `0.31×`, `1.47 → 0.93`,
  `32 docs`. Display large in monospaced.
- Use the antique gold sparingly — accent only. Most of the frame should be
  cream + ink.
- Let whitespace breathe. The demo is calm; the video should be calm.

### Don'ts
- No bright primary colors, no neon, no dark mode. The whole video stays in
  cream + ink + gold.
- No drop shadows. No glassmorphism. No bevels. Depth comes from rules and
  type weight only.
- No fast / jittery transitions. Motion should feel like turning a newspaper
  page — slow fade-throughs, soft slides.
- Don't decorate the newspaper render. Just show it; the rendered HTML is
  itself the demo content.
- Don't overlay text on the newspaper render itself — overlay only on cream
  background frames or between scenes.
