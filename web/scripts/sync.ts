/**
 * Sync data/gold/ and data/bench/qsets.json from the repo root into the
 * Workers static-asset directory (public/). Run before deploy.
 *
 *   bun run scripts/sync.ts
 *
 * Generates:
 *   public/samples/<type>/<slug>.md
 *   public/samples/<type>/<slug>.eom.json
 *   public/samples/manifest.json   (list of {id, type, slug, title})
 *   public/qsets.json              (verbatim copy of data/bench/qsets.json)
 *   public/bench-results.json      (latest benchmark summary, if present)
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "fs";
import { dirname, join } from "path";

const REPO_ROOT = join(import.meta.dir, "..", "..");
const GOLD_DIR = join(REPO_ROOT, "data", "gold");
const BENCH_DIR = join(REPO_ROOT, "data", "bench");
const OUT_DIR = join(import.meta.dir, "..", "public");

interface ManifestEntry { id: string; type: string; slug: string; title: string; }

function readTitle(mdPath: string): string {
  const text = readFileSync(mdPath, "utf8");
  const m = text.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1]! : "";
}

function ensureDir(p: string) {
  mkdirSync(p, { recursive: true });
}

function syncSamples(): ManifestEntry[] {
  const samplesOut = join(OUT_DIR, "samples");
  ensureDir(samplesOut);
  const manifest: ManifestEntry[] = [];
  for (const entry of readdirSync(GOLD_DIR)) {
    const typeDir = join(GOLD_DIR, entry);
    if (!statSync(typeDir).isDirectory()) continue;
    for (const f of readdirSync(typeDir).filter((f) => f.endsWith(".eom.json"))) {
      const slug = f.replace(".eom.json", "");
      const mdSrc = join(typeDir, `${slug}.md`);
      const eomSrc = join(typeDir, f);
      if (!existsSync(mdSrc)) continue;
      const dstDir = join(samplesOut, entry);
      ensureDir(dstDir);
      copyFileSync(mdSrc, join(dstDir, `${slug}.md`));
      copyFileSync(eomSrc, join(dstDir, `${slug}.eom.json`));
      manifest.push({
        id: `${entry}/${slug}`,
        type: entry,
        slug,
        title: readTitle(mdSrc),
      });
    }
  }
  manifest.sort((a, b) =>
    a.type.localeCompare(b.type) || a.slug.localeCompare(b.slug),
  );
  writeFileSync(
    join(samplesOut, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  return manifest;
}

function syncQsets() {
  const src = join(BENCH_DIR, "qsets.json");
  if (existsSync(src)) {
    ensureDir(OUT_DIR);
    copyFileSync(src, join(OUT_DIR, "qsets.json"));
  }
}

function syncBenchResults() {
  const resultsDir = join(BENCH_DIR, "results");
  if (!existsSync(resultsDir)) return;
  // Pick the lexicographically latest .json (run-id is a timestamp).
  const candidates = readdirSync(resultsDir).filter((f) => f.endsWith(".json")).sort();
  const latest = candidates[candidates.length - 1];
  if (latest) {
    copyFileSync(join(resultsDir, latest), join(OUT_DIR, "bench-results.json"));
  }
}

ensureDir(OUT_DIR);
const manifest = syncSamples();
syncQsets();
syncBenchResults();
console.log(`synced ${manifest.length} sample docs into ${OUT_DIR}`);
