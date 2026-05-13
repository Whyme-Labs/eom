/**
 * Build transcript.json from SCRIPT.md (captions-only video, no TTS).
 *
 * Each caption line gets a {text, start, end} entry. Durations within a
 * beat are weighted by word count, then scaled to fit the beat window
 * declared in STORYBOARD.md. A 0.3s padding sits between captions for
 * the page-turn transition, and the last caption of each beat ends 0.3s
 * before the next beat starts.
 *
 *   bun run scripts/build-transcript.ts
 *     -> writes transcript.json next to SCRIPT.md
 */

import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const ROOT = join(import.meta.dir, "..");

// Beat windows, must match STORYBOARD.md.
const BEATS: Array<{ id: string; start: number; end: number; name: string }> = [
  { id: "beat-1-hook",         start:   0, end:  15, name: "Hook" },
  { id: "beat-2-outbound",     start:  15, end:  50, name: "Outbound" },
  { id: "beat-3-inbound",      start:  50, end: 115, name: "Inbound" },
  { id: "beat-4-architecture", start: 115, end: 150, name: "Architecture" },
  { id: "beat-5-finetune",     start: 150, end: 170, name: "Fine-tune" },
  { id: "beat-6-close",        start: 170, end: 180, name: "Close" },
];

const MIN_CAPTION_HOLD = 1.6;   // seconds — never below this
const INTER_CAPTION_PAUSE = 0.25;

function wordCount(s: string): number {
  return s.trim().split(/\s+/).filter(Boolean).length;
}

function parseScript(): Record<string, string[]> {
  const text = readFileSync(join(ROOT, "SCRIPT.md"), "utf8");
  // Each beat starts with `## Beat N — …`. Captions are markdown paragraphs:
  // text separated by blank lines. Hard wraps within a paragraph (e.g. a
  // line split for readability in the source file) join with a space.
  const lines = text.split("\n");
  const beats: Record<string, string[]> = {};
  let currentBeatIdx: number | null = null;
  let buf: string[] = [];

  const flush = () => {
    // Always drain the buffer; only emit if we're inside a beat.
    const text = buf.join(" ").replace(/\s+/g, " ").trim();
    buf = [];
    if (currentBeatIdx === null) return;
    if (!text) return;
    if (text.startsWith("#") || text.startsWith(">") || text.startsWith("-")) return;
    const beat = BEATS[currentBeatIdx];
    if (!beat) return;
    (beats[beat.id] ??= []).push(text);
  };

  for (const line of lines) {
    const m = line.match(/^##\s+Beat\s+(\d+)\b/i);
    if (m) {
      flush();
      buf = [];  // belt-and-suspenders for any pre-beat exposition
      currentBeatIdx = Number(m[1]) - 1;
      const beat = BEATS[currentBeatIdx];
      if (!beat) throw new Error(`Beat ${m[1]} out of range`);
      beats[beat.id] = [];
      continue;
    }
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    if (trimmed.startsWith("#")) continue;
    buf.push(trimmed);
  }
  flush();
  return beats;
}

function buildTranscript() {
  const beats = parseScript();
  const out: Array<{ text: string; start: number; end: number; beat: string }> = [];

  for (const beat of BEATS) {
    const captions = beats[beat.id] ?? [];
    if (captions.length === 0) {
      console.warn(`  ${beat.id}: no captions parsed`);
      continue;
    }
    const beatDur = beat.end - beat.start;
    // Reserve INTER_CAPTION_PAUSE between adjacent captions and 0.3s tail.
    const totalGapTime = INTER_CAPTION_PAUSE * (captions.length - 1) + 0.3;
    const availableForCaptions = beatDur - totalGapTime;
    const totalWords = captions.reduce((s, c) => s + wordCount(c), 0);

    let cursor = beat.start;
    for (let i = 0; i < captions.length; i++) {
      const c = captions[i]!;
      const weight = wordCount(c) / totalWords;
      let dur = Math.max(MIN_CAPTION_HOLD, availableForCaptions * weight);
      // Bound by remaining beat budget so the last caption never overruns.
      const remaining = beat.end - 0.3 - cursor;
      if (remaining < MIN_CAPTION_HOLD) {
        console.warn(`  WARN ${beat.id}: caption ${i + 1}/${captions.length} ` +
                     `has only ${remaining.toFixed(2)}s left; clamping`);
      }
      if (dur > remaining) dur = Math.max(0.4, remaining);
      const start = cursor;
      const end = Number((cursor + dur).toFixed(3));
      out.push({ text: c, start: Number(start.toFixed(3)), end, beat: beat.id });
      cursor = end + INTER_CAPTION_PAUSE;
    }
    // log
    console.log(`  ${beat.id} (${beatDur}s, ${captions.length} captions)`);
  }

  writeFileSync(
    join(ROOT, "transcript.json"),
    JSON.stringify({ captions: out, total_seconds: 180 }, null, 2) + "\n",
  );
  console.log(`\nwrote transcript.json (${out.length} caption rows, 180s total)`);
}

buildTranscript();
