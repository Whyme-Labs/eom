-- EOM D1 schema (binding: DB, database: eom-data).
--
-- Applied via `wrangler d1 execute eom-data --remote --file schema.sql`.
--
-- Three tables, normalised:
--   docs           — registry of compileable docs (mirrors public/samples/manifest.json)
--   qsets          — benchmark questions + reference answers per doc
--   bench_results  — per-row inbound-benchmark output (raw vs pack, per question)

CREATE TABLE IF NOT EXISTS docs (
  id          TEXT PRIMARY KEY,        -- "policy/gdpr"
  type        TEXT NOT NULL,           -- "policy"
  slug        TEXT NOT NULL,           -- "gdpr"
  title       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qsets (
  q_id        TEXT PRIMARY KEY,        -- "gdpr-q1"
  doc_id      TEXT NOT NULL REFERENCES docs(id),
  question    TEXT NOT NULL,
  reference   TEXT NOT NULL,
  position    INTEGER NOT NULL         -- order within the doc
);
CREATE INDEX IF NOT EXISTS qsets_doc_idx ON qsets(doc_id);

CREATE TABLE IF NOT EXISTS bench_results (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          TEXT NOT NULL,
  doc_id          TEXT NOT NULL REFERENCES docs(id),
  question_id     TEXT NOT NULL REFERENCES qsets(q_id),
  mode            TEXT NOT NULL CHECK (mode IN ('raw', 'pack')),
  model           TEXT NOT NULL,
  input_tokens    INTEGER NOT NULL,
  output_tokens   INTEGER NOT NULL,
  latency_ms      INTEGER NOT NULL,
  citations       INTEGER NOT NULL DEFAULT 0,
  judge_score     INTEGER,             -- 0/1/2, NULL when judge skipped
  judge_rationale TEXT,
  recorded_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS bench_results_run_idx ON bench_results(run_id);
CREATE INDEX IF NOT EXISTS bench_results_doc_idx ON bench_results(doc_id);
