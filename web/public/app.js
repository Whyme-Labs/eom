// EOM live demo frontend — vanilla ES module.
// Wires the static asset manifest + Worker API into a 5-tab UI.

const $ = (sel) => document.querySelector(sel);
const tabs = document.querySelectorAll(".tab");
const panels = {
  newspaper: $("#panel-newspaper"),
  pack: $("#panel-pack"),
  json: $("#panel-json"),
  harness: $("#panel-harness"),
  ask: $("#panel-ask"),
};
const samplePicker = $("#sample-picker");
const sampleMeta = $("#sample-meta");
const newspaperFrame = $("#newspaper-frame");
const newspaperDefault = panels.newspaper.querySelector("[data-default]");
const packOut = $("#pack-out");
const packBudget = $("#pack-budget");
const packBudgetValue = $("#pack-budget-value");
const jsonOut = $("#json-out");
const harnessOut = $("#harness-out");
const questionPicker = $("#question-picker");
const customQuestion = $("#custom-question");
const askBudget = $("#ask-budget");
const askBudgetValue = $("#ask-budget-value");
const askRun = $("#ask-run");
const askReference = $("#ask-reference");
const askHeadline = $("#ask-headline");
const askRawAnswer = $("#ask-raw-answer");
const askRawMetrics = $("#ask-raw-metrics");
const askRawContext = $("#ask-raw-context");
const askPackAnswer = $("#ask-pack-answer");
const askPackMetrics = $("#ask-pack-metrics");
const askPackContext = $("#ask-pack-context");

let state = {
  manifest: [],
  qsets: {},
  currentId: null,
  currentEom: null,
};

// --- API helpers ---------------------------------------------------------

async function fetchJson(path, init) {
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
async function fetchText(path, init) {
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.text();
}

async function loadManifest() {
  const data = await fetchJson("/api/samples");
  state.manifest = data.samples || [];
  state.manifestSource = data.source || "?";
  samplePicker.innerHTML =
    `<option value="">— choose —</option>` +
    state.manifest.map((m) =>
      `<option value="${m.id}">${m.type} — ${escape(m.title || m.slug)}</option>`,
    ).join("");
}

async function loadQsetsForCurrent() {
  if (!state.currentId) return;
  try {
    const data = await fetchJson(`/api/qsets/${state.currentId}`);
    state.qsetSource = data.source || "?";
    return data.questions || [];
  } catch {
    return [];
  }
}

// --- tab switching -------------------------------------------------------

tabs.forEach((t) => {
  t.addEventListener("click", () => {
    tabs.forEach((x) => x.classList.toggle("active", x === t));
    Object.entries(panels).forEach(([k, p]) =>
      p.hidden = (k !== t.dataset.tab),
    );
  });
});

// --- sample load ---------------------------------------------------------

samplePicker.addEventListener("change", async () => {
  const id = samplePicker.value;
  if (!id) return;
  state.currentId = id;
  sampleMeta.textContent = "loading…";
  await Promise.all([
    loadJsonPanel(id),
    loadNewspaperPanel(id),
    loadPackPanel(id),
    loadHarnessPanel(id),
    loadAskPresets(id),
  ]);
  const m = state.manifest.find((x) => x.id === id);
  const tag = state.manifestSource === "d1" ? "D1" : "static";
  sampleMeta.innerHTML = m
    ? `${escape(m.type)} · ${escape(m.title || m.slug)} <small style="opacity:0.6">(via ${tag})</small>`
    : id;
});

async function loadJsonPanel(id) {
  // Fetch the eom.json via /api/render/* round-trip would re-validate; for
  // raw JSON display we fetch the static asset directly (R2-backed in prod
  // via a /samples/* rewrite is a Phase-4 nicety).
  const eom = await fetchJson(`/samples/${id}.eom.json`);
  state.currentEom = eom;
  jsonOut.textContent = JSON.stringify(eom, null, 2);
}

async function loadNewspaperPanel(id) {
  const html = await fetchText("/api/render/newspaper", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  newspaperFrame.src = url;
  newspaperFrame.hidden = false;
  if (newspaperDefault) newspaperDefault.style.display = "none";
}

async function loadPackPanel(id) {
  await renderPackText(id);
}
packBudget.addEventListener("input", () => {
  packBudgetValue.textContent = packBudget.value;
  if (state.currentId) renderPackText(state.currentId);
});
async function renderPackText(id) {
  const text = await fetchText("/api/render/pack", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id, budget: parseInt(packBudget.value, 10) }),
  });
  packOut.textContent = text;
}

async function loadHarnessPanel(id) {
  const report = await fetchJson("/api/render/validate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ id }),
  });
  const badge = report.passed
    ? `<span class="badge pass">PASS</span> all H1–H12 OK`
    : `<span class="badge fail">FAIL</span> ${report.failures.length} failure(s)`;
  const fails = report.failures.length === 0
    ? ""
    : `<h3>Failures</h3><ul class="fail-list">` +
      report.failures.map((f) =>
        `<li><strong>${f.rule}</strong> ${f.block_id ? `[${f.block_id}] ` : ""}${escape(f.message)}</li>`,
      ).join("") + `</ul>`;
  const metricRows = Object.entries(report.metrics)
    .map(([k, v]) => `<tr><td><code>${k}</code></td><td>${v}</td></tr>`)
    .join("");
  const warns = (report.warnings || [])
    .map((w) => `<li><strong>${w.rule}</strong> ${escape(w.message)}</li>`).join("");
  harnessOut.innerHTML =
    `<p>${badge}</p>${fails}` +
    `<h3>Metrics</h3><table>${metricRows}</table>` +
    (warns ? `<h3>Notes</h3><ul>${warns}</ul>` : "");
}

// --- ask AI --------------------------------------------------------------

askBudget.addEventListener("input", () => {
  askBudgetValue.textContent = askBudget.value;
});

async function loadAskPresets(id) {
  const presets = await loadQsetsForCurrent();
  state.currentQsets = presets;
  questionPicker.innerHTML =
    `<option value="">— preset or use custom —</option>` +
    presets.map((p) =>
      `<option value="${escape(p.q)}" data-ref="${escape(p.ref)}">${escape(p.q)}</option>`,
    ).join("");
  questionPicker.disabled = presets.length === 0;
  customQuestion.value = "";
  askRun.disabled = false;
  askReference.hidden = true;
}

questionPicker.addEventListener("change", () => {
  if (!questionPicker.value) {
    askReference.hidden = true;
    return;
  }
  const opt = questionPicker.options[questionPicker.selectedIndex];
  customQuestion.value = "";
  askReference.innerHTML = `<strong>Reference answer:</strong> ${escape(opt.dataset.ref || "")}`;
  askReference.hidden = false;
});
customQuestion.addEventListener("input", () => {
  if (customQuestion.value) {
    questionPicker.value = "";
    askReference.hidden = true;
  }
});

askRun.addEventListener("click", async () => {
  if (!state.currentId) return;
  const question = customQuestion.value.trim() || questionPicker.value;
  if (!question) return;
  askRun.disabled = true;
  askRun.textContent = "Running…";
  askRawMetrics.textContent = "";
  askPackMetrics.textContent = "";
  askRawAnswer.textContent = "";
  askPackAnswer.textContent = "";
  askHeadline.textContent = "";
  const body = {
    id: state.currentId,
    question,
    budget: parseInt(askBudget.value, 10),
  };
  try {
    const [raw, pack] = await Promise.all([
      fetchJson("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...body, mode: "raw" }),
      }),
      fetchJson("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...body, mode: "pack" }),
      }),
    ]);
    renderAskCell("raw", raw);
    renderAskCell("pack", pack);
    const compression = pack.contextTokens / Math.max(1, raw.contextTokens);
    askHeadline.innerHTML =
      `<strong>Compression:</strong> pack uses <strong>${compression.toFixed(2)}×</strong> ` +
      `the tokens of raw (${pack.contextTokens.toLocaleString()} vs ${raw.contextTokens.toLocaleString()}, ` +
      `${((1 - compression) * 100).toFixed(0)}% reduction). Same model, same question.`;
  } catch (e) {
    askHeadline.textContent = "Error: " + String(e.message || e);
  } finally {
    askRun.disabled = false;
    askRun.textContent = "Run side-by-side →";
  }
});

function renderAskCell(mode, r) {
  const ansEl = mode === "raw" ? askRawAnswer : askPackAnswer;
  const metricsEl = mode === "raw" ? askRawMetrics : askPackMetrics;
  const ctxEl = mode === "raw" ? askRawContext : askPackContext;
  ansEl.textContent = r.answer || "(empty)";
  metricsEl.innerHTML =
    `<span>Input <span class="num">${r.contextTokens.toLocaleString()}</span> tok</span>` +
    `<span>Out <span class="num">${r.outputTokens}</span> tok</span>` +
    `<span>Latency <span class="num">${(r.latencyMs / 1000).toFixed(1)}</span>s</span>`;
  ctxEl.textContent = r.contextPreview || "";
}

// --- util ----------------------------------------------------------------

function escape(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// --- boot ----------------------------------------------------------------

(async () => {
  try {
    await loadManifest();
    // Health check is a nice diagnostic for which CF bindings are wired.
    fetchJson("/api/health").then((h) => {
      const tags = [];
      if (h.bindings?.r2) tags.push("R2");
      if (h.bindings?.d1) tags.push(`D1 (${h.d1_docs ?? "?"} docs)`);
      if (h.bindings?.kv) tags.push("KV");
      if (h.bindings?.ai) tags.push("AI");
      if (h.bindings?.openrouter) tags.push("OpenRouter");
      const el = document.querySelector(".hero .meta");
      if (el && tags.length) {
        el.innerHTML += ` &middot; bindings: <code>${tags.join(", ")}</code>`;
      }
    }).catch(() => {});
  } catch (e) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div style="background:#b71c1c;color:#fff;padding:1em;">Boot error: ${escape(e.message || e)}</div>`,
    );
  }
})();
