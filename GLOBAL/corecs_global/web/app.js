// CoreCS Global web UI. Vanilla JS, hash routing:
//   #/            new tutor (landing + form)
//   #/retry/<id>  the same form, pre-filled from a failed/cancelled job, to edit and re-run it
//   #/jobs        job history
//   #/jobs/<id>   live progress (+ syllabus review when the job pauses for it)
//   #/deploy/<id> download + deployment guide

const $view = document.getElementById("view");
let OPTIONS = null;
let cleanup = () => {};

// ---------------------------------------------------------------- utilities
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const rtl = (s) => /[֐-׿؀-ۿ]/.test(s || "") ? ' dir="auto"' : "";

function fmtDuration(s) {
  if (s == null) return "—";
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  return `${m}m ${s % 60 ? (s % 60) + "s" : ""}`.trim();
}
const fmtBytes = (b) => b > 1 << 30 ? `${(b / (1 << 30)).toFixed(2)} GB` : `${(b / (1 << 20)).toFixed(1)} MB`;
const fmtTime = (t) => new Date(t * 1000).toLocaleString();

function toast(msg, ms = 3500) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.hidden = true), ms);
}

// Tiny Markdown subset for the deployment guide: paragraphs, "- " lists, **bold**, `code`, links.
function md(text) {
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(https?:\/\/[^\s<)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  return text.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => l.startsWith("- "))) return `<ul>${lines.map((l) => `<li>${inline(l.slice(2))}</li>`).join("")}</ul>`;
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  }).join("");
}

function token() { try { return localStorage.getItem("global_token") || ""; } catch { return ""; } }

// Backend base URL. config.js sets window.GLOBAL_API_URL: "" when the FastAPI server serves this
// page itself (same origin), the GLOBAL_API_URL env var on Vercel. Otherwise: local dev backend.
const API_BASE = (window.GLOBAL_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// For URLs the browser loads without fetch() (EventSource, download links), which can't carry the
// Authorization header and, cross-origin, don't get the backend's cookie either.
function apiUrl(path) {
  if (!token()) return API_BASE + path;
  return `${API_BASE}${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token())}`;
}

async function api(path, opts = {}) {
  // ngrok's free tier answers browser requests with an HTML warning page unless this header is set.
  const headers = { "ngrok-skip-browser-warning": "1", ...(opts.headers || {}) };
  if (token()) headers.Authorization = `Bearer ${token()}`;
  let res;
  // no-store: otherwise the browser can replay a cached ngrok "endpoint offline" page after the
  // server is back, and retrying never actually reaches it.
  try { res = await fetch(API_BASE + path, { cache: "no-store", ...opts, headers }); }
  // The browser reports a CORS rejection exactly like a network failure, so name both causes.
  catch {
    throw new Error(API_BASE && new URL(API_BASE, location.href).origin !== location.origin
      ? `Could not reach the CoreCS Global server at ${API_BASE}. Check that the server and the ngrok tunnel are running, and that GLOBAL_CORS_ORIGINS on the server allows ${location.origin} (exactly, no trailing slash). Then try again.`
      : "Could not reach the CoreCS Global server. Check your connection and that the server is running, then try again.");
  }
  if (res.status === 401) {
    const t = prompt("This CoreCS Global server requires an access token:");
    if (t) {
      try { localStorage.setItem("global_token", t); } catch {}
      document.cookie = `global_token=${encodeURIComponent(t)}; path=/; SameSite=Strict`;
      return api(path, opts);
    }
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("json")) return res.json();
  // Every /api/ route answers JSON. Anything else is a page from something in between (ngrok's
  // warning/offline page, or this frontend's own index.html when GLOBAL_API_URL is wrong).
  if (path.startsWith("/api/")) throw new Error(`The backend at ${API_BASE || location.origin} did not return JSON. Check that the server and the ngrok tunnel are running and that GLOBAL_API_URL points at the tunnel (https://…).`);
  return res.text();
}

// Responses are checked before rendering so a malformed one shows an error instead of crashing.
const arr = (v) => (Array.isArray(v) ? v : []);
function checkOptions(o) {
  const ok = o && typeof o === "object" && o.defaults && ["base_models", "languages", "quant_types", "teacher_models"].every((k) => Array.isArray(o[k]));
  if (!ok) throw new Error("The backend sent an unexpected /api/options response. Is GLOBAL_API_URL pointing at a CoreCS Global server?");
  return o;
}

// ---------------------------------------------------------------- router
let routeSeq = 0;   // only the newest route() call may render (retry clicks, hash changes)

async function route() {
  cleanup();
  cleanup = () => {};
  const seq = ++routeSeq;
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  window.scrollTo(0, 0);
  try {
    if (!OPTIONS) {
      const opts = checkOptions(await api("/api/options"));   // not cached unless valid, so a retry can succeed
      if (seq !== routeSeq) return;
      OPTIONS = opts;
      document.getElementById("demo-banner").hidden = !OPTIONS.demo_mode;
    }
    if (parts[0] === "jobs" && parts[1]) return await viewJob(parts[1]);
    if (parts[0] === "jobs") return await viewJobs();
    if (parts[0] === "deploy" && parts[1]) return await viewDeploy(parts[1]);
    if (parts[0] === "retry" && parts[1]) return await viewRetry(parts[1]);
    return viewNew();
  } catch (e) {
    if (seq !== routeSeq) return;
    $view.innerHTML = `<div class="alert err" role="alert">${esc(e.message)}
      <div class="hint" style="margin-top:6px">Last attempt: ${new Date().toLocaleTimeString()}</div>
      <div style="margin-top:10px"><button type="button" class="btn btn-ghost" id="retry-route">Try again</button></div></div>`;
    document.getElementById("retry-route").onclick = retryRoute;
  }
}

// "Try again": clear the error, show that a new attempt is running, then re-run the whole route
// (options + the current view's data) from scratch.
async function retryRoute() {
  $view.innerHTML = `<div class="card empty"><p class="hint">Connecting to the CoreCS Global server…</p></div>`;
  await route();
  if ($view.querySelector("#retry-route")) toast("Still can't reach the server.");
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- new tutor / edit & re-run
async function viewRetry(id) {
  const inputs = await api(`/api/jobs/${id}/inputs`);
  if (inputs.replaced_by) {
    $view.innerHTML = `<div class="alert warn">This job was already edited and re-run as
      <a href="#/jobs/${esc(inputs.replaced_by)}">job ${esc(inputs.replaced_by)}</a>.</div>`;
    return;
  }
  if (!["failed", "cancelled"].includes(inputs.status)) {
    $view.innerHTML = `<div class="alert err">Only failed or cancelled jobs can be edited and re-run.
      <a href="#/jobs/${esc(id)}">Back to the job</a></div>`;
    return;
  }
  viewNew({ id, ...inputs });
}

// `prefill` (from viewRetry) = {id, config, pasted_text, files, status, error}.
function viewNew(prefill = null) {
  const d = OPTIONS.defaults;
  const models = OPTIONS.base_models.map((m) => `<option value="${esc(m.id)}">${esc(m.label)}${m.gated ? " · gated" : ""}</option>`).join("");
  const hero = prefill ? `
  <section class="hero">
    <h1>Edit &amp; re-run</h1>
    <p>Fix the inputs below, then re-run. This starts a new job from scratch; the
       ${esc(prefill.status)} job is replaced and hidden from <a href="#/jobs">My jobs</a>.</p>
  </section>
  <div class="alert warn"><b>Job ${esc(prefill.id)} ${esc(prefill.status)}${prefill.error ? ":" : "."}</b>
    ${prefill.error ? `<span${rtl(prefill.error)}>${esc(prefill.error)}</span>` : ""}
    <div class="hint" style="margin-top:6px">API keys are never stored after a job ends, so enter yours again.
      <a href="#/jobs/${esc(prefill.id)}">Back to the job</a></div></div>` : `
  <section class="hero">
    <h1>Turn your syllabi into a private AI tutor</h1>
    <p>Upload the syllabi of your degree program. CoreCS Global writes a syllabus-grounded training set with
       your own API key, fine-tunes an open model on it, and hands you a bundle your students run
       offline on their own computers.</p>
    <div class="flow"><span>1 · Syllabus ingestion</span><span>2 · Dataset generation</span><span>3 · Fine-tuning</span><span>4 · Packaging &amp; GGUF</span><span>5 · Deployment guide</span></div>
  </section>`;
  $view.innerHTML = `${hero}
  <div id="form-error" class="alert err" role="alert" tabindex="-1" hidden></div>
  <form id="f" class="grid" autocomplete="off" novalidate>
    <div>
      <div class="card">
        <h2><span class="num">1</span> Your program</h2>
        <div class="row">
          <div><label for="program">Degree / program name</label>
            <input id="program" name="program_name" type="text" required placeholder="e.g. B.Sc. Computer Science"></div>
          <div><label for="language">Tutor answers in</label>
            <select id="language" name="language">${OPTIONS.languages.map((l) => `<option ${l === d.language ? "selected" : ""}>${esc(l)}</option>`).join("")}<option value="__other">Other…</option></select>
            <input id="language-other" name="language_other" type="text" placeholder="Language name" hidden style="margin-top:8px"></div>
        </div>
      </div>

      <div class="card">
        <h2><span class="num">2</span> Syllabi</h2>
        <p class="hint">One file per course, or one document describing the whole program. PDF, DOCX, TXT or MD,
           up to ${OPTIONS.max_upload_mb} MB each. Scanned PDFs need OCR first.</p>
        <div id="drop" class="drop" tabindex="0">Drop syllabus files here, or <b>click to browse</b></div>
        <input id="file" name="files" type="file" multiple accept=".pdf,.docx,.txt,.md" hidden>
        <ul id="files" class="files"></ul>
        <label for="pasted">Or paste syllabus text <span class="hint">— separate courses with a line containing only <code>---</code></span></label>
        <textarea id="pasted" name="pasted_text" placeholder="Course: Data Structures&#10;Week 1: Arrays and linked lists&#10;Week 2: Stacks and queues&#10;...&#10;---&#10;Course: Algorithms&#10;..."></textarea>
        <label class="check"><input id="review" name="review_syllabi" type="checkbox" ${d.review_syllabi ? "checked" : ""}>
          <span>Let me review the extracted courses and topics before data generation <span class="hint">(recommended)</span></span></label>
      </div>

      <div class="card">
        <h2><span class="num">3</span> Model &amp; credentials</h2>
        <label for="base">Base model</label>
        <select id="base" name="base_model">${models}<option value="__custom">Custom Hugging Face model…</option></select>
        <input id="base-custom" name="base_model_custom" type="text" placeholder="org/model-name (must have a chat template)" hidden style="margin-top:8px">
        <div class="row">
          <div>
            <label for="key">${OPTIONS.teacher_models[0]} API key <span class="hint">(OpenAI)</span></label>
            <input id="key" name="api_key" type="password" placeholder="${OPTIONS.demo_mode ? "not needed in demo mode" : "sk-..."}" ${OPTIONS.demo_mode ? "" : "required"}>
          </div>
          <div>
            <label for="hf">Hugging Face token <span class="hint">(only for gated models)</span></label>
            <input id="hf" name="hf_token" type="password" placeholder="hf_...">
          </div>
        </div>
        <p class="hint" style="margin-top:8px">Keys are encrypted at rest, never shown again, and deleted as soon as
           the stage that needs them finishes. The teacher model is fixed to <code>${esc(OPTIONS.teacher_models.join(", "))}</code>.</p>

        <details>
          <summary>Advanced training settings</summary>
          <label for="n">Examples per course: <b id="n-val">${d.examples_per_course}</b></label>
          <input id="n" name="examples_per_course" type="range" min="50" max="2000" step="50" value="${d.examples_per_course}">
          <div class="row">
            <div><label for="epochs">Epochs</label><input id="epochs" name="epochs" type="number" min="1" max="5" value="${d.epochs}"></div>
            <div><label for="lora">LoRA rank</label><input id="lora" name="lora_r" type="number" min="4" max="64" step="4" value="${d.lora_r}"></div>
            <div><label for="lr">Learning rate</label><input id="lr" name="learning_rate" type="number" min="0.000001" max="0.001" step="any" value="${d.learning_rate}"></div>
            <div><label for="seq">Max sequence length</label><input id="seq" name="max_seq_len" type="number" min="256" max="4096" step="64" value="${d.max_seq_len}"></div>
            <div><label for="quant">Quantization</label><select id="quant" name="quant">${OPTIONS.quant_types.map((q) => `<option ${q === d.quant ? "selected" : ""}>${q}</option>`).join("")}</select></div>
            <div><label for="conc">Parallel courses (API)</label><input id="conc" name="generation_concurrency" type="number" min="1" max="16" value="${d.generation_concurrency}"></div>
          </div>
          <label class="check"><input id="adapter" name="include_adapter" type="checkbox" ${d.include_adapter ? "checked" : ""}><span>Include the LoRA adapter in the bundle</span></label>
        </details>
      </div>
    </div>

    <aside class="sticky">
      <div class="card est">
        <h2>Estimate</h2>
        <div id="est"><p class="hint">Add a syllabus to see an estimate.</p></div>
      </div>
      <button class="btn btn-primary btn-lg" id="go" type="submit">${prefill ? "Re-run with changes" : "Build my tutor"}</button>
      <p class="hint" style="margin-top:10px">You can close this page; the job keeps running and appears under <a href="#/jobs">My jobs</a>.</p>
    </aside>
  </form>`;

  const files = [];
  const kept = prefill ? [...arr(prefill.files)] : [];   // files already on the server from the job being re-run
  const $ = (id) => document.getElementById(id);
  const fileInput = $("file"), drop = $("drop");
  const submitLabel = prefill ? "Re-run with changes" : "Build my tutor";

  const renderFiles = () => {
    $("files").innerHTML =
      kept.map((f, i) => `<li><span>${esc(f.name)} <span class="hint">${fmtBytes(f.bytes)} · from the previous run</span></span><button type="button" data-k="${i}" aria-label="Remove">×</button></li>`).join("") +
      files.map((f, i) => `<li><span>${esc(f.name)} <span class="hint">${fmtBytes(f.size)}</span></span><button type="button" data-i="${i}" aria-label="Remove">×</button></li>`).join("");
    estimateSoon();
  };
  const addFiles = (list) => {
    for (const f of list) {
      if (!/\.(pdf|docx|txt|md)$/i.test(f.name)) { toast(`${f.name}: unsupported file type`); continue; }
      if (!files.some((x) => x.name === f.name && x.size === f.size)) files.push(f);
      const dup = kept.findIndex((x) => x.name === f.name);   // a re-upload replaces the old copy
      if (dup >= 0) kept.splice(dup, 1);
    }
    renderFiles();
  };
  drop.onclick = () => fileInput.click();
  drop.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && fileInput.click();
  fileInput.onchange = () => { addFiles(fileInput.files); fileInput.value = ""; };
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); };
  $("files").onclick = (e) => {
    if (e.target.dataset.i) { files.splice(+e.target.dataset.i, 1); renderFiles(); }
    if (e.target.dataset.k) { kept.splice(+e.target.dataset.k, 1); renderFiles(); }
  };
  $("base").onchange = () => { $("base-custom").hidden = $("base").value !== "__custom"; estimateSoon(); };
  $("language").onchange = () => { $("language-other").hidden = $("language").value !== "__other"; };
  $("n").oninput = () => { $("n-val").textContent = $("n").value; };

  const pastedBlocks = () => $("pasted").value.split(/^\s*-{3,}\s*$/m).filter((b) => b.trim()).length;
  const config = () => ({
    program_name: $("program").value.trim(),
    language: $("language").value === "__other" ? $("language-other").value.trim() : $("language").value,
    base_model: $("base").value === "__custom" ? $("base-custom").value.trim() : $("base").value,
    examples_per_course: +$("n").value,
    review_syllabi: $("review").checked,
    epochs: +$("epochs").value,
    lora_r: +$("lora").value,
    learning_rate: +$("lr").value,
    max_seq_len: +$("seq").value,
    quant: $("quant").value,
    generation_concurrency: +$("conc").value,
    include_adapter: $("adapter").checked,
  });

  let timer = null;
  function estimateSoon() { clearTimeout(timer); timer = setTimeout(runEstimate, 300); }
  async function runEstimate() {
    const n = files.length + kept.length + pastedBlocks();
    if (!n) { $("est").innerHTML = '<p class="hint">Add a syllabus to see an estimate.</p>'; return; }
    try {
      const cfg = config();
      if (cfg.base_model === "") return;
      const e = await api("/api/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ config: cfg, n_sources: n }) });
      const s = e.stage_seconds;
      $("est").innerHTML = `
        <div class="big">${fmtDuration(e.total_seconds)}</div><p class="hint">end-to-end, assuming ~${n} course(s)</p>
        <dl>
          <dt>Training examples</dt><dd>${e.rows.toLocaleString()}</dd>
          <dt>Teacher API cost</dt><dd>≈ $${e.api_cost_usd.toFixed(2)}</dd>
          <dt>Dataset generation</dt><dd>${fmtDuration(s.generate)}</dd>
          <dt>Fine-tuning (${e.train_steps.toLocaleString()} steps)</dt><dd>${fmtDuration(s.train)}</dd>
          <dt>Packaging</dt><dd>${fmtDuration(s.package)}</dd>
          <dt>Model download size</dt><dd>≈ ${e.gguf_gb} GB</dd>
        </dl>
        <small>Rough figures. The exact course count is known after ingestion, and the live ETA then replaces this estimate.</small>`;
    } catch (err) { $("est").innerHTML = `<p class="hint">${esc(err.message)}</p>`; }
  }
  document.getElementById("f").addEventListener("input", estimateSoon);

  if (prefill) {
    const c = prefill.config;
    const hasOption = (sel, v) => [...$(sel).options].some((o) => o.value === v);
    $("program").value = c.program_name || "";
    if (hasOption("language", c.language)) $("language").value = c.language;
    else { $("language").value = "__other"; $("language-other").value = c.language || ""; $("language-other").hidden = false; }
    if (hasOption("base", c.base_model)) $("base").value = c.base_model;
    else { $("base").value = "__custom"; $("base-custom").value = c.base_model || ""; $("base-custom").hidden = false; }
    $("n").value = c.examples_per_course; $("n-val").textContent = $("n").value;
    $("review").checked = !!c.review_syllabi;
    $("epochs").value = c.epochs;
    $("lora").value = c.lora_r;
    $("lr").value = c.learning_rate;
    $("seq").value = c.max_seq_len;
    if (hasOption("quant", c.quant)) $("quant").value = c.quant;
    $("conc").value = c.generation_concurrency;
    $("adapter").checked = !!c.include_adapter;
    $("pasted").value = prefill.pasted_text || "";
    // Show the advanced settings if any of them differ from the defaults, so the lecturer sees them.
    const adv = ["examples_per_course", "epochs", "lora_r", "learning_rate", "max_seq_len", "quant", "generation_concurrency", "include_adapter"];
    if (adv.some((k) => c[k] !== d[k])) document.querySelector("#f details").open = true;
    renderFiles();
  }

  // The form is `novalidate`: native validation silently aborts submit when the offending control is
  // hidden (e.g. inside the collapsed "Advanced" <details>), so validation lives here and mirrors
  // JobConfig.validate() on the server.
  const $err = $("form-error");
  const showError = (title, items = []) => {
    $err.innerHTML = `<b>${esc(title)}</b>${items.length ? `<ul>${items.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : ""}`;
    $err.hidden = false;
    $err.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const clearErrors = () => {
    $err.hidden = true;
    document.querySelectorAll('#f [aria-invalid="true"]').forEach((el) => el.removeAttribute("aria-invalid"));
  };

  function validate() {
    const errs = [];
    const bad = (id, msg) => errs.push({ id, msg });
    const num = (id) => ($(id).value.trim() === "" ? NaN : Number($(id).value));
    const intIn = (id, label, lo, hi) => {
      const v = num(id);
      if (!Number.isInteger(v) || v < lo || v > hi) bad(id, `${label} must be a whole number between ${lo} and ${hi}.`);
    };
    if (!$("program").value.trim()) bad("program", "Degree / program name is required.");
    if ($("language").value === "__other" && !$("language-other").value.trim()) bad("language-other", "Enter the language the tutor should answer in.");
    if ($("base").value === "__custom" && !/^[\w.-]+\/[\w.-]+$/.test($("base-custom").value.trim())) bad("base-custom", "Custom base model must be a Hugging Face id like org/model-name.");
    if (!OPTIONS.demo_mode && !$("key").value.trim()) bad("key", "An OpenAI API key is required to generate the training data.");
    if (!files.length && !kept.length && !pastedBlocks()) bad("drop", "Add at least one syllabus file or paste syllabus text.");
    intIn("epochs", "Epochs", 1, 5);
    intIn("lora", "LoRA rank", 4, 64);
    intIn("seq", "Max sequence length", 256, 4096);
    intIn("conc", "Parallel courses (API)", 1, 16);
    const lr = num("lr");
    if (!Number.isFinite(lr) || lr < 1e-6 || lr > 1e-3) bad("lr", "Learning rate must be a number between 0.000001 and 0.001.");
    return errs;
  }

  const form = document.getElementById("f");
  form.addEventListener("input", (e) => e.target.removeAttribute?.("aria-invalid"));

  form.onsubmit = async (e) => {
    e.preventDefault();
    clearErrors();
    const errs = validate();
    if (errs.length) {
      errs.forEach(({ id }) => $(id).setAttribute("aria-invalid", "true"));
      showError("Please fix the following before building your tutor:", errs.map((x) => x.msg));
      const first = $(errs[0].id);
      const det = first.closest("details");
      if (det) det.open = true;
      first.focus({ preventScroll: true });
      return;
    }
    const fd = new FormData();
    fd.append("config", JSON.stringify(config()));
    fd.append("api_key", $("key").value);
    fd.append("hf_token", $("hf").value);
    fd.append("pasted_text", $("pasted").value);
    files.forEach((f) => fd.append("files", f, f.name));
    if (prefill) {
      fd.append("retry_of", prefill.id);
      fd.append("keep_files", JSON.stringify(kept.map((f) => f.file)));
    }
    const btn = $("go");
    btn.disabled = true;
    btn.textContent = OPTIONS.demo_mode ? "Starting…" : "Verifying API key…";
    try {
      const { id } = await api("/api/jobs", { method: "POST", body: fd });
      location.hash = `#/jobs/${id}`;
    } catch (err) {
      showError("Could not start the job.", [err.message || "Unknown error."]);
      btn.disabled = false;
      btn.textContent = submitLabel;
    }
  };
}

// ---------------------------------------------------------------- job list
async function viewJobs() {
  const res = await api("/api/jobs");
  if (!Array.isArray(res)) throw new Error("The backend sent an unexpected job list. Reload the page to try again.");
  const jobs = res.filter((j) => j && j.config);
  if (!jobs.length) {
    $view.innerHTML = `<div class="card empty"><h2>No jobs yet</h2><p>Build your first tutor to see it here.</p><a class="btn btn-primary" href="#/">New tutor</a></div>`;
    return;
  }
  $view.innerHTML = `<div class="card"><h2>My jobs</h2><table>
    <thead><tr><th>Program</th><th>Status</th><th>Progress</th><th>Started</th><th></th></tr></thead>
    <tbody>${jobs.map((j) => `<tr>
      <td><b${rtl(j.config.program_name)}>${esc(j.config.program_name)}</b><div class="hint">${esc(j.config.base_model)}</div></td>
      <td><span class="pill ${esc(j.status)}">${esc(String(j.status ?? "").replace("_", " "))}</span></td>
      <td style="min-width:140px"><div class="bar sm"><i style="width:${Math.round((j.fraction || 0) * 100)}%"></i></div></td>
      <td class="hint">${fmtTime(j.created_at)}</td>
      <td class="row-acts">${["failed", "cancelled"].includes(j.status) ? `<a href="#/retry/${j.id}">Edit &amp; re-run</a>` : ""}
        <a href="${j.status === "succeeded" ? "#/deploy/" : "#/jobs/"}${j.id}">Open →</a></td></tr>`).join("")}
    </tbody></table></div>`;
}

// ---------------------------------------------------------------- live job view
async function viewJob(id) {
  let job = await api(`/api/jobs/${id}`);
  if (!job?.config) throw new Error(`The backend sent an unexpected response for job ${id}.`);
  $view.innerHTML = `
    <div class="card">
      <div class="job-head">
        <div><h1 style="font-size:1.6rem"${rtl(job.config.program_name)}>${esc(job.config.program_name)}</h1>
          <div class="hint">${esc(job.config.base_model)} · ${esc(job.config.language)} · job ${esc(id)}</div></div>
        <div class="actions" id="acts"></div>
      </div>
      <div class="overall"><div class="bar"><i id="ov-bar" style="width:0"></i></div></div>
      <div class="meta" id="meta"></div>
      <div id="alert"></div>
    </div>
    <div id="review"></div>
    <div class="grid">
      <div class="card"><h2>Pipeline</h2><ul class="stages" id="stages"></ul></div>
      <div class="card"><h2>Activity</h2><div class="log" id="log"></div></div>
    </div>`;

  const $log = document.getElementById("log");
  const appendEvent = (ev) => {
    if (!ev.message && ev.kind !== "stage") return;
    const line = document.createElement("div");
    const t = new Date(ev.ts * 1000).toLocaleTimeString();
    const text = ev.kind === "stage" ? `▸ ${ev.stage}: ${ev.status}${ev.message ? " — " + ev.message : ""}` : ev.message;
    line.className = ev.level === "error" ? "err" : ev.kind === "stage" ? "stg" : "";
    line.innerHTML = `<time>${t}</time>${esc(text)}`;
    const stick = $log.scrollTop + $log.clientHeight >= $log.scrollHeight - 30;
    $log.appendChild(line);
    while ($log.childNodes.length > 1500) $log.removeChild($log.firstChild);
    if (stick) $log.scrollTop = $log.scrollHeight;
  };

  let reviewShown = false;
  const render = (j) => {
    if (!j?.config) return;   // ignore a malformed poll/stream update rather than crash the view
    j = { ...j, stages: arr(j.stages) };
    job = j;
    document.getElementById("ov-bar").style.width = `${Math.round(j.fraction * 100)}%`;
    const running = j.stages.find((s) => s.status === "running");
    const usage = j.stages.map((s) => s.usage?.cost_usd || 0).reduce((a, b) => a + b, 0);
    document.getElementById("meta").innerHTML = `
      <span>Status <span class="pill ${j.status}">${esc(j.status.replace("_", " "))}</span></span>
      <span>Overall <b>${Math.round(j.fraction * 100)}%</b></span>
      <span>Current stage <b>${esc(running ? running.label : j.status === "queued" && j.next_stage ? "waiting for a worker" : "—")}</b></span>
      <span>Time remaining <b>${j.status === "awaiting_review" ? "paused for your review" : j.status === "succeeded" ? "finished" : ["failed", "cancelled"].includes(j.status) ? "—" : fmtDuration(j.eta_s)}</b></span>
      ${usage ? `<span>API spend so far <b>$${usage.toFixed(2)}</b></span>` : ""}`;

    document.getElementById("stages").innerHTML = j.stages.map((s, i) => {
      const st = s.status || "pending";
      const icon = st === "done" ? "✓" : st === "failed" ? "!" : st === "cancelled" ? "–" : i + 1;
      const showBar = st === "running" && s.total;
      return `<li class="stage ${st}">
        <div class="dot">${icon}</div>
        <div><b>${esc(s.label)}</b><div class="msg"${rtl(s.message)}>${esc(s.message || (st === "pending" ? "Waiting" : st))}</div>
          ${showBar ? `<div class="bar sm"><i style="width:${Math.round((s.fraction || 0) * 100)}%"></i></div>` : ""}</div>
        <div class="eta">${st === "running" ? (s.eta_s != null ? fmtDuration(s.eta_s) + " left" : "estimating…") : st === "done" && s.finished_at && s.started_at ? fmtDuration(s.finished_at - s.started_at) : ""}</div>
      </li>`;
    }).join("");

    const acts = document.getElementById("acts");
    acts.innerHTML = "";
    const btn = (label, cls, fn) => { const b = document.createElement("button"); b.className = `btn ${cls}`; b.textContent = label; b.onclick = fn; acts.appendChild(b); return b; };
    if (["queued", "running", "awaiting_review", "staging"].includes(j.status)) {
      btn(j.cancel_requested ? "Cancelling…" : "Cancel", "btn-danger", async () => {
        if (!confirm("Cancel this job? Completed stages are kept and you can retry later.")) return;
        try { render(await api(`/api/jobs/${id}/cancel`, { method: "POST" })); } catch (e) { toast(e.message); }
      }).disabled = j.cancel_requested;
    }
    if (["failed", "cancelled"].includes(j.status) && !j.replaced_by) {
      // Retry = edit the inputs first. Resuming unchanged only makes sense once some stage has
      // succeeded (e.g. a training crash after generation); before that it would just fail again.
      btn("Edit & re-run", "btn-primary", () => { location.hash = `#/retry/${id}`; });
      if (j.stages.some((s) => s.status === "done")) btn("Resume unchanged", "btn-ghost", () => retryDialog(id, j, render));
    }
    if (j.stages.find((s) => s.name === "generate")?.status === "done") {
      const a = document.createElement("a"); a.className = "btn btn-ghost"; a.textContent = "Download dataset"; a.href = apiUrl(`/api/jobs/${id}/dataset`); acts.appendChild(a);
    }
    if (j.status === "succeeded") { const a = document.createElement("a"); a.className = "btn btn-primary"; a.textContent = "Download & deploy →"; a.href = `#/deploy/${id}`; acts.appendChild(a); }

    const al = document.getElementById("alert");
    al.innerHTML = j.replaced_by ? `<div class="alert warn">This job was edited and re-run as <a href="#/jobs/${esc(j.replaced_by)}">job ${esc(j.replaced_by)}</a>.</div>`
      : j.error ? `<div class="alert err"><b>${j.status === "failed" ? "Failed" : "Error"}:</b> ${esc(j.error)}</div>`
      : j.status === "awaiting_review" ? `<div class="alert warn">Review the courses and topics extracted from your syllabi below, then approve to start data generation.</div>` : "";

    if (j.status === "awaiting_review" && !reviewShown) { reviewShown = true; showReview(id, j.config.examples_per_course); }
    if (j.status !== "awaiting_review" && reviewShown) { reviewShown = false; document.getElementById("review").innerHTML = ""; }
  };
  render(job);

  // Live updates via Server-Sent Events; falls back to polling if the stream drops.
  let es = null, poll = null, done = false;
  const connect = () => {
    es = new EventSource(apiUrl(`/api/jobs/${id}/stream?after=0`));
    es.addEventListener("event", (m) => appendEvent(JSON.parse(m.data)));
    es.addEventListener("snapshot", (m) => {
      const j = JSON.parse(m.data);
      const was = job.status;
      render(j);
      if (j.status === "succeeded" && was !== "succeeded") {
        toast("Your tutor is ready!");
        setTimeout(() => { if (location.hash === `#/jobs/${id}`) location.hash = `#/deploy/${id}`; }, 1800);
      }
    });
    es.addEventListener("end", () => { done = true; es.close(); });
    es.onerror = () => {
      es.close();
      if (done) return;
      clearInterval(poll);
      poll = setInterval(async () => { try { render(await api(`/api/jobs/${id}`)); } catch {} }, 5000);
      setTimeout(() => { if (!done) { clearInterval(poll); connect(); } }, 15000);
    };
  };
  connect();
  cleanup = () => { done = true; es && es.close(); clearInterval(poll); };
}

function retryDialog(id, job, render) {
  const needsKey = !OPTIONS.demo_mode && ["ingest", "generate"].some((s) => arr(job.stages).find((x) => x.name === s)?.status !== "done");
  const back = document.createElement("div");
  back.className = "modal-back";
  back.innerHTML = `<div class="modal"><h2>Resume job</h2>
    <p class="hint">The job resumes from the first unfinished stage with the same inputs, reusing every checkpoint.
      To change the syllabus or settings, use <b>Edit &amp; re-run</b> instead.</p>
    ${needsKey ? `<label for="rk">API key <span class="hint">(needed again for data generation)</span></label><input id="rk" type="password" placeholder="sk-...">` : ""}
    <div class="actions" style="margin-top:18px"><button class="btn btn-primary" id="rgo">Resume</button><button class="btn btn-ghost" id="rno">Close</button></div></div>`;
  document.body.appendChild(back);
  back.querySelector("#rno").onclick = () => back.remove();
  back.querySelector("#rgo").onclick = async () => {
    try {
      const body = needsKey ? { api_key: back.querySelector("#rk").value } : {};
      render(await api(`/api/jobs/${id}/retry`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
      back.remove();
      toast("Resume queued.");
      route();
    } catch (e) { toast(e.message, 6000); }
  };
}

async function showReview(id, perCourse) {
  const box = document.getElementById("review");
  let data;
  try { data = await api(`/api/jobs/${id}/syllabi`); } catch (e) { box.innerHTML = `<div class="alert err">${esc(e.message)}</div>`; return; }
  const courses = arr(data?.courses).map((c) => ({ ...c, topics: arr(c.topics), out_of_scope: arr(c.out_of_scope) }));
  if (!courses.length) { box.innerHTML = `<div class="alert warn">No courses were extracted from the syllabi. Cancel the job and use Edit &amp; re-run to fix the inputs.</div>`; return; }
  const courseHtml = (c, i) => `<div class="course" data-i="${i}">
      <div class="course-head"><input type="text" class="cname" value="${esc(c.course_name)}"${rtl(c.course_name)}>
        <button type="button" class="btn btn-ghost rm" title="Remove course">Remove</button></div>
      <div class="row">
        <div><label>Topics <span class="hint">(one per line — each question is anchored to one)</span></label>
          <textarea class="topics"${rtl(c.topics.join(" "))} style="min-height:180px">${esc(c.topics.join("\n"))}</textarea></div>
        <div><label>Out of scope <span class="hint">(never mentioned in this course)</span></label>
          <textarea class="oos"${rtl(c.out_of_scope.join(" "))}>${esc(c.out_of_scope.join("\n"))}</textarea>
          <label>Short English label</label><input type="text" class="domain" value="${esc(c.domain)}">
          <label>Special style note <span class="hint">(optional)</span></label><input type="text" class="style" value="${esc(c.style_note || "")}"></div>
      </div>
      <p class="hint" style="margin:6px 0 0">From: ${esc(c.source || "")}</p>
    </div>`;
  box.innerHTML = `<div class="card"><h2>Review extracted courses (${courses.length})</h2>
    <p class="hint">Edit names and topics, remove anything that isn't a course, then approve. Generation creates
      ${perCourse.toLocaleString()} examples for each course you keep.</p>
    <div id="courses">${courses.map(courseHtml).join("")}</div>
    <div class="actions"><button class="btn btn-primary" id="approve">Approve &amp; start generation</button></div></div>`;
  box.querySelector("#courses").onclick = (e) => { if (e.target.classList.contains("rm")) e.target.closest(".course").remove(); };
  box.querySelector("#approve").onclick = async (e) => {
    const lines = (s) => s.split("\n").map((x) => x.trim()).filter(Boolean);
    const courses = [...box.querySelectorAll(".course")].map((el) => ({
      course_name: el.querySelector(".cname").value.trim(),
      domain: el.querySelector(".domain").value.trim(),
      topics: lines(el.querySelector(".topics").value),
      out_of_scope: lines(el.querySelector(".oos").value),
      style_note: el.querySelector(".style").value.trim() || null,
    }));
    e.target.disabled = true;
    try {
      await api(`/api/jobs/${id}/syllabi`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ courses }) });
      await api(`/api/jobs/${id}/approve`, { method: "POST" });
      toast("Approved — generating the dataset.");
    } catch (err) { toast(err.message, 6000); e.target.disabled = false; }
  };
}

// ---------------------------------------------------------------- deploy guide
async function viewDeploy(id) {
  const [job, guide] = await Promise.all([api(`/api/jobs/${id}`), api(`/api/jobs/${id}/deploy`)]);
  if (!job?.config || !guide || typeof guide !== "object") throw new Error(`The backend sent an unexpected response for job ${id}.`);
  job.stages = arr(job.stages);
  guide.sections = arr(guide.sections).map((sec) => ({ ...sec, body: sec.body || "", commands: arr(sec.commands) }));
  let user = "";
  try { user = localStorage.getItem("hf_user") || ""; } catch {}
  const fill = (s) => (user ? s.split(guide.placeholder).join(user) : s);
  const renderGuide = () => {
    document.getElementById("guide").innerHTML = guide.sections.map((sec, i) => `
      <section class="guide-sec"><h3><span class="num">${i + 1}</span>${esc(sec.title)}</h3>
        <div class="body">${md(fill(sec.body))}</div>
        ${sec.commands.map((c) => `<div class="cmd"><div class="lbl">${esc(c.label)}</div><pre><code>${esc(fill(c.code))}</code><button type="button" data-copy>Copy</button></pre></div>`).join("")}
      </section>`).join("");
  };
  $view.innerHTML = `
    <div class="card success">
      <div class="check-big">✓</div>
      <h1${rtl(guide.display_name)}>${esc(guide.display_name)} is ready</h1>
      <p class="hint">Fine-tuned from ${esc(job.config.base_model)} · ${job.stages.find((s) => s.name === "generate")?.rows?.toLocaleString() ?? ""} training examples</p>
      <div class="actions" style="justify-content:center;margin-top:16px">
        ${job.bundle ? `<a class="btn btn-primary" href="${esc(apiUrl(`/api/jobs/${id}/download`))}">Download bundle (${fmtBytes(job.bundle.bytes)})</a>` : `<span class="alert err">Bundle file missing on the server.</span>`}
        <a class="btn btn-ghost" href="#/jobs/${id}">Pipeline details</a>
      </div>
    </div>
    <div class="grid">
      <div class="card">
        <h2>Deployment guide</h2>
        <label for="hfu">Your Hugging Face username <span class="hint">— fills in the commands below</span></label>
        <input id="hfu" type="text" placeholder="e.g. prof-cohen" value="${esc(user)}" style="margin-bottom:20px">
        <div id="guide"></div>
      </div>
      <aside>
        <div class="card"><h2>Sample answers</h2>
          ${arr(guide.samples).map((s) => `<div class="sample"><div class="q"${rtl(s.question)}>${esc(s.question)}</div><div class="a"${rtl(s.answer)}>${esc(s.answer)}</div></div>`).join("") || '<p class="hint">No samples recorded.</p>'}
          <p class="hint">Generated by your fine-tuned model on held-out questions.</p>
        </div>
      </aside>
    </div>`;
  renderGuide();
  document.getElementById("hfu").oninput = (e) => {
    user = e.target.value.trim().replace(/[^\w.-]/g, "");
    try { localStorage.setItem("hf_user", user); } catch {}
    renderGuide();
  };
  document.getElementById("guide").onclick = async (e) => {
    if (!e.target.hasAttribute("data-copy")) return;
    const code = e.target.parentElement.querySelector("code").textContent;
    try { await navigator.clipboard.writeText(code); e.target.textContent = "Copied"; setTimeout(() => (e.target.textContent = "Copy"), 1500); }
    catch { toast("Copy failed — select the text manually."); }
  };
}

route();
