/* ================================================================
   Smart Signal Dashboard — JavaScript
   Handles: training control, policy inspector, YOLO upload,
            results loading, SSE progress streaming
   ================================================================ */

const API = "";  // same origin (served by FastAPI)
let mode = "fast";
let eventSource = null;

// ━━━━━━━━━━━━━━━━━━━━━ INIT ━━━━━━━━━━━━━━━━━━━━━
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("episodesSlider").addEventListener("input", e => {
    document.getElementById("episodesVal").textContent = e.target.value;
  });
  refreshStatus();
  loadResults();
});

// ━━━━━━━━━━━━━━━━━━━━━ STATUS ━━━━━━━━━━━━━━━━━━━━━
async function refreshStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) return;
    const d = await r.json();

    // Q-table nonzero %
    if (d.q_table_stats) {
      document.getElementById("valQValues").textContent =
        `${d.q_table_stats.nonzero_pct}%`;
    }

    // Best params
    const ql = d.best_params?.q_learning;
    if (ql) {
      setEl("mpAlpha", ql.alpha);
      setEl("mpGamma", ql.gamma);
      setEl("mpEps",   ql.epsilon);
    }

    // Status dot
    if (d.training_running) {
      setStatusDot("running", "Training…");
    } else if (d.model_trained) {
      setStatusDot("done", "Model Ready");
    } else {
      setStatusDot("idle", "Ready");
    }
  } catch (_) {}
}

// ━━━━━━━━━━━━━━━━━━━━━ RESULTS ━━━━━━━━━━━━━━━━━━━━━
async function loadResults() {
  try {
    const r = await fetch(`${API}/api/results`);
    if (!r.ok) return;
    const data = await r.json();
    renderResultsTable(data);
    updateKPIs(data);
  } catch (_) {}
}

function updateKPIs(results) {
  const fixed = results["Fixed Timer"]?.mean_wait_time;
  const ql    = results["Q-Learning"]?.mean_wait_time;
  if (fixed && ql) {
    const pct = ((1 - ql / fixed) * 100).toFixed(1);
    document.getElementById("valWaitReduction").textContent = `${pct}%`;
  }
}

function renderResultsTable(results) {
  const tbody = document.getElementById("resultsBody");
  const fixed = results["Fixed Timer"]?.mean_wait_time ?? 1;
  const rows  = Object.entries(results).map(([name, m]) => {
    const wait = m.mean_wait_time;
    const pct  = name === "Fixed Timer" ? 0 : (1 - wait / fixed) * 100;
    const barW = Math.max(5, 100 - (wait / fixed) * 100);
    const impCls  = pct > 0 ? "imp-good" : "imp-base";
    const impText = pct > 0 ? `↓ ${pct.toFixed(1)}%` : "baseline";
    return `<tr>
      <td><b>${name}</b></td>
      <td style="font-family:var(--mono)">${wait.toFixed(2)}s</td>
      <td class="improvement ${impCls}">${impText}</td>
      <td><div class="result-bar" style="width:${barW}px"></div></td>
    </tr>`;
  });
  tbody.innerHTML = rows.join("");
}

// ━━━━━━━━━━━━━━━━━━━━━ TRAINING ━━━━━━━━━━━━━━━━━━━━━
function setMode(m) {
  mode = m;
  document.getElementById("btnFast").classList.toggle("active", m === "fast");
  document.getElementById("btnSumo").classList.toggle("active", m === "sumo");
}

async function startTraining() {
  const episodes = document.getElementById("episodesSlider").value;
  const tune     = document.getElementById("chkTune").checked;
  const noSumo   = mode === "fast";

  const btn = document.getElementById("btnTrain");
  btn.disabled = true;
  btn.textContent = "⏳ Starting…";
  setStatusDot("running", "Training…");
  setEl("trainStatusTag", "Running");
  document.getElementById("trainStatusTag").style.background = "rgba(16,185,129,0.15)";

  try {
    const params = new URLSearchParams({ episodes, tune, no_sumo: noSumo });
    await fetch(`${API}/api/train/start?${params}`, { method: "POST" });

    // Show progress bar
    document.getElementById("progressWrap").style.display = "block";
    document.getElementById("liveStats").style.display    = "flex";

    // SSE progress stream
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`${API}/api/train/progress`);
    eventSource.onmessage = e => {
      const d = JSON.parse(e.data);
      if (d.done) {
        onTrainingDone(episodes);
        return;
      }
      const pct = (d.episode / d.total * 100).toFixed(0);
      document.getElementById("progressFill").style.width = `${pct}%`;
      document.getElementById("progressLabel").textContent =
        `Episode ${d.episode} / ${d.total}`;
      setEl("liveReward", (d.reward > 0 ? "+" : "") + d.reward.toFixed(1));
      setEl("liveWait",   d.wait.toFixed(1) + "s");
      setEl("liveEps",    d.epsilon.toFixed(3));
    };
    eventSource.onerror = () => { onTrainingDone(episodes); };
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "▶ Start Training";
    alert("Error: " + err.message);
  }
}

function onTrainingDone(episodes) {
  if (eventSource) { eventSource.close(); eventSource = null; }
  document.getElementById("btnTrain").disabled = false;
  document.getElementById("btnTrain").textContent = "▶ Start Training";
  document.getElementById("progressFill").style.width = "100%";
  document.getElementById("progressLabel").textContent = `Complete — ${episodes} episodes`;
  setEl("trainStatusTag", "Done ✓");
  document.getElementById("trainStatusTag").style.background = "rgba(59,130,246,0.15)";
  setStatusDot("done", "Model Ready");

  // Reload results and refresh plots
  loadResults();
  reloadPlots();
  refreshStatus();
}

function reloadPlots() {
  const plotIds = ["plotTraining", "plotHeatmap", "plotEps", "plotTD"];
  const plotSrcs = [
    "/api/plot/training_curves", "/api/plot/q_table_heatmap",
    "/api/plot/epsilon_decay",   "/api/plot/td_error"
  ];
  plotIds.forEach((id, i) => {
    const img = document.getElementById(id);
    if (img) img.src = plotSrcs[i] + "?t=" + Date.now();
  });
}

// ━━━━━━━━━━━━━━━━━━━━━ POLICY INSPECTOR ━━━━━━━━━━━━━━━━━━━━━
function updateDensityLabel(val) {
  document.getElementById("densityLabel").textContent = val;
  const badge = document.getElementById("densityBadge");
  const v = parseInt(val);
  if (v >= 70) {
    badge.textContent = "HIGH 🔴";
    badge.style.cssText = "background:rgba(239,68,68,0.15);color:#fca5a5;border:1px solid rgba(239,68,68,0.3);padding:3px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;";
  } else if (v >= 35) {
    badge.textContent = "MED 🟡";
    badge.style.cssText = "background:rgba(245,158,11,0.15);color:#fcd34d;border:1px solid rgba(245,158,11,0.3);padding:3px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;";
  } else {
    badge.textContent = "LOW 🟢";
    badge.style.cssText = "background:rgba(16,185,129,0.15);color:#6ee7b7;border:1px solid rgba(16,185,129,0.3);padding:3px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;";
  }
}

async function queryPolicy(state) {
  updateDensityLabel(state);
  try {
    const r = await fetch(`${API}/api/policy/${state}`);
    if (!r.ok) {
      document.getElementById("policyResult").innerHTML =
        '<div class="pr-empty">Model not trained yet — run training first</div>';
      return;
    }
    const d = await r.json();
    renderPolicyResult(d);
  } catch (_) {
    document.getElementById("policyResult").innerHTML =
      '<div class="pr-empty">API not running. Start with: uvicorn api.server:app</div>';
  }
}

function renderPolicyResult(d) {
  const pr = document.getElementById("policyResult");
  pr.innerHTML = `
    <div class="pr-action">
      <div class="pr-dur">${d.duration}s</div>
      <div>
        <div style="font-weight:600;font-size:0.9rem">Optimal Green Phase</div>
        <div class="pr-info">Action index: ${d.action} | State: ${d.state}</div>
        <div class="pr-val">V(s) = ${d.value.toFixed(4)}</div>
      </div>
    </div>`;

  // Q-value bar chart
  const maxQ   = Math.max(...d.q_values.map(Math.abs), 0.01);
  const qBars  = document.getElementById("qBars");
  const actions = [10,20,30,40,50,60,70,80,90,100];
  qBars.innerHTML = d.q_values.map((q, i) => {
    const h    = Math.abs(q) / maxQ * 100;
    const best = i === d.action ? "best" : "";
    return `<div class="q-bar ${best}" style="height:${Math.max(4,h)}%"
              title="Action=${actions[i]}s Q=${q.toFixed(3)}">
              <span class="q-bar-label">${actions[i]}</span>
            </div>`;
  }).join("");
}

// ━━━━━━━━━━━━━━━━━━━━━ YOLO ━━━━━━━━━━━━━━━━━━━━━
function handleDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) runYOLO(file);
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) runYOLO(file);
}

async function runYOLO(file) {
  const yoloResultsDiv = document.getElementById("yoloResults");
  yoloResultsDiv.innerHTML = '<div class="yolo-ph">🔍 Running YOLOv8 detection…</div>';

  const formData = new FormData();
  formData.append("file", file);

  try {
    const r = await fetch(`${API}/api/detect`, { method: "POST", body: formData });
    if (!r.ok) {
      const err = await r.json();
      yoloResultsDiv.innerHTML = `<div class="yolo-ph" style="color:#fca5a5">Error: ${err.detail}</div>`;
      return;
    }
    const d = await r.json();
    renderYOLOResults(d);
  } catch (e) {
    yoloResultsDiv.innerHTML =
      `<div class="yolo-ph" style="color:#fca5a5">API error — make sure the server is running</div>`;
  }
}

function renderYOLOResults(d) {
  const maxCount = Math.max(...Object.values(d.counts), 1);
  const rows = Object.entries(d.counts).map(([cls, cnt]) => {
    const barW = Math.max(4, (cnt / maxCount) * 100);
    return `<div class="det-row">
      <span class="det-class">${cls}</span>
      <div class="det-bar-wrap"><div class="det-bar" style="width:${barW}%"></div></div>
      <span class="det-count">${cnt}</span>
    </div>`;
  }).join("");

  const level = d.density >= 70 ? "🔴 HIGH" : d.density >= 35 ? "🟡 MED" : "🟢 LOW";

  document.getElementById("yoloResults").innerHTML = `
    ${rows}
    <div class="yolo-density-row">
      <span class="yolo-density-label">RL State / Density ${level}</span>
      <span class="yolo-density-val">${d.density}<span style="font-size:1rem;font-weight:400;color:var(--text-muted)">/100</span></span>
    </div>
    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">
      ${d.num_detections} vehicles detected · Raw density: ${d.raw_density}
    </div>`;

  // Show annotated image
  if (d.annotated_b64) {
    const preview = document.getElementById("yoloPreview");
    document.getElementById("yoloAnnotated").src = "data:image/jpeg;base64," + d.annotated_b64;
    preview.style.display = "block";
  }
}

// ━━━━━━━━━━━━━━━━━━━━━ HELPERS ━━━━━━━━━━━━━━━━━━━━━
function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setStatusDot(state, label) {
  const dot = document.querySelector(".dot");
  const lbl = document.getElementById("statusLabel");
  if (dot) { dot.className = "dot dot-" + state; }
  if (lbl) lbl.textContent = label;
}
