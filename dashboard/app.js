/* ================================================================
   Smart Signal Dashboard — JavaScript v3
   Q-Learning focused: YOLO → RL → Q-Table Before/After Diff
   ================================================================ */

const API = "";  // same origin (served by FastAPI)
let mode = "fast";
let eventSource = null;

// Session-level state
let lastDetectedState   = null;
let lastActionIdx       = null;
let lastDecisionId      = null;
let lastQContext        = null;  // Q-table context from detection

// ━━━━━━━━━━━━━━━━━━━━━ INIT ━━━━━━━━━━━━━━━━━━━━━
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("episodesSlider").addEventListener("input", e => {
    document.getElementById("episodesVal").textContent = e.target.value;
  });
  refreshStatus();
  loadResults();
  loadHistory();
  loadQTableSnapshot();

  // Auto-refresh
  setInterval(loadHistory, 30_000);
  setInterval(loadQTableSnapshot, 60_000);
});

// ━━━━━━━━━━━━━━━━━━━━━ STATUS ━━━━━━━━━━━━━━━━━━━━━
async function refreshStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) return;
    const d = await r.json();

    if (d.q_table_stats) {
      setEl("valNonZero", `${d.q_table_stats.nonzero_pct}%`);
    }

    const ql = d.best_params?.q_learning;
    if (ql) {
      setEl("mpAlpha", ql.alpha);
      setEl("mpGamma", ql.gamma);
      setEl("mpEps",   ql.epsilon);
    }

    // Agent config (fallback for hyperparams)
    if (d.agent_config) {
      if (!ql) {
        setEl("mpAlpha", d.agent_config.alpha);
        setEl("mpGamma", d.agent_config.gamma);
        setEl("mpEps",   d.agent_config.epsilon);
      }
    }

    if (d.online_updates !== undefined) {
      setEl("valOnlineUpdates", d.online_updates.toLocaleString());
    }

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

// ━━━━━━━━━━━━━━━━━━━━━ Q-TABLE HEATMAP ━━━━━━━━━━━━━━━━━━━━━
async function loadQTableSnapshot() {
  try {
    const r = await fetch(`${API}/api/q-table-snapshot`);
    if (!r.ok) {
      setEl("qtableStatus", "No model");
      return;
    }
    const d = await r.json();
    renderQTableHeatmap(d);
    setEl("qtableStatus",
      `${d.stats.nonzero_pct}% learned · max Q=${d.stats.max_q}`);
  } catch (_) {
    setEl("qtableStatus", "API offline");
  }
}

function renderQTableHeatmap(data, highlightState = null) {
  const canvas = document.getElementById("qtableCanvas");
  const container = document.getElementById("qtableContainer");
  const ctx = canvas.getContext("2d");

  const numStates = data.num_states;    // 100
  const numActions = data.num_actions;  // 10
  const qt = data.q_table;             // 100 x 10

  // Responsive sizing
  const cw = container.clientWidth - 60;  // leave room for labels
  const ch = Math.max(300, Math.min(500, cw * 0.5));
  canvas.width = cw;
  canvas.height = ch;

  const cellW = cw / numStates;
  const cellH = ch / numActions;

  // Find min/max for color scaling
  let qMin = Infinity, qMax = -Infinity;
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < numActions; a++) {
      const v = qt[s][a];
      if (v < qMin) qMin = v;
      if (v > qMax) qMax = v;
    }
  }
  const qRange = Math.max(qMax - qMin, 0.001);

  // Draw cells
  for (let s = 0; s < numStates; s++) {
    for (let a = 0; a < numActions; a++) {
      const v = qt[s][a];
      const norm = (v - qMin) / qRange;  // 0 to 1

      // Color: red (low) → yellow (mid) → green (high)
      let r, g, b;
      if (norm < 0.5) {
        const t = norm * 2;
        r = Math.round(180 * (1 - t) + 220 * t);
        g = Math.round(40 * (1 - t) + 180 * t);
        b = Math.round(40 * (1 - t) + 30 * t);
      } else {
        const t = (norm - 0.5) * 2;
        r = Math.round(220 * (1 - t) + 16 * t);
        g = Math.round(180 * (1 - t) + 185 * t);
        b = Math.round(30 * (1 - t) + 129 * t);
      }

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(s * cellW, (numActions - 1 - a) * cellH, cellW, cellH);
    }
  }

  // Highlight current state column
  if (highlightState !== null && highlightState >= 1 && highlightState <= 100) {
    const sIdx = highlightState - 1;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(sIdx * cellW, 0, cellW, ch);

    // Draw state indicator
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px Inter";
    ctx.textAlign = "center";
    ctx.fillText(`s=${highlightState}`, sIdx * cellW + cellW / 2, ch - 4);
  }

  // Axis labels (sparse to avoid clutter)
  ctx.fillStyle = "#64748b";
  ctx.font = "10px JetBrains Mono";
  ctx.textAlign = "center";

  // State labels (x-axis) — every 10
  for (let s = 0; s < numStates; s += 10) {
    ctx.fillText(String(s + 1), s * cellW + cellW / 2, ch + 14);
  }

  // Action labels (y-axis)
  const actions = data.actions || [10,20,30,40,50,60,70,80,90,100];
  ctx.textAlign = "right";
  for (let a = 0; a < numActions; a++) {
    ctx.fillText(`${actions[a]}s`,
      -4, (numActions - 1 - a) * cellH + cellH / 2 + 4);
  }

  // Tooltip on hover
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const sIdx = Math.floor(x / cellW);
    const aIdx = numActions - 1 - Math.floor(y / cellH);

    if (sIdx >= 0 && sIdx < numStates && aIdx >= 0 && aIdx < numActions) {
      const tooltip = document.getElementById("qtableTooltip");
      const val = qt[sIdx][aIdx];
      const bestA = data.policy[sIdx].action_idx;
      tooltip.innerHTML = `
        <b>Q(${sIdx+1}, ${aIdx})</b> = ${val.toFixed(4)}<br>
        State: ${sIdx+1} · Action: ${actions[aIdx]}s<br>
        ${aIdx === bestA ? "⭐ Best action for this state" : ""}
      `;
      tooltip.style.display = "block";
      tooltip.style.left = (e.clientX - rect.left + 12) + "px";
      tooltip.style.top = (e.clientY - rect.top - 10) + "px";
    }
  };
  canvas.onmouseleave = () => {
    document.getElementById("qtableTooltip").style.display = "none";
  };
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
  btn.textContent = "⏳ Starting Q-Learning…";
  setStatusDot("running", "Training…");
  setEl("trainStatusTag", "Running");
  document.getElementById("trainStatusTag").style.background = "rgba(16,185,129,0.15)";

  try {
    const params = new URLSearchParams({ episodes, tune, no_sumo: noSumo });
    await fetch(`${API}/api/train/start?${params}`, { method: "POST" });

    document.getElementById("progressWrap").style.display = "block";
    document.getElementById("liveStats").style.display    = "flex";

    if (eventSource) eventSource.close();
    eventSource = new EventSource(`${API}/api/train/progress`);
    eventSource.onmessage = e => {
      const d = JSON.parse(e.data);
      if (d.done) { onTrainingDone(episodes); return; }
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
    btn.textContent = "▶ Start Q-Learning Training";
    alert("Error: " + err.message);
  }
}

function onTrainingDone(episodes) {
  if (eventSource) { eventSource.close(); eventSource = null; }
  document.getElementById("btnTrain").disabled = false;
  document.getElementById("btnTrain").textContent = "▶ Start Q-Learning Training";
  document.getElementById("progressFill").style.width = "100%";
  document.getElementById("progressLabel").textContent = `Complete — ${episodes} episodes`;
  setEl("trainStatusTag", "Done ✓");
  document.getElementById("trainStatusTag").style.background = "rgba(59,130,246,0.15)";
  setStatusDot("done", "Model Ready");

  loadResults();
  reloadPlots();
  refreshStatus();
  loadQTableSnapshot();
}

function reloadPlots() {
  const plotIds  = ["plotTraining", "plotHeatmap", "plotEps", "plotTD"];
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
        <div class="pr-val">V(s) = max<sub>a</sub> Q(s,a) = ${d.value.toFixed(4)}</div>
      </div>
    </div>`;

  const maxQ  = Math.max(...d.q_values.map(Math.abs), 0.01);
  const qBars = document.getElementById("qBars");
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

// ━━━━━━━━━━━━━━━━━━━━━ YOLO + RL LOOP ━━━━━━━━━━━━━━━━━━━━━

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

  activateRLStep(1);
  document.getElementById("rlDecisionPanel").style.display = "none";
  document.getElementById("mdpTuplePanel").style.display = "none";

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

    // Step 1: Detection results
    activateRLStep(1);
    renderYOLOResults(d);

    // Save session state
    lastDetectedState = d.density;
    lastActionIdx     = d.rl_action?.action_idx ?? null;
    lastDecisionId    = d.decision_id ?? null;
    lastQContext      = d.q_table_context ?? null;

    // Step 2: Show RL decision
    if (d.rl_action) {
      activateRLStep(2);
      renderRLDecision(d.rl_action, d.density);
    }

    // Step 3: Outcome
    if (d.outcome) {
      activateRLStep(3);
      renderOutcome(d.outcome);
    }

    // MDP Tuple
    if (d.mdp_tuple) {
      renderMDPTuple(d.mdp_tuple);
    }

    document.getElementById("rlDecisionPanel").style.display = "grid";

    // Highlight current state on Q-table heatmap
    loadQTableSnapshot().then(() => {
      // Re-render with highlight if we already have data
    });

    loadHistory();
    refreshStatus();

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
      <span class="yolo-density-label">RL State (s) / Density ${level}</span>
      <span class="yolo-density-val">${d.density}<span style="font-size:1rem;font-weight:400;color:var(--text-muted)">/100</span></span>
    </div>
    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:6px">
      ${d.num_detections} vehicles detected · Raw density: ${d.raw_density}
    </div>`;

  if (d.annotated_b64) {
    const preview = document.getElementById("yoloPreview");
    document.getElementById("yoloAnnotated").src = "data:image/jpeg;base64," + d.annotated_b64;
    preview.style.display = "block";
  }
}

function renderMDPTuple(mdp) {
  const panel = document.getElementById("mdpTuplePanel");
  panel.style.display = "block";

  document.getElementById("mdpTupleRow").innerHTML = `
    <div class="mdp-item">
      <div class="mdp-symbol">S</div>
      <div class="mdp-val">${mdp.S}</div>
      <div class="mdp-desc">State (density)</div>
    </div>
    <div class="mdp-arrow">→</div>
    <div class="mdp-item">
      <div class="mdp-symbol">A</div>
      <div class="mdp-val">${mdp.A}</div>
      <div class="mdp-desc">${mdp.A_label}</div>
    </div>
    <div class="mdp-arrow">→</div>
    <div class="mdp-item">
      <div class="mdp-symbol">R</div>
      <div class="mdp-val" style="color:${mdp.R >= 0 ? '#10b981' : '#ef4444'}">${mdp.R >= 0 ? '+' : ''}${mdp.R}</div>
      <div class="mdp-desc">Reward</div>
    </div>
    <div class="mdp-arrow">→</div>
    <div class="mdp-item">
      <div class="mdp-symbol">S'</div>
      <div class="mdp-val">${mdp.S_prime}</div>
      <div class="mdp-desc">Next state</div>
    </div>
    <div class="mdp-item">
      <div class="mdp-symbol">max Q(S',a)</div>
      <div class="mdp-val">${mdp.max_Q_S_prime}</div>
      <div class="mdp-desc">Future value</div>
    </div>`;
}

function renderRLDecision(rl, density) {
  setEl("rlDuration",      rl.duration + "s");
  setEl("rlActionIdx",     `${rl.action_idx} / 9`);
  setEl("rlQValue",        rl.q_value.toFixed(4));
  setEl("rlDensityBefore", `${density} / 100`);

  // Q-value bar chart
  const maxQ   = Math.max(...rl.q_values.map(Math.abs), 0.01);
  const qBarsRL = document.getElementById("qBarsRL");
  const actions = [10,20,30,40,50,60,70,80,90,100];
  qBarsRL.innerHTML = rl.q_values.map((q, i) => {
    const h    = Math.abs(q) / maxQ * 100;
    const best = i === rl.action_idx ? "best" : "";
    return `<div class="q-bar ${best}" style="height:${Math.max(4,h)}%"
              title="Action=${actions[i]}s Q=${q.toFixed(3)}">
              <span class="q-bar-label">${actions[i]}</span>
            </div>`;
  }).join("");

  // Policy explanation
  if (rl.explanation) {
    document.getElementById("rlExplanation").innerHTML = `
      <div class="explanation-text">${rl.explanation}</div>`;
  }
}

function renderOutcome(o) {
  const success = o.success;
  const changeText = o.density_change > 0
    ? `+${o.density_change} 📈`
    : `${o.density_change} 📉`;

  const verdictColor = success ? "#10b981" : "#ef4444";
  const verdictIcon  = success ? "✅" : "❌";
  const verdictText  = success ? "Traffic Cleared!" : "Congestion Persisted";

  document.getElementById("outcomeVerdict").innerHTML = `
    <div class="verdict-box" style="border-color:${verdictColor}20;background:${verdictColor}10">
      <span class="verdict-icon">${verdictIcon}</span>
      <div>
        <div class="verdict-title" style="color:${verdictColor}">${verdictText}</div>
        <div class="verdict-sub">Density: ${o.density_before} → ${o.density_after} (${changeText})</div>
      </div>
    </div>`;

  document.getElementById("outcomeMetrics").innerHTML = `
    <div class="outcome-metric-grid">
      <div class="om-item">
        <div class="om-val">${o.density_before}</div>
        <div class="om-label">Before</div>
      </div>
      <div class="om-arrow">→ ${o.green_duration}s green →</div>
      <div class="om-item">
        <div class="om-val" style="color:${success ? '#10b981' : '#ef4444'}">${o.density_after}</div>
        <div class="om-label">After</div>
      </div>
      <div class="om-item">
        <div class="om-val" style="color:${o.reward >= 0 ? '#10b981' : '#ef4444'}">${o.reward >= 0 ? '+' : ''}${o.reward}</div>
        <div class="om-label">Reward</div>
      </div>
      <div class="om-item">
        <div class="om-val">${o.wait_reduction_pct}%</div>
        <div class="om-label">Flow Improved</div>
      </div>
    </div>`;

  // Reward breakdown
  if (o.reward_breakdown && o.reward_breakdown.length > 0) {
    const breakdown = o.reward_breakdown.map(rb => {
      const sign = rb.value >= 0 ? "+" : "";
      const color = rb.value >= 0 ? "#10b981" : "#ef4444";
      return `<div class="rb-row">
        <span>${rb.component}</span>
        <span style="color:${color};font-weight:700">${sign}${rb.value}</span>
      </div>`;
    }).join("");
    document.getElementById("rewardBreakdown").innerHTML = `
      <div class="rb-title">Reward Breakdown (Q-Learning)</div>
      ${breakdown}
      <div class="rb-row rb-total">
        <span>Total Reward</span>
        <span style="color:${o.reward >= 0 ? '#10b981' : '#ef4444'};font-weight:800">
          ${o.reward >= 0 ? '+' : ''}${o.reward}
        </span>
      </div>`;
  }
}

function activateRLStep(stepNum) {
  [1,2,3,4].forEach(n => {
    const el = document.getElementById(`step${ ["Detect","Decide","Simulate","Learn"][n-1] }`);
    if (el) {
      el.classList.toggle("step-active", n === stepNum);
      el.classList.toggle("step-done",   n < stepNum);
    }
  });
}

// ━━━━━━━━━━━━━━━━━━━━━ ONLINE LEARNING (FEEDBACK) — Q-TABLE DIFF ━━━━━━━━━━━━━━━━━━━━━

async function handleAfterImageUpload(event) {
  const afterFile = event.target.files[0];
  if (!afterFile || lastActionIdx === null) return;

  const resultDiv = document.getElementById("feedbackResult");
  resultDiv.innerHTML = '<span style="color:var(--text-muted)">⏳ Processing real feedback & updating Q-table…</span>';
  activateRLStep(4);

  const beforeInput = document.getElementById("fileInput");
  const beforeFile  = beforeInput.files[0];

  if (!beforeFile) {
    resultDiv.innerHTML = '<span style="color:#fca5a5">⚠ Please upload the "before" image first via the main upload zone, then upload "after".</span>';
    return;
  }

  const formData = new FormData();
  formData.append("before_file", beforeFile);
  formData.append("after_file",  afterFile);
  formData.append("action_idx",  lastActionIdx);

  try {
    const r = await fetch(`${API}/api/feedback`, { method: "POST", body: formData });
    if (!r.ok) {
      const err = await r.json();
      resultDiv.innerHTML = `<span style="color:#fca5a5">Error: ${err.detail}</span>`;
      return;
    }
    const d = await r.json();

    const icon  = d.success ? "✅" : "❌";
    const color = d.success ? "#10b981" : "#ef4444";
    resultDiv.innerHTML = `
      <div class="feedback-success" style="border-left:3px solid ${color}">
        <div style="font-weight:700;color:${color}">${icon} Q-Table Updated from Real Data!</div>
        <div>State: ${d.state} → ${d.next_state} | Action: ${d.action_idx} (${d.duration}s) | Reward: ${d.reward >= 0 ? '+' : ''}${d.reward}</div>
        <div style="color:var(--text-muted);font-size:0.75rem">TD error: ${d.td_error} | Total Q-updates: ${d.total_updates}</div>
      </div>`;

    // ── RENDER BELLMAN EQUATION WITH REAL NUMBERS ────────────
    if (d.bellman) {
      renderBellmanEquation(d.bellman, d.state, d.action_idx);
    }

    // ── RENDER Q-ROW BEFORE/AFTER DIFF ──────────────────────
    if (d.q_row_before && d.q_row_after) {
      renderQTableDiff(d.q_row_before, d.q_row_after, d.q_diff,
                       d.state, d.action_idx);
    }

    // Refresh everything
    refreshStatus();
    loadHistory();
    loadQTableSnapshot();

  } catch (e) {
    resultDiv.innerHTML = `<span style="color:#fca5a5">API error: ${e.message}</span>`;
  }
}

function renderBellmanEquation(bellman, state, actionIdx) {
  const panel = document.getElementById("bellmanDisplay");
  panel.style.display = "block";
  const c = bellman.components;

  panel.innerHTML = `
    <div class="bellman-title">🔬 Q-Learning Update — Sutton & Barto Eq. 6.8</div>
    <div class="bellman-eq-main">
      ${bellman.equation}
    </div>
    <div class="bellman-eq-result">
      ${bellman.result}
    </div>
    <div class="bellman-components">
      <div class="bc-item">
        <span class="bc-label">Q<sub>old</sub>(${state},${actionIdx})</span>
        <span class="bc-val">${c.Q_old.toFixed(4)}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">α (learning rate)</span>
        <span class="bc-val">${c.alpha}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">R (reward)</span>
        <span class="bc-val" style="color:${c.reward >= 0 ? '#10b981' : '#ef4444'}">${c.reward >= 0 ? '+' : ''}${c.reward}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">γ (discount)</span>
        <span class="bc-val">${c.gamma}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">max Q(S',a)</span>
        <span class="bc-val">${c.max_Q_next.toFixed(4)}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">TD Target</span>
        <span class="bc-val">${c.td_target.toFixed(4)}</span>
      </div>
      <div class="bc-item">
        <span class="bc-label">δ (TD Error)</span>
        <span class="bc-val" style="color:#f59e0b;font-weight:700">${c.td_error.toFixed(4)}</span>
      </div>
      <div class="bc-item bc-highlight">
        <span class="bc-label">ΔQ (change)</span>
        <span class="bc-val" style="color:#8b5cf6;font-weight:800">${c.delta_Q >= 0 ? '+' : ''}${c.delta_Q.toFixed(6)}</span>
      </div>
      <div class="bc-item bc-highlight">
        <span class="bc-label">Q<sub>new</sub>(${state},${actionIdx})</span>
        <span class="bc-val" style="color:#3b82f6;font-weight:800">${c.Q_new.toFixed(4)}</span>
      </div>
    </div>
  `;
}

function renderQTableDiff(before, after, diff, state, actionIdx) {
  const panel = document.getElementById("qDiffPanel");
  panel.style.display = "block";
  const actions = [10,20,30,40,50,60,70,80,90,100];

  // Find max for scaling bars
  const allVals = [...before.map(Math.abs), ...after.map(Math.abs)];
  const maxVal = Math.max(...allVals, 0.001);

  let rows = "";
  for (let i = 0; i < before.length; i++) {
    const isChanged = Math.abs(diff[i]) > 0.000001;
    const isAction = i === actionIdx;
    const rowClass = isAction ? "qdiff-row-active" : (isChanged ? "qdiff-row-changed" : "");
    const changeColor = diff[i] > 0 ? "#10b981" : diff[i] < 0 ? "#ef4444" : "var(--text-muted)";
    const changeSign = diff[i] > 0 ? "+" : "";

    const beforeW = Math.abs(before[i]) / maxVal * 100;
    const afterW = Math.abs(after[i]) / maxVal * 100;

    rows += `
      <div class="qdiff-row ${rowClass}">
        <div class="qdiff-action">${actions[i]}s ${isAction ? "⭐" : ""}</div>
        <div class="qdiff-before">
          <div class="qdiff-bar-bg"><div class="qdiff-bar qdiff-bar-before" style="width:${Math.max(2, beforeW)}%"></div></div>
          <span>${before[i].toFixed(4)}</span>
        </div>
        <div class="qdiff-arrow">→</div>
        <div class="qdiff-after">
          <div class="qdiff-bar-bg"><div class="qdiff-bar qdiff-bar-after" style="width:${Math.max(2, afterW)}%"></div></div>
          <span>${after[i].toFixed(4)}</span>
        </div>
        <div class="qdiff-change" style="color:${changeColor}">
          ${isChanged ? `${changeSign}${diff[i].toFixed(6)}` : "—"}
        </div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="qdiff-title">📊 Q-Table Row Diff — State s=${state}</div>
    <div class="qdiff-header">
      <span>Action</span>
      <span>Q-value BEFORE</span>
      <span></span>
      <span>Q-value AFTER</span>
      <span>Change (ΔQ)</span>
    </div>
    ${rows}
    <div class="qdiff-note">
      ⭐ = action taken · Only Q(${state}, ${actionIdx}) changes — Q-Learning updates one cell per transition
    </div>`;
}

// ━━━━━━━━━━━━━━━━━━━━━ DECISION HISTORY ━━━━━━━━━━━━━━━━━━━━━

async function loadHistory() {
  try {
    const r = await fetch(`${API}/api/decision-history`);
    if (!r.ok) return;
    const d = await r.json();
    renderHistory(d.history, d.total);
  } catch (_) {}
}

function renderHistory(history, total) {
  setEl("historyCount", `${total} decision${total !== 1 ? "s" : ""}`);
  const tbody = document.getElementById("historyBody");

  if (!history || history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="placeholder">No decisions yet — upload a traffic image to begin</td></tr>';
    return;
  }

  const rows = history.map(h => {
    const o        = h.outcome || {};
    const success  = o.success;
    const reward   = o.reward ?? "—";
    const rewardTxt = typeof reward === "number"
      ? (reward >= 0 ? `<span style="color:#10b981">+${reward}</span>` : `<span style="color:#ef4444">${reward}</span>`)
      : "—";
    const successBadge = success === true
      ? '<span class="hist-badge hist-success">✅ Clear</span>'
      : success === false
        ? '<span class="hist-badge hist-fail">❌ Persist</span>'
        : "—";
    const densityLevel = h.density >= 70 ? "🔴" : h.density >= 35 ? "🟡" : "🟢";
    const afterDensity = o.density_after !== undefined
      ? `${o.density_before}→${o.density_after}` : "—";
    const isReal = h.is_real_feedback
      ? '<span class="hist-badge hist-real">🔄 Real</span>' : '';
    const time = h.timestamp ? h.timestamp.replace("T", " ") : "—";

    return `<tr>
      <td class="hist-id">${h.id}${isReal}</td>
      <td class="hist-time">${time}</td>
      <td class="hist-file" title="${h.filename}">${truncate(h.filename, 18)}</td>
      <td>${densityLevel} ${h.density}</td>
      <td>${h.action_idx !== undefined ? h.action_idx : "—"}</td>
      <td><b>${h.duration}s</b></td>
      <td>${afterDensity}</td>
      <td>${rewardTxt}</td>
      <td>${successBadge}</td>
    </tr>`;
  }).join("");

  tbody.innerHTML = rows;
}

async function clearHistory() {
  if (!confirm("Clear all decision history for this session?")) return;
  await fetch(`${API}/api/decision-history`, { method: "DELETE" });
  loadHistory();
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

function truncate(str, n) {
  return str && str.length > n ? str.slice(0, n) + "…" : str;
}
