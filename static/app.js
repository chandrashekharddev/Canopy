// ==========================================================================
// Canopy front-end logic — no external dependencies (self-contained SVG chart)
// ==========================================================================

// Must match SMOOTHING_WINDOW in crop_core.py - used only for the chart legend label.
const SMOOTHING_WINDOW = 5;

const HEALTH_LEGEND = [
  { label: "Water / Cloud / No Vegetation", color: "#5fa8e0", range: "NDVI < 0.0" },
  { label: "Bare Soil / Very Poor", color: "#b45309", range: "0.0 – 0.2" },
  { label: "Sparse / Stressed Vegetation", color: "#f2b84b", range: "0.2 – 0.4" },
  { label: "Moderate Health", color: "#eab308", range: "0.4 – 0.6" },
  { label: "Healthy / Vigorous Growth", color: "#84cc16", range: "0.6 – 0.8" },
  { label: "Very Healthy / Peak Vigor", color: "#16a34a", range: "0.8 – 1.0" },
];

document.addEventListener("DOMContentLoaded", () => {
  renderLegend();
  checkHealth();
  setupTabs();
  setupSingleForm();
  setupBatchForm();

  // default the date picker to today
  const d = new Date().toISOString().slice(0, 10);
  document.getElementById("targetDate").value = d;
});

// -------------------------------------------------------------------------
// Health legend
// -------------------------------------------------------------------------
function renderLegend() {
  const grid = document.getElementById("legendGrid");
  grid.innerHTML = HEALTH_LEGEND.map(h => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${h.color}"></span>
      <div>
        <strong>${h.label}</strong>
        <span class="rng">${h.range}</span>
      </div>
    </div>
  `).join("");
}

// -------------------------------------------------------------------------
// Status pills
// -------------------------------------------------------------------------
async function checkHealth() {
  const pillModel = document.getElementById("pillModel");
  const pillGee = document.getElementById("pillGee");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    setPill(pillModel, data.model_ready, "model ready", data.model_error || "model not trained");
    setPill(pillGee, data.gee_ready, "earth engine ready", data.gee_error || "earth engine not configured");
  } catch (e) {
    setPill(pillModel, false, "", "backend unreachable");
    setPill(pillGee, false, "", "backend unreachable");
  }
}
function setPill(el, ok, okText, badText) {
  el.classList.remove("pill-loading", "pill-ok", "pill-bad");
  el.classList.add(ok ? "pill-ok" : "pill-bad");
  el.textContent = (ok ? "● " : "● ") + (ok ? okText : badText);
  if (!ok) el.title = badText;
}

// -------------------------------------------------------------------------
// Tabs
// -------------------------------------------------------------------------
function setupTabs() {
  const tabSingle = document.getElementById("tabSingleBtn");
  const tabBatch = document.getElementById("tabBatchBtn");
  const panelSingle = document.getElementById("panelSingle");
  const panelBatch = document.getElementById("panelBatch");

  tabSingle.addEventListener("click", () => {
    tabSingle.classList.add("active"); tabSingle.setAttribute("aria-selected", "true");
    tabBatch.classList.remove("active"); tabBatch.setAttribute("aria-selected", "false");
    panelSingle.classList.add("active"); panelBatch.classList.remove("active");
  });
  tabBatch.addEventListener("click", () => {
    tabBatch.classList.add("active"); tabBatch.setAttribute("aria-selected", "true");
    tabSingle.classList.remove("active"); tabSingle.setAttribute("aria-selected", "false");
    panelBatch.classList.add("active"); panelSingle.classList.remove("active");
  });
}

// -------------------------------------------------------------------------
// Single-point form
// -------------------------------------------------------------------------
function setupSingleForm() {
  const form = document.getElementById("singleForm");
  const btn = document.getElementById("singleSubmitBtn");
  const errorBox = document.getElementById("singleError");
  const resultArea = document.getElementById("singleResult");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hide(errorBox); hide(resultArea);
    setBusy(btn, true);

    const payload = {
      latitude: parseFloat(document.getElementById("lat").value),
      longitude: parseFloat(document.getElementById("lon").value),
      target_date: document.getElementById("targetDate").value,
      crop_hint: document.getElementById("cropHint").value || null,
      farm_id: "Point_1",
    };

    try {
      const res = await fetch("/api/predict/single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed.");

      if (data.status === "failed") {
        showError(errorBox, data.error || "No prediction could be made for this point/date.");
      } else if (data.status === "unknown") {
        resultArea.innerHTML = "";
        resultArea.appendChild(buildUnknownBanner(data));
        resultArea.appendChild(buildResultCard({
          ...data,
          predicted_crop: data.closest_match ? data.closest_match.crop : null,
          confidence: data.closest_match ? data.closest_match.confidence : data.confidence,
        }));
        show(resultArea);
      } else {
        resultArea.innerHTML = "";
        resultArea.appendChild(buildResultCard(data));
        show(resultArea);
      }
    } catch (err) {
      showError(errorBox, err.message || String(err));
    } finally {
      setBusy(btn, false);
    }
  });
}

// -------------------------------------------------------------------------
// Batch form
// -------------------------------------------------------------------------
function setupBatchForm() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileNameLabel = document.getElementById("fileNameLabel");
  const submitBtn = document.getElementById("batchSubmitBtn");
  const progress = document.getElementById("batchProgress");
  const errorBox = document.getElementById("batchError");
  const resultArea = document.getElementById("batchResult");

  let selectedFile = null;

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    selectedFile = file;
    fileNameLabel.textContent = file.name;
    submitBtn.disabled = false;
  }

  submitBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hide(errorBox); hide(resultArea);
    show(progress);
    setBusy(submitBtn, true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/api/predict/batch", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Batch prediction failed.");
      resultArea.innerHTML = "";
      resultArea.appendChild(buildBatchSummary(data));
      resultArea.appendChild(buildBatchTable(data.results));
      show(resultArea);
    } catch (err) {
      showError(errorBox, err.message || String(err));
    } finally {
      hide(progress);
      setBusy(submitBtn, false);
    }
  });
}

function buildBatchSummary(data) {
  const div = document.createElement("div");
  div.className = "batch-summary";
  div.innerHTML = `
    <span><b>${data.n_ok}</b> / ${data.n_total} predicted</span>
    <span>${data.n_failed} failed</span>
    <button class="btn-download">Download CSV</button>
  `;
  div.querySelector(".btn-download").addEventListener("click", () => {
    window.location.href = `/api/predict/batch/${data.job_id}/csv`;
  });
  return div;
}

function buildBatchTable(results) {
  const wrap = document.createElement("div");
  wrap.className = "batch-table-wrap";
  const rows = results.map(r => {
    const health = (r.health && r.health.nearest_to_target_date) ? r.health.nearest_to_target_date.label : "—";
    const isUnknown = r.status === "unknown";
    const cropDisplay = isUnknown
      ? `${r.closest_match ? r.closest_match.crop : "—"} (uncertain)`
      : (r.predicted_crop ?? "—");
    const confSource = isUnknown ? (r.closest_match ? r.closest_match.confidence : r.confidence) : r.confidence;
    const conf = confSource != null ? (confSource * 100).toFixed(1) + "%" : "—";
    const statusClass = r.status === "ok" ? "status-ok" : (isUnknown ? "" : "status-failed");
    const statusStyle = isUnknown ? ' style="color:#f2b84b;"' : "";
    const duration = r.lifecycle
      ? `${r.lifecycle.duration_days}d${r.lifecycle.duration_plausible === false ? " ⚠" : ""}`
      : "—";
    return `
      <tr>
        <td>${escapeHtml(r.farm_id ?? "—")}</td>
        <td>${r.latitude?.toFixed(5) ?? "—"}</td>
        <td>${r.longitude?.toFixed(5) ?? "—"}</td>
        <td>${r.target_date ?? "—"}</td>
        <td class="${statusClass}"${statusStyle}>${r.status}</td>
        <td>${escapeHtml(cropDisplay)}</td>
        <td>${conf}</td>
        <td>${duration}</td>
        <td>${escapeHtml(health)}</td>
        <td>${escapeHtml(r.error ?? r.message ?? "")}</td>
      </tr>`;
  }).join("");

  wrap.innerHTML = `
    <table class="batch-table">
      <thead>
        <tr>
          <th>Farm</th><th>Lat</th><th>Lon</th><th>Date</th><th>Status</th>
          <th>Predicted crop</th><th>Confidence</th><th>Duration</th><th>Health (nearest)</th><th>Note</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
  return wrap;
}

// -------------------------------------------------------------------------
// "Doesn't match any of the 5 supported crops" banner
// -------------------------------------------------------------------------
function buildUnknownBanner(data) {
  const div = document.createElement("div");
  div.className = "result-card";
  div.style.borderColor = "#f2b84b";
  const closestCrop = data.closest_match ? data.closest_match.crop : "—";
  const closestConf = data.closest_match && data.closest_match.confidence != null
    ? (data.closest_match.confidence * 100).toFixed(1) + "%"
    : "—";
  div.innerHTML = `
    <div style="padding:18px 22px; display:flex; gap:14px; align-items:flex-start;">
      <span style="font-size:22px; line-height:1;">⚠️</span>
      <div>
        <div style="font-weight:700; color:#f2b84b; font-size:15px; margin-bottom:6px;">
          Not a confident match for any of the 5 supported crops
        </div>
        <div style="font-size:13.5px; color:#b7c9be; line-height:1.6;">
          ${escapeHtml(data.message || "This location/date's NDVI pattern doesn't clearly match Tomato, Cotton, Paddy, Sugarcane, or Onion.")}
        </div>
        <div style="font-size:12.5px; color:#7c9186; margin-top:8px; font-family:'IBM Plex Mono',monospace;">
          Closest match: <strong style="color:#e8f0eb;">${escapeHtml(closestCrop)}</strong> at ${closestConf} confidence
        </div>
      </div>
    </div>
  `;
  return div;
}

// -------------------------------------------------------------------------
// Single-point result card (+ inline SVG NDVI chart)
// -------------------------------------------------------------------------
function buildResultCard(data) {
  const tpl = document.getElementById("resultCardTemplate");
  const node = tpl.content.cloneNode(true);
  const card = node.querySelector(".result-card");

  card.querySelector(".result-farm-id").textContent = `${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)} · ${data.target_date}`;
  card.querySelector(".result-crop").textContent = data.predicted_crop || "Unclassified";
  card.querySelector(".confidence-value").textContent = data.confidence != null ? (data.confidence * 100).toFixed(1) + "%" : "—";

  // probability bars
  const probWrap = card.querySelector(".probability-bars");
  if (data.probabilities) {
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    probWrap.innerHTML = entries.map(([crop, p]) => `
      <div class="prob-row ${crop === data.predicted_crop ? "is-predicted" : ""}">
        <span class="prob-label">${crop}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
        <span class="prob-pct">${(p * 100).toFixed(0)}%</span>
      </div>
    `).join("");
  }

  card.querySelector(".lifecycle-window").textContent = `${data.lifecycle.start_date} → ${data.lifecycle.end_date}`;

  const expected = data.lifecycle.expected_duration_days;
  const plausible = data.lifecycle.duration_plausible;
  const durationEl = card.querySelector(".lifecycle-duration");
  if (expected) {
    const rangeText = `(expected ${expected.min}–${expected.max}d)`;
    if (plausible === false) {
      durationEl.innerHTML = `${data.lifecycle.duration_days} days <span style="color:#f2b84b;">⚠ ${rangeText}</span>`;
      durationEl.title = "Duration falls outside the typical range for this crop.";
    } else {
      durationEl.innerHTML = `${data.lifecycle.duration_days} days <span style="color:#7c9186; font-size:11px;">${rangeText}</span>`;
    }
  } else {
    durationEl.textContent = `${data.lifecycle.duration_days} days`;
  }

  card.querySelector(".peak-ndvi").textContent = data.lifecycle.peak_ndvi.toFixed(4);
  card.querySelector(".peak-date").textContent = data.lifecycle.peak_date;

  const peakChip = card.querySelector(".health-chip-peak");
  const peakHealth = data.health.at_peak;
  peakChip.querySelector(".health-dot").style.background = peakHealth.color;
  peakChip.querySelector(".health-label").textContent = peakHealth.label;

  const nearestChip = card.querySelector(".health-chip-nearest");
  const nearestHealth = data.health.nearest_to_target_date;
  if (nearestHealth) {
    nearestChip.querySelector(".health-dot").style.background = nearestHealth.color;
    nearestChip.querySelector(".health-label").textContent = `${nearestHealth.label} (${nearestHealth.date}, NDVI ${nearestHealth.ndvi.toFixed(3)})`;
  } else {
    nearestChip.style.display = "none";
  }

  const chartHost = card.querySelector(".chart-svg");
  chartHost.innerHTML = renderNdviChartSvg(data.ndvi_series, data.lifecycle, data.target_date);

  return card;
}

// -------------------------------------------------------------------------
// Self-contained SVG NDVI time-series chart
// -------------------------------------------------------------------------
function renderNdviChartSvg(series, lifecycle, targetDateStr) {
  if (!series || series.length === 0) return "<p>No NDVI series available.</p>";

  const W = 900, H = 260, padL = 44, padR = 16, padT = 16, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const dates = series.map(p => new Date(p.date).getTime());
  const rawVals = series.map(p => p.ndvi);
  const smoothVals = series.map(p => (p.ndvi_smooth != null ? p.ndvi_smooth : p.ndvi));
  const minX = Math.min(...dates), maxX = Math.max(...dates);
  const allVals = rawVals.concat(smoothVals);
  const minY = Math.min(-0.1, Math.min(...allVals) - 0.05);
  const maxY = Math.max(0.9, Math.max(...allVals) + 0.05);

  const xScale = (t) => padL + ((t - minX) / (maxX - minX || 1)) * plotW;
  const yScale = (v) => padT + plotH - ((v - minY) / (maxY - minY || 1)) * plotH;

  const rawPoints = series.map(p => `${xScale(new Date(p.date).getTime()).toFixed(1)},${yScale(p.ndvi).toFixed(1)}`).join(" ");
  const smoothPoints = series.map(p => `${xScale(new Date(p.date).getTime()).toFixed(1)},${yScale(p.ndvi_smooth != null ? p.ndvi_smooth : p.ndvi).toFixed(1)}`).join(" ");

  // lifecycle shaded region
  const lcStartX = xScale(new Date(lifecycle.start_date).getTime());
  const lcEndX = xScale(new Date(lifecycle.end_date).getTime());

  // peak marker (raw peak, matching what the classifier actually used)
  const peakX = xScale(new Date(lifecycle.peak_date).getTime());
  const peakY = yScale(lifecycle.peak_ndvi);

  // target date marker (clamped into range)
  const targetT = new Date(targetDateStr).getTime();
  const targetX = xScale(Math.min(Math.max(targetT, minX), maxX));

  // gridlines at NDVI 0, 0.2, 0.4, 0.6, 0.8
  const gridLevels = [0, 0.2, 0.4, 0.6, 0.8];
  const gridLines = gridLevels
    .filter(v => v >= minY && v <= maxY)
    .map(v => `
      <line x1="${padL}" y1="${yScale(v).toFixed(1)}" x2="${W - padR}" y2="${yScale(v).toFixed(1)}"
            stroke="#1a2b24" stroke-width="1"/>
      <text x="${padL - 8}" y="${(yScale(v) + 4).toFixed(1)}" text-anchor="end"
            font-family="IBM Plex Mono, monospace" font-size="10" fill="#7c9186">${v.toFixed(1)}</text>
    `).join("");

  const firstLabel = new Date(minX).toISOString().slice(0, 10);
  const lastLabel = new Date(maxX).toISOString().slice(0, 10);

  return `
    <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
      <rect x="${lcStartX.toFixed(1)}" y="${padT}" width="${Math.max(0, lcEndX - lcStartX).toFixed(1)}" height="${plotH}"
            fill="#6fe38f" opacity="0.08"/>
      ${gridLines}
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#24382f"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#24382f"/>

      <polyline points="${rawPoints}" fill="none" stroke="#6fe38f" stroke-width="1" stroke-opacity="0.35" stroke-linejoin="round" stroke-linecap="round"/>
      <polyline points="${smoothPoints}" fill="none" stroke="#6fe38f" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>

      <line x1="${targetX.toFixed(1)}" y1="${padT}" x2="${targetX.toFixed(1)}" y2="${padT + plotH}"
            stroke="#5fa8e0" stroke-width="1.5" stroke-dasharray="4,3"/>

      <circle cx="${peakX.toFixed(1)}" cy="${peakY.toFixed(1)}" r="4.5" fill="#f2b84b" stroke="#0f1b17" stroke-width="1.5"/>

      <text x="${padL}" y="${H - 8}" font-family="IBM Plex Mono, monospace" font-size="10" fill="#7c9186">${firstLabel}</text>
      <text x="${W - padR}" y="${H - 8}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="10" fill="#7c9186">${lastLabel}</text>
    </svg>
    <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:#7c9186;font-family:'IBM Plex Mono',monospace;flex-wrap:wrap;">
      <span><span style="display:inline-block;width:14px;height:2.5px;background:#6fe38f;margin-right:5px;vertical-align:middle;"></span>NDVI (smoothed, ${SMOOTHING_WINDOW}-pt avg)</span>
      <span><span style="display:inline-block;width:14px;height:1px;background:#6fe38f;opacity:0.35;margin-right:5px;vertical-align:middle;"></span>NDVI (raw)</span>
      <span><span style="display:inline-block;width:8px;height:8px;background:#f2b84b;border-radius:50%;margin-right:5px;"></span>peak</span>
      <span><span style="display:inline-block;width:10px;height:2px;background:#5fa8e0;margin-right:5px;vertical-align:middle;"></span>target date</span>
      <span><span style="display:inline-block;width:10px;height:8px;background:#6fe38f;opacity:0.3;margin-right:5px;vertical-align:middle;"></span>lifecycle window</span>
    </div>
  `;
}

// -------------------------------------------------------------------------
// small utils
// -------------------------------------------------------------------------
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }
function showError(el, msg) { el.textContent = msg; show(el); }
function setBusy(btn, busy) {
  btn.disabled = busy;
  btn.querySelector(".btn-label").style.opacity = busy ? 0.5 : 1;
  const spinner = btn.querySelector(".btn-spinner");
  if (spinner) spinner.hidden = !busy;
}
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str).replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[m]));
}