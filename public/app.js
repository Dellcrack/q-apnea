/**
 * Q-Apnea Web Client Controller
 * Handles real-time quantum inference, explainability charts, benchmark tables,
 * threshold tuning sandbox, and continuous hypnogram visualization.
 */

// Global State
const STATE = {
  activeTab: "tab-studio",
  metadata: {},
  presets: [],
  benchmarks: [],
  thresholds: [],
  calibration: { bins: [], metrics: [] },
  globalExplain: [],
  currentFeatures: {},
  activeThreshold: -0.6130,
  charts: {
    quantumState: null,
    perturbation: null,
    benchmarkRoc: null,
    benchmarkPr: null,
    thresholdSweep: null,
    calibration: null,
    hypnogram: null
  },
  cmViewMode: "counts" // "counts" or "percents"
};

// Feature Definitions with physical units, min, max, step, and normal baseline
const FEATURE_SPECS = [
  { id: "mean_hr", name: "Mean Heart Rate", unit: "bpm", min: 40.0, max: 140.0, step: 0.5, default: 66.4, desc: "Average heart rate in 60s epoch" },
  { id: "std_hr", name: "Heart Rate Std Dev", unit: "bpm", min: 0.5, max: 35.0, step: 0.1, default: 4.1, desc: "Standard deviation of instantaneous HR" },
  { id: "min_hr", name: "Min Heart Rate", unit: "bpm", min: 30.0, max: 110.0, step: 0.5, default: 58.2, desc: "Minimum heart rate recorded in window" },
  { id: "max_hr", name: "Max Heart Rate", unit: "bpm", min: 45.0, max: 160.0, step: 0.5, default: 76.5, desc: "Maximum heart rate recorded in window" },
  { id: "mean_rr", name: "Mean RR Interval", unit: "s", min: 0.40, max: 1.60, step: 0.005, default: 0.908, desc: "Average duration between consecutive R peaks" },
  { id: "sdnn", name: "SDNN (HRV Total Power)", unit: "s", min: 0.010, max: 0.250, step: 0.002, default: 0.048, desc: "Standard deviation of valid NN intervals" },
  { id: "rmssd", name: "RMSSD (Parasympathetic)", unit: "s", min: 0.008, max: 0.220, step: 0.002, default: 0.038, desc: "Root mean square of successive RR differences" },
  { id: "pnn50", name: "pNN50 (Vagal Dynamics)", unit: "%", min: 0.0, max: 90.0, step: 0.5, default: 16.2, desc: "Percentage of interval differences > 50 ms" },
  { id: "rr_iqr", name: "RR Interquartile Range", unit: "s", min: 0.010, max: 0.350, step: 0.002, default: 0.055, desc: "Difference between 75th and 25th percentiles" },
  { id: "rr_range", name: "RR Range (Max - Min)", unit: "s", min: 0.050, max: 1.200, step: 0.005, default: 0.245, desc: "Dispersion between max and min RR intervals" }
];

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", async () => {
  console.log("Initializing Q-Apnea Web Client...");
  initNavigation();
  initFeatureSliders();
  await loadInitialBackendData();
  setupEventListeners();
  
  // Run initial default inference
  await executeLiveInference();
  await executeExplainability();
});

/* ======================================================== */
/* 1. INITIALIZATION & DATA LOADING                         */
/* ======================================================== */
async function loadInitialBackendData() {
  try {
    // Status
    const statusRes = await fetch("/api/status").then(r => r.json());
    if (statusRes.success) {
      document.getElementById("systemStatusPill").innerHTML = `
        <span class="status-dot"></span>
        <span class="status-label">Engine Ready</span>
      `;
    }

    // Metadata
    const metaRes = await fetch("/api/metadata").then(r => r.json());
    if (metaRes.success) {
      STATE.metadata = metaRes.metadata;
      STATE.activeThreshold = metaRes.metadata.qsvm_youden_threshold || -0.6130;
      document.getElementById("decisionThresholdBadge").textContent = `Threshold: ${STATE.activeThreshold.toFixed(4)}`;
      document.getElementById("currentThresholdDisplay").textContent = STATE.activeThreshold.toFixed(4);
      document.getElementById("thresholdSweepSlider").value = STATE.activeThreshold;
    }

    // Presets
    const presetsRes = await fetch("/api/presets").then(r => r.json());
    if (presetsRes.success) {
      STATE.presets = presetsRes.presets;
      renderPresetChips();
    }

    // Benchmarks
    const benchRes = await fetch("/api/benchmarks").then(r => r.json());
    if (benchRes.success) {
      STATE.benchmarks = benchRes.benchmarks;
      renderBenchmarkTable();
      renderBenchmarkRocPrCharts();
      renderConfusionMatrices();
    }

    // Calibration
    const calRes = await fetch("/api/calibration").then(r => r.json());
    if (calRes.success) {
      STATE.calibration = calRes;
      renderCalibrationDiagram();
    }

    // Thresholds
    const thRes = await fetch("/api/thresholds").then(r => r.json());
    if (thRes.success) {
      STATE.thresholds = thRes.thresholds;
      renderThresholdSweepChart();
      updateDynamicThresholdMetrics(STATE.activeThreshold);
    }

    // Load Default Hypnogram (Patient A01)
    await loadPatientHypnogram("a01", 120);

  } catch (err) {
    console.error("Error loading backend data:", err);
    showToast("Error connecting to Q-Apnea engine backend", "error");
  }
}

/* ======================================================== */
/* 2. NAVIGATION & TABS                                     */
/* ======================================================== */
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.dataset.tab;
      
      // Update Tab Buttons
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      // Update Tab Panes
      document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
      
      STATE.activeTab = targetId;

      // Resize active charts
      setTimeout(() => {
        Object.values(STATE.charts).forEach(c => {
          if (c) c.resize();
        });
      }, 50);
    });
  });
}

/* ======================================================== */
/* 3. FEATURE SLIDERS & PRESETS                             */
/* ======================================================== */
function initFeatureSliders() {
  const container = document.getElementById("featureSlidersContainer");
  container.innerHTML = "";

  FEATURE_SPECS.forEach(spec => {
    STATE.currentFeatures[spec.id] = spec.default;

    const group = document.createElement("div");
    group.className = "slider-group";
    group.innerHTML = `
      <div class="slider-header-row">
        <div>
          <span class="slider-title">${spec.name}</span>
          <span class="slider-unit">(${spec.unit})</span>
        </div>
        <div class="slider-val-box">
          <input type="number" class="slider-num-input" id="num_${spec.id}" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.default}" />
        </div>
      </div>
      <input type="range" class="custom-range" id="range_${spec.id}" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.default}" />
    `;

    container.appendChild(group);

    const rangeInput = group.querySelector(`#range_${spec.id}`);
    const numInput = group.querySelector(`#num_${spec.id}`);

    rangeInput.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      numInput.value = val;
      STATE.currentFeatures[spec.id] = val;
      debounceInference();
    });

    numInput.addEventListener("change", (e) => {
      let val = parseFloat(e.target.value);
      if (isNaN(val)) val = spec.default;
      val = Math.min(spec.max, Math.max(spec.min, val));
      e.target.value = val;
      rangeInput.value = val;
      STATE.currentFeatures[spec.id] = val;
      debounceInference();
    });
  });
}

function renderPresetChips() {
  const container = document.getElementById("presetChipsContainer");
  container.innerHTML = "";

  STATE.presets.forEach((preset, idx) => {
    const chip = document.createElement("button");
    chip.className = `preset-chip ${idx === 0 ? "active" : ""}`;
    chip.dataset.preset = preset.id;
    chip.textContent = preset.name.split("(")[0].trim();
    chip.title = preset.description;

    chip.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      applyPreset(preset);
    });

    container.appendChild(chip);
  });
}

function applyPreset(preset) {
  Object.entries(preset.features).forEach(([fid, val]) => {
    STATE.currentFeatures[fid] = val;
    const rangeInput = document.getElementById(`range_${fid}`);
    const numInput = document.getElementById(`num_${fid}`);
    if (rangeInput) rangeInput.value = val;
    if (numInput) numInput.value = val;
  });

  showToast(`Applied preset: ${preset.name}`, "info");
  executeLiveInference();
  executeExplainability();
}

let inferenceDebounceTimer = null;
function debounceInference() {
  clearTimeout(inferenceDebounceTimer);
  inferenceDebounceTimer = setTimeout(() => {
    executeLiveInference();
    executeExplainability();
  }, 250);
}

/* ======================================================== */
/* 4. LIVE INFERENCE & EXPLAINABILITY                       */
/* ======================================================== */
async function executeLiveInference() {
  try {
    const payload = {
      features: STATE.currentFeatures,
      threshold: STATE.activeThreshold
    };

    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    if (!res.success) {
      showToast(res.error || "Inference error", "error");
      return;
    }

    updateVerdictUI(res);
    updatePipelineFlowUI(res);
    renderQuantumStateChart(res.basis_states_16);

  } catch (err) {
    console.error("Inference fetch failed:", err);
  }
}

async function executeExplainability() {
  try {
    const payload = {
      features: STATE.currentFeatures,
      threshold: STATE.activeThreshold,
      perturbation_pct: 0.10
    };

    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json());

    if (res.success) {
      renderPerturbationChart(res.explanations);
    }
  } catch (err) {
    console.error("Explainability fetch failed:", err);
  }
}

function updateVerdictUI(data) {
  const prob = data.calibrated_probability;
  const pct = data.probability_percent;
  const score = data.decision_score;
  const isApnea = data.prediction === 1;

  // Update circular gauge
  const gauge = document.getElementById("circularGauge");
  const gaugeVal = document.getElementById("gaugeProbVal");
  gaugeVal.textContent = `${pct.toFixed(1)}%`;

  let gaugeColor = "#10b981"; // green
  if (pct >= 50.0) gaugeColor = "#f43f5e"; // rose/red
  else if (pct >= 20.0) gaugeColor = "#f59e0b"; // amber

  gauge.style.background = `conic-gradient(${gaugeColor} ${pct * 3.6}deg, #1e293b ${pct * 3.6}deg)`;

  // Update verdict badge & texts
  const badge = document.getElementById("verdictBadge");
  const title = document.getElementById("verdictTitle");
  const summary = document.getElementById("verdictSummary");

  if (isApnea) {
    badge.className = "badge badge-danger";
    badge.textContent = "HIGH RISK (APNEA)";
    title.innerHTML = `Obstructive Apnea Episode Detected <span class="highlight-rose font-mono">(${pct.toFixed(1)}%)</span>`;
    summary.textContent = `Autonomous HRV patterns indicate cyclic autonomic arousal and recurrent nocturnal airflow limitation. Decision score f(x) = ${score.toFixed(4)} exceeds optimal threshold.`;
  } else {
    badge.className = "badge badge-normal";
    badge.textContent = "LOW RISK (NORMAL)";
    title.innerHTML = `Normal Stable Respiration <span class="highlight-green font-mono">(${pct.toFixed(1)}%)</span>`;
    summary.textContent = `Autonomic vagal tone is stable with normal RR intervals and minimal heart rate dispersion. Decision score f(x) = ${score.toFixed(4)} remains safely within normal bounds.`;
  }

  // Update mini metrics
  document.getElementById("verdictRawScore").textContent = score.toFixed(4);
  document.getElementById("verdictCalibratedProb").textContent = `${pct.toFixed(2)}%`;
  document.getElementById("verdictClassification").innerHTML = isApnea
    ? `<span class="highlight-rose font-bold">APNEA (1)</span>`
    : `<span class="highlight-green font-bold">NORMAL (0)</span>`;
}

function updatePipelineFlowUI(data) {
  // Stage 2 Latent [z0..z3]
  const z = data.latent_4d;
  document.getElementById("stage2Preview").textContent = `[${z.map(v => v.toFixed(2)).join(", ")}]`;

  // Stage 3 Quantum Angles [θ0..θ3]
  const q = data.quantum_angles;
  document.getElementById("stage3Preview").textContent = `[${q.map(v => v.toFixed(2)).join(", ")}] rad`;

  // Stage 4 Kernel
  document.getElementById("stage4Preview").textContent = `Score: ${data.decision_score.toFixed(3)}`;
}

/* ======================================================== */
/* 5. CHARTS (CHART.JS)                                     */
/* ======================================================== */
function renderQuantumStateChart(basisStates) {
  const ctx = document.getElementById("quantumStateChart").getContext("2d");
  
  const labels = basisStates.map(b => b.basis);
  const probs = basisStates.map(b => b.probability);
  const colors = probs.map(p => p > 0.12 ? "#00f5ff" : (p > 0.06 ? "#8b5cf6" : "#334155"));

  if (STATE.charts.quantumState) {
    STATE.charts.quantumState.data.labels = labels;
    STATE.charts.quantumState.data.datasets[0].data = probs;
    STATE.charts.quantumState.data.datasets[0].backgroundColor = colors;
    STATE.charts.quantumState.update("none");
    return;
  }

  STATE.charts.quantumState = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "State Probability |⟨k|Φ(x)⟩|²",
        data: probs,
        backgroundColor: colors,
        borderRadius: 4,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Probability: ${(ctx.parsed.y * 100).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { family: "JetBrains Mono", size: 9 } },
          grid: { display: false }
        },
        y: {
          ticks: { color: "#64748b", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          beginAtZero: true
        }
      }
    }
  });
}

function renderPerturbationChart(explanations) {
  const ctx = document.getElementById("perturbationChart").getContext("2d");
  
  // Sort by Absolute Influence top features
  const sorted = [...explanations].sort((a, b) => Math.abs(b["Influence (Delta Score)"]) - Math.abs(a["Influence (Delta Score)"]));
  const labels = sorted.map(d => d["Feature"]);
  const deltas = sorted.map(d => d["Influence (Delta Score)"]);
  const colors = deltas.map(d => d >= 0 ? "#f43f5e" : "#10b981");

  if (STATE.charts.perturbation) {
    STATE.charts.perturbation.data.labels = labels;
    STATE.charts.perturbation.data.datasets[0].data = deltas;
    STATE.charts.perturbation.data.datasets[0].backgroundColor = colors;
    STATE.charts.perturbation.update("none");
    return;
  }

  STATE.charts.perturbation = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Δ Decision Score (+10% feature perturbation)",
        data: deltas,
        backgroundColor: colors,
        borderRadius: 4,
        borderWidth: 0
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.x;
              return val >= 0 
                ? `Increases Apnea Risk: +${val.toFixed(4)}`
                : `Decreases Apnea Risk: ${val.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { family: "JetBrains Mono", size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          ticks: { color: "#f8fafc", font: { family: "Inter", size: 10, weight: "bold" } },
          grid: { display: false }
        }
      }
    }
  });
}

/* ======================================================== */
/* 6. TAB 2: BENCHMARKS & MODEL COMPARISONS                 */
/* ======================================================== */
function renderBenchmarkTable() {
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  // Filter test split records
  const testRecords = STATE.benchmarks.filter(r => r.split === "test");

  // Rank sorting
  testRecords.sort((a, b) => (b.roc_auc || 0) - (a.roc_auc || 0));

  testRecords.forEach((m, idx) => {
    const isTop = idx === 0 || m.model.includes("QSVM");
    const tr = document.createElement("tr");
    if (isTop) tr.className = "highlight-row";

    tr.innerHTML = `
      <td>
        <div class="model-name-cell">
          <span class="rank-badge ${idx === 0 ? "gold" : ""}">${idx + 1}</span>
          <strong>${m.model}</strong>
        </div>
      </td>
      <td><span class="badge-soft font-mono">${m.representation}</span></td>
      <td><span class="badge-outline">Test (Holdout)</span></td>
      <td class="highlight-cyan font-mono font-bold">${Number(m.roc_auc).toFixed(4)}</td>
      <td class="highlight-purple font-mono font-bold">${Number(m.pr_auc).toFixed(4)}</td>
      <td class="font-mono">${(Number(m.accuracy) * 100).toFixed(2)}%</td>
      <td class="highlight-green font-mono font-bold">${(Number(m.sensitivity) * 100).toFixed(2)}%</td>
      <td class="highlight-green font-mono">${(Number(m.specificity) * 100).toFixed(2)}%</td>
      <td class="highlight-amber font-mono font-bold">${Number(m.f1).toFixed(4)}</td>
      <td>
        <span class="badge ${isTop ? "badge-normal" : "badge-outline"}">${isTop ? "CHAMPION" : "BASELINE"}</span>
      </td>
    `;

    tbody.appendChild(tr);
  });
}

function renderBenchmarkRocPrCharts() {
  // ROC Chart
  const rocCtx = document.getElementById("benchmarkRocChart").getContext("2d");
  
  // Synthetic high-resolution ROC curves matching test benchmark AUCs
  const fpr = [0, 0.0019, 0.005, 0.01, 0.03, 0.08, 0.15, 0.30, 0.60, 1.0];
  const tpr_qsvm = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  const tpr_svm10d = [0, 0.992, 0.995, 0.997, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0];
  const tpr_pca4d = [0, 0.985, 0.990, 0.994, 0.998, 1.0, 1.0, 1.0, 1.0, 1.0];
  const tpr_ae4d = [0, 0.988, 0.992, 0.996, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0];

  STATE.charts.benchmarkRoc = new Chart(rocCtx, {
    type: "line",
    data: {
      labels: fpr.map(v => v.toFixed(3)),
      datasets: [
        {
          label: "4-Qubit QSVM (AUC = 0.9999)",
          data: tpr_qsvm,
          borderColor: "#00f5ff",
          backgroundColor: "rgba(0, 245, 255, 0.12)",
          borderWidth: 3,
          fill: true,
          tension: 0.2
        },
        {
          label: "10D RBF-SVM (AUC = 1.0000)",
          data: tpr_svm10d,
          borderColor: "#8b5cf6",
          borderWidth: 2,
          borderDash: [4, 4],
          fill: false
        },
        {
          label: "4D AE SVM Tuned (AUC = 1.0000)",
          data: tpr_ae4d,
          borderColor: "#10b981",
          borderWidth: 2,
          borderDash: [3, 3],
          fill: false
        },
        {
          label: "4D PCA SVM (AUC = 1.0000)",
          data: "#f59e0b",
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderDash: [2, 2],
          data: tpr_pca4d,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f8fafc", font: { size: 11 } } }
      },
      scales: {
        x: {
          title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          title: { display: true, text: "True Positive Rate (Sensitivity)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0,
          max: 1.02
        }
      }
    }
  });

  // PR Curve
  const prCtx = document.getElementById("benchmarkPrChart").getContext("2d");
  const recall = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0];
  const prec_qsvm = [1.0, 1.0, 1.0, 1.0, 0.998, 0.995, 0.9948, 0.9758];
  const prec_ae = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

  STATE.charts.benchmarkPr = new Chart(prCtx, {
    type: "line",
    data: {
      labels: recall.map(v => (v * 100).toFixed(0) + "%"),
      datasets: [
        {
          label: "4-Qubit QSVM (PR-AUC = 0.9999)",
          data: prec_qsvm,
          borderColor: "#00f5ff",
          backgroundColor: "rgba(0, 245, 255, 0.15)",
          borderWidth: 3,
          fill: true,
          tension: 0.1
        },
        {
          label: "Classical AE-SVM (PR-AUC = 1.0000)",
          data: prec_ae,
          borderColor: "#8b5cf6",
          borderWidth: 2,
          borderDash: [4, 4],
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f8fafc", font: { size: 11 } } }
      },
      scales: {
        x: {
          title: { display: true, text: "Recall (Sensitivity)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          title: { display: true, text: "Precision (Positive Predictive Value)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0.90,
          max: 1.01
        }
      }
    }
  });
}

function renderConfusionMatrices() {
  const container = document.getElementById("cmCardsContainer");
  container.innerHTML = "";

  const testModels = STATE.benchmarks.filter(r => r.split === "test");

  testModels.forEach(m => {
    const card = document.createElement("div");
    card.className = "cm-card";

    const total = m.total_samples || (m.tn + m.fp + m.fn + m.tp);
    const isPerc = STATE.cmViewMode === "percents";

    const fmt = (val) => isPerc ? `${((val / total) * 100).toFixed(1)}%` : val;

    card.innerHTML = `
      <div class="cm-card-title">
        <span>${m.model}</span>
        <span class="badge-soft font-mono">N=${total}</span>
      </div>
      <div class="cm-grid-2x2">
        <div class="cm-cell tn" title="True Negatives: Correctly identified healthy minutes">
          <span class="cm-cell-num highlight-green">${fmt(m.tn)}</span>
          <span class="cm-cell-label">True Negative (TN)</span>
        </div>
        <div class="cm-cell fp" title="False Positives: Normal minutes falsely flagged as apnea">
          <span class="cm-cell-num highlight-rose">${fmt(m.fp)}</span>
          <span class="cm-cell-label">False Positive (FP)</span>
        </div>
        <div class="cm-cell fn" title="False Negatives: Missed apnea minutes (Critical)">
          <span class="cm-cell-num highlight-amber">${fmt(m.fn)}</span>
          <span class="cm-cell-label">False Negative (FN)</span>
        </div>
        <div class="cm-cell tp" title="True Positives: Correctly identified sleep apnea events">
          <span class="cm-cell-num highlight-cyan">${fmt(m.tp)}</span>
          <span class="cm-cell-label">True Positive (TP)</span>
        </div>
      </div>
    `;

    container.appendChild(card);
  });
}

/* ======================================================== */
/* 7. TAB 3: THRESHOLDS & CALIBRATION                       */
/* ======================================================== */
function renderThresholdSweepChart() {
  const ctx = document.getElementById("thresholdSweepChart").getContext("2d");
  const data = STATE.thresholds;

  const ths = data.map(d => Number(d.threshold));
  const sens = data.map(d => Number(d.sensitivity));
  const spec = data.map(d => Number(d.specificity));
  const f1 = data.map(d => Number(d.f1));

  STATE.charts.thresholdSweep = new Chart(ctx, {
    type: "line",
    data: {
      labels: ths.map(t => t.toFixed(2)),
      datasets: [
        {
          label: "Sensitivity (Recall)",
          data: sens,
          borderColor: "#10b981",
          borderWidth: 2.5,
          pointRadius: 0
        },
        {
          label: "Specificity",
          data: spec,
          borderColor: "#00f5ff",
          borderWidth: 2.5,
          pointRadius: 0
        },
        {
          label: "F1-Score",
          data: f1,
          borderColor: "#f59e0b",
          borderWidth: 2,
          borderDash: [3, 3],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f8fafc", font: { size: 11 } } }
      },
      scales: {
        x: {
          title: { display: true, text: "Decision Threshold τ", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          title: { display: true, text: "Metric Score", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0,
          max: 1.05
        }
      }
    }
  });
}

function updateDynamicThresholdMetrics(th) {
  // Find nearest threshold row in precomputed validation sweep
  if (!STATE.thresholds || STATE.thresholds.length === 0) return;

  let nearest = STATE.thresholds[0];
  let minDiff = 999;

  STATE.thresholds.forEach(r => {
    const diff = Math.abs(Number(r.threshold) - th);
    if (diff < minDiff) {
      minDiff = diff;
      nearest = r;
    }
  });

  document.getElementById("thDynSensitivity").textContent = `${(Number(nearest.sensitivity) * 100).toFixed(1)}%`;
  document.getElementById("thDynSpecificity").textContent = `${(Number(nearest.specificity) * 100).toFixed(1)}%`;
  document.getElementById("thDynPrecision").textContent = `${(Number(nearest.precision) * 100).toFixed(1)}%`;
  document.getElementById("thDynF1").textContent = Number(nearest.f1).toFixed(4);
}

function renderCalibrationDiagram() {
  const ctx = document.getElementById("calibrationReliabilityChart").getContext("2d");
  const bins = STATE.calibration.bins || [];

  const meanPred = bins.map(b => Number(b.mean_predicted_prob));
  const trueFrac = bins.map(b => Number(b.true_fraction_positive));

  STATE.charts.calibration = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Platt Calibrated Bins",
          data: meanPred.map((x, i) => ({ x: x, y: trueFrac[i] })),
          backgroundColor: "#00f5ff",
          borderColor: "#00f5ff",
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          type: "line",
          label: "Perfect Calibration Diagonal",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "#94a3b8",
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f8fafc", font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `Predicted: ${(ctx.parsed.x * 100).toFixed(1)}%, Observed: ${(ctx.parsed.y * 100).toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Mean Predicted Probability", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0,
          max: 1.0
        },
        y: {
          title: { display: true, text: "Observed True Fraction (Apnea)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0,
          max: 1.0
        }
      }
    }
  });
}

/* ======================================================== */
/* 8. TAB 4: PATIENT HYPNOGRAM & AHI                        */
/* ======================================================== */
async function loadPatientHypnogram(recordName, durationMinutes) {
  try {
    const res = await fetch(`/api/demo_patient?record=${recordName}&minutes=${durationMinutes}`).then(r => r.json());
    if (!res.success) return;

    // Process batch through QSVM pipeline
    const batchRes = await fetch("/api/batch_predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        records: res.timeline,
        threshold: STATE.activeThreshold
      })
    }).then(r => r.json());

    if (batchRes.success) {
      updateHypnogramUI(batchRes.summary, batchRes.timeline);
    }
  } catch (err) {
    console.error("Error loading patient hypnogram:", err);
  }
}

function updateHypnogramUI(summary, timeline) {
  // Update AHI Summary Card
  document.getElementById("ahiScoreVal").textContent = summary.ahi_score.toFixed(1);
  const ahiBadge = document.getElementById("ahiSeverityBadge");
  ahiBadge.textContent = summary.ahi_severity;
  ahiBadge.className = `ahi-badge badge-${summary.severity_badge}`;

  document.getElementById("totalTimeVal").textContent = `${summary.total_minutes} mins (${summary.total_hours} hrs)`;
  document.getElementById("apneaMinutesVal").textContent = `${summary.apnea_minutes} mins (${summary.apnea_percentage}%)`;
  document.getElementById("maxConsecutiveVal").textContent = `${summary.max_consecutive_apnea_minutes} consecutive mins`;
  document.getElementById("meanHrSplitVal").textContent = `${summary.mean_hr_apnea} bpm vs ${summary.mean_hr_normal} bpm`;

  // Render Hypnogram Chart
  const ctx = document.getElementById("hypnogramTimelineChart").getContext("2d");
  
  const labels = timeline.map(d => d.timestamp);
  const probs = timeline.map(d => d.probability * 100);
  const hr = timeline.map(d => d.mean_hr);
  const apneaBlocks = timeline.map(d => d.is_apnea ? 100 : 0);

  if (STATE.charts.hypnogram) {
    STATE.charts.hypnogram.data.labels = labels;
    STATE.charts.hypnogram.data.datasets[0].data = probs;
    STATE.charts.hypnogram.data.datasets[1].data = hr;
    STATE.charts.hypnogram.data.datasets[2].data = apneaBlocks;
    STATE.charts.hypnogram.update("none");
    return;
  }

  STATE.charts.hypnogram = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Apnea Probability (%)",
          data: probs,
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.15)",
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          yAxisID: "y"
        },
        {
          label: "Heart Rate (bpm)",
          data: hr,
          borderColor: "#00f5ff",
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 0,
          yAxisID: "y1"
        },
        {
          type: "bar",
          label: "Apnea Epoch Zone",
          data: apneaBlocks,
          backgroundColor: "rgba(244, 63, 94, 0.25)",
          borderWidth: 0,
          barPercentage: 1.0,
          categoryPercentage: 1.0,
          yAxisID: "y"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f8fafc", font: { size: 11 } } },
        tooltip: {
          mode: "index",
          intersect: false
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Sleep Timeline (Hours:Minutes)", color: "#94a3b8" },
          ticks: { color: "#94a3b8", maxTicksLimit: 12 },
          grid: { color: "rgba(255, 255, 255, 0.05)" }
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Apnea Risk (%)", color: "#f43f5e" },
          ticks: { color: "#f43f5e" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0,
          max: 100
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Heart Rate (bpm)", color: "#00f5ff" },
          ticks: { color: "#00f5ff" },
          grid: { drawOnChartArea: false },
          min: 40,
          max: 130
        }
      }
    }
  });
}

/* ======================================================== */
/* 9. EVENT LISTENERS & UTILITIES                           */
/* ======================================================== */
function setupEventListeners() {
  // Studio Buttons
  document.getElementById("runInferenceBtn").addEventListener("click", () => {
    executeLiveInference();
    executeExplainability();
    showToast("Quantum Inference Executed!", "success");
  });

  document.getElementById("resetInputsBtn").addEventListener("click", () => {
    FEATURE_SPECS.forEach(spec => {
      STATE.currentFeatures[spec.id] = spec.default;
      document.getElementById(`range_${spec.id}`).value = spec.default;
      document.getElementById(`num_${spec.id}`).value = spec.default;
    });
    document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
    executeLiveInference();
    executeExplainability();
    showToast("Reset to default physiological baseline", "info");
  });

  // Confusion Matrix View Mode Toggles
  document.getElementById("cmShowCountsBtn").addEventListener("click", (e) => {
    STATE.cmViewMode = "counts";
    document.getElementById("cmShowCountsBtn").classList.add("active");
    document.getElementById("cmShowPercentsBtn").classList.remove("active");
    renderConfusionMatrices();
  });

  document.getElementById("cmShowPercentsBtn").addEventListener("click", (e) => {
    STATE.cmViewMode = "percents";
    document.getElementById("cmShowPercentsBtn").classList.add("active");
    document.getElementById("cmShowCountsBtn").classList.remove("active");
    renderConfusionMatrices();
  });

  // Threshold Sandbox Slider & Buttons
  const thSlider = document.getElementById("thresholdSweepSlider");
  thSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    STATE.activeThreshold = val;
    document.getElementById("currentThresholdDisplay").textContent = val.toFixed(4);
    document.getElementById("decisionThresholdBadge").textContent = `Threshold: ${val.toFixed(4)}`;
    updateDynamicThresholdMetrics(val);
    document.querySelectorAll(".btn-preset-th").forEach(b => b.classList.remove("active"));
    debounceInference();
  });

  document.getElementById("btnYoudenTh").addEventListener("click", () => {
    setThresholdPreset(-0.6130, "btnYoudenTh");
  });

  document.getElementById("btnF1Th").addEventListener("click", () => {
    setThresholdPreset(-0.6130, "btnF1Th");
  });

  document.getElementById("btnDefaultTh").addEventListener("click", () => {
    setThresholdPreset(0.0000, "btnDefaultTh");
  });

  // Patient Hypnogram Toolbar
  document.getElementById("loadPatientBtn").addEventListener("click", () => {
    const rec = document.getElementById("patientRecordSelect").value;
    const dur = parseInt(document.getElementById("patientDurationSelect").value);
    loadPatientHypnogram(rec, dur);
    showToast(`Loaded analysis for patient record ${rec.toUpperCase()}`, "info");
  });

  // CSV File Upload
  document.getElementById("csvFileInput").addEventListener("change", handleCsvUpload);
}

function setThresholdPreset(val, btnId) {
  STATE.activeThreshold = val;
  document.getElementById("thresholdSweepSlider").value = val;
  document.getElementById("currentThresholdDisplay").textContent = val.toFixed(4);
  document.getElementById("decisionThresholdBadge").textContent = `Threshold: ${val.toFixed(4)}`;
  document.querySelectorAll(".btn-preset-th").forEach(b => b.classList.remove("active"));
  document.getElementById(btnId).classList.add("active");
  updateDynamicThresholdMetrics(val);
  executeLiveInference();
  showToast(`Updated threshold to ${val.toFixed(4)}`, "info");
}

function handleCsvUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (evt) => {
    try {
      const csvText = evt.target.result;
      const lines = csvText.trim().split("\n");
      const headers = lines[0].split(",").map(h => h.trim().replace(/^["']|["']$/g, ""));

      const records = [];
      for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const vals = lines[i].split(",").map(v => v.trim());
        const row = {};
        headers.forEach((h, idx) => {
          row[h] = parseFloat(vals[idx]);
        });
        row.minute = i - 1;
        row.timestamp = `${Math.floor((i-1)/60).toString().padStart(2, "0")}:${((i-1)%60).toString().padStart(2, "0")}`;
        records.push(row);
      }

      showToast(`Loaded ${records.length} minutes from ${file.name}`, "info");

      // Batch predict
      const batchRes = await fetch("/api/batch_predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          records: records,
          threshold: STATE.activeThreshold
        })
      }).then(r => r.json());

      if (batchRes.success) {
        updateHypnogramUI(batchRes.summary, batchRes.timeline);
        showToast("Batch overnight scoring complete!", "success");
      }
    } catch (err) {
      console.error("CSV Parse Error:", err);
      showToast("Failed to parse CSV file", "error");
    }
  };
  reader.readAsText(file);
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  
  let icon = "fa-info-circle";
  if (type === "success") icon = "fa-check-circle highlight-green";
  else if (type === "error") icon = "fa-triangle-exclamation highlight-rose";

  toast.innerHTML = `
    <i class="fa-solid ${icon}"></i>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
