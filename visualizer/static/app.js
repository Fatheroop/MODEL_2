/**
 * ORGANISM COGNITIVE STREAM // CLIENT JAVASCRIPT
 * Retrieves telemetry over HTTP, renders Seeing/Thinking/Speaking/Listening, and manages controls.
 */

// =================================================================================================
// DOM REFERENCES
// =================================================================================================

// Header Ticker & Badges
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const navStep = document.getElementById("nav-step");
const navFps = document.getElementById("nav-fps");
const navAction = document.getElementById("nav-action");

const metReward = document.getElementById("met-reward");
const metNovelty = document.getElementById("met-novelty");
const metDelta = document.getElementById("met-delta");
const metError = document.getElementById("met-error");
const metProgress = document.getElementById("met-progress");
const metConfidence = document.getElementById("met-confidence");
const metVocab = document.getElementById("met-vocab");
const metScenes = document.getElementById("met-scenes");

// Controls
const btnPausePlay = document.getElementById("btn-pause-play");
const labelPlayPause = document.getElementById("label-play-pause");
const iconPlayPause = document.getElementById("icon-play-pause");
const btnStep = document.getElementById("btn-step");
const btnSleep = document.getElementById("btn-sleep");
const btnReward = document.getElementById("btn-reward");
const btnPunish = document.getElementById("btn-punish");
const btnReset = document.getElementById("btn-reset");

// 1. SEEING
const canvasRetina = document.getElementById("canvas-retina");
const ctxRetina = canvasRetina.getContext("2d");
const canvasRetinaOverlay = document.getElementById("canvas-retina-overlay");
const ctxRetinaOverlay = canvasRetinaOverlay.getContext("2d");
const seeingActionBadge = document.getElementById("seeing-action-badge");
const seeingPageInfo = document.getElementById("seeing-page-info");
const seeingClickInfo = document.getElementById("seeing-click-info");
const seeingText = document.getElementById("seeing-text");
const lblSeeingNov = document.getElementById("lbl-seeing-nov");
const barSeeingNov = document.getElementById("bar-seeing-nov");
const lblSeeingMotion = document.getElementById("lbl-seeing-motion");
const barSeeingMotion = document.getElementById("bar-seeing-motion");

// 2. THINKING
const thinkingThought = document.getElementById("thinking-thought");
const thinkingConcepts = document.getElementById("thinking-concepts");
const thinkingWorkspace = document.getElementById("thinking-workspace");
const dValCuriosity = document.getElementById("d-val-curiosity");
const dValBoredom = document.getElementById("d-val-boredom");
const dValFatigue = document.getElementById("d-val-fatigue");
const dValConf = document.getElementById("d-val-conf");

// 3. SPEAKING
const speakingCurrent = document.getElementById("speaking-current");
const speakingHistory = document.getElementById("speaking-history");
const chkAudioSpeak = document.getElementById("chk-audio-speak");

// 4. LISTENING
const listeningCurrent = document.getElementById("listening-current");
const listeningHistory = document.getElementById("listening-history");
const formSpeak = document.getElementById("form-speak");
const inputSpeech = document.getElementById("input-speech");
const presetButtons = document.querySelectorAll(".preset-chip");

// Hyperparameters
const paramK = document.getElementById("param-k");
const paramEncEta = document.getElementById("param-enc-eta");
const paramPolEta = document.getElementById("param-pol-eta");
const paramGamma = document.getElementById("param-gamma");
const paramLam = document.getElementById("param-lam");
const paramAlloc = document.getElementById("param-alloc");
const paramDelay = document.getElementById("param-delay");
const paramSleep = document.getElementById("param-sleep");

const valK = document.getElementById("val-k");
const valEncEta = document.getElementById("val-enc-eta");
const valPolEta = document.getElementById("val-pol-eta");
const valGamma = document.getElementById("val-gamma");
const valLam = document.getElementById("val-lam");
const valAlloc = document.getElementById("val-alloc");
const valDelay = document.getElementById("val-delay");
const valSleep = document.getElementById("val-sleep");
const btnApplyHyperparams = document.getElementById("btn-apply-hyperparams");

// Local Client State
let isPaused = false;
let lastSpokenText = "";
const retinaImage = new Image();

// =================================================================================================
// RENDERERS
// =================================================================================================

function renderSeeing(seeingData) {
  if (!seeingData) return;

  // 1. Draw Retinal Image
  if (seeingData.image) {
    retinaImage.onload = () => {
      ctxRetina.drawImage(retinaImage, 0, 0, canvasRetina.width, canvasRetina.height);
      drawFixationOverlay(seeingData.action);
    };
    retinaImage.src = seeingData.image;
  }

  // 2. Update Metadata
  seeingPageInfo.textContent = `Page: ${seeingData.page_id + 1}/${seeingData.total_pages} (${seeingData.pages_visited} visited)`;
  seeingClickInfo.textContent = `Clicks: ${seeingData.effective_clicks}/${seeingData.total_clicks}`;
  seeingText.textContent = seeingData.scene_text || "No readable text on current visual frame.";

  // 3. Update Visual Meters
  lblSeeingNov.textContent = seeingData.novelty.toFixed(2);
  barSeeingNov.style.width = `${Math.min(100, Math.max(0, seeingData.novelty * 100))}%`;

  lblSeeingMotion.textContent = seeingData.change.toFixed(2);
  barSeeingMotion.style.width = `${Math.min(100, Math.max(0, seeingData.change * 100))}%`;

  // 4. Action Badge
  if (seeingData.action) {
    if (seeingData.action.is_click && seeingData.action.col !== null) {
      seeingActionBadge.textContent = `CLICK @ (${seeingData.action.col},${seeingData.action.row})`;
    } else {
      seeingActionBadge.textContent = seeingData.action.name.toUpperCase();
    }
  }
}

function drawFixationOverlay(action) {
  const w = canvasRetinaOverlay.width;
  const h = canvasRetinaOverlay.height;
  ctxRetinaOverlay.clearRect(0, 0, w, h);

  const cols = 8;
  const rows = 4;
  const cellW = w / cols;
  const cellH = h / rows;

  // Subtle fixation grid
  ctxRetinaOverlay.strokeStyle = "rgba(0, 240, 255, 0.14)";
  ctxRetinaOverlay.lineWidth = 0.5;

  for (let c = 1; c < cols; c++) {
    ctxRetinaOverlay.beginPath();
    ctxRetinaOverlay.moveTo(c * cellW, 0);
    ctxRetinaOverlay.lineTo(c * cellW, h);
    ctxRetinaOverlay.stroke();
  }
  for (let r = 1; r < rows; r++) {
    ctxRetinaOverlay.beginPath();
    ctxRetinaOverlay.moveTo(0, r * cellH);
    ctxRetinaOverlay.lineTo(w, r * cellH);
    ctxRetinaOverlay.stroke();
  }

  // If action is click on fixation cell, draw glowing reticle
  if (action && action.is_click && action.col !== null) {
    const x0 = action.col * cellW;
    const y0 = action.row * cellH;

    // Glowing cell target
    ctxRetinaOverlay.fillStyle = "rgba(0, 240, 255, 0.28)";
    ctxRetinaOverlay.fillRect(x0, y0, cellW, cellH);

    ctxRetinaOverlay.strokeStyle = "#00f0ff";
    ctxRetinaOverlay.lineWidth = 1.0;
    ctxRetinaOverlay.strokeRect(x0, y0, cellW, cellH);

    // Crosshair
    const cx = x0 + cellW / 2;
    const cy = y0 + cellH / 2;
    ctxRetinaOverlay.strokeStyle = "#ffffff";
    ctxRetinaOverlay.beginPath();
    ctxRetinaOverlay.moveTo(cx - 3, cy);
    ctxRetinaOverlay.lineTo(cx + 3, cy);
    ctxRetinaOverlay.moveTo(cx, cy - 3);
    ctxRetinaOverlay.lineTo(cx, cy + 3);
    ctxRetinaOverlay.stroke();
  }
}

function renderThinking(thinkingData) {
  if (!thinkingData) return;

  // 1. Inner Monologue
  if (thinkingData.current_thought) {
    thinkingThought.textContent = thinkingData.current_thought;
  }

  // 2. Active Concept Chips
  if (thinkingData.active_concepts && thinkingData.active_concepts.length > 0) {
    thinkingConcepts.innerHTML = thinkingData.active_concepts
      .map((c) => `<span class="concept-tag">${escapeHtml(c)}</span>`)
      .join("");
  } else {
    thinkingConcepts.innerHTML = `<span class="empty-hint">Forming conceptual clusters...</span>`;
  }

  // 3. Workspace Items Competition
  if (thinkingData.workspace_items && thinkingData.workspace_items.length > 0) {
    thinkingWorkspace.innerHTML = thinkingData.workspace_items
      .map(
        (it) => `
        <div class="ws-item font-mono">
          <span>[${escapeHtml(it.src).toUpperCase()}] ${escapeHtml(it.words.join(" "))}</span>
          <span style="color: var(--violet); font-weight: 700;">${Math.round(it.salience * 100)}%</span>
        </div>`
      )
      .join("");
  } else {
    thinkingWorkspace.innerHTML = `<div class="empty-hint">No items currently competing in workspace.</div>`;
  }

  // 4. Drives
  dValCuriosity.textContent = thinkingData.curiosity.toFixed(2);
  dValBoredom.textContent = thinkingData.boredom.toFixed(2);
  dValFatigue.textContent = thinkingData.fatigue.toFixed(2);
  dValConf.textContent = thinkingData.confidence.toFixed(2);
}

function renderSpeaking(speakingData) {
  if (!speakingData) return;

  // 1. Current Spoken Utterance
  const spokenText = speakingData.last_spoken || "(Listening...)";
  speakingCurrent.textContent = `"${spokenText}"`;

  // 2. Web Speech API Vocalization (if toggled)
  if (chkAudioSpeak.checked && spokenText !== lastSpokenText && "speechSynthesis" in window) {
    lastSpokenText = spokenText;
    const utterance = new SpeechSynthesisUtterance(spokenText);
    utterance.rate = 1.05;
    utterance.pitch = 0.95;
    window.speechSynthesis.speak(utterance);
  }

  // 3. Speech History
  if (speakingData.history && speakingData.history.length > 0) {
    speakingHistory.innerHTML = speakingData.history
      .slice(-6)
      .reverse()
      .map(
        (h) => `
        <div class="history-item">
          <span class="h-time font-mono">${escapeHtml(h.time)}</span>
          <span class="h-text">&ldquo;${escapeHtml(h.text)}&rdquo;</span>
        </div>`
      )
      .join("");
  }
}

function renderListening(listeningData) {
  if (!listeningData) return;

  // 1. Last Heard Text
  listeningCurrent.textContent = listeningData.last_heard || "(No external input received yet)";

  // 2. Listening History
  if (listeningData.history && listeningData.history.length > 0) {
    listeningHistory.innerHTML = listeningData.history
      .slice(-6)
      .reverse()
      .map(
        (h) => `
        <div class="history-item">
          <span class="h-time font-mono">${escapeHtml(h.time)}</span>
          <span class="h-text">${escapeHtml(h.text)}</span>
        </div>`
      )
      .join("");
  }
}

function renderTickerAndMetrics(data) {
  navStep.textContent = data.step || 0;
  navFps.textContent = `${data.fps || 0.0}`;
  if (data.seeing && data.seeing.action) {
    navAction.textContent = data.seeing.action.name.toUpperCase();
  }

  if (data.thinking) {
    const r = data.thinking.reward;
    metReward.textContent = `${r >= 0 ? "+" : ""}${r.toFixed(2)}`;
    metReward.style.color = r > 0 ? "var(--emerald)" : r < 0 ? "var(--rose)" : "var(--cyan)";
    metDelta.textContent = `${data.thinking.td_delta >= 0 ? "+" : ""}${data.thinking.td_delta.toFixed(2)}`;
    metError.textContent = data.thinking.forward_error.toFixed(3);
    metProgress.textContent = `${data.thinking.learning_progress >= 0 ? "+" : ""}${data.thinking.learning_progress.toFixed(3)}`;
    metConfidence.textContent = data.thinking.confidence.toFixed(2);
    metScenes.textContent = `${data.thinking.engrams_count} / ${data.thinking.engrams_capacity}`;
  }

  if (data.seeing) {
    metNovelty.textContent = data.seeing.novelty.toFixed(2);
  }

  if (data.speaking) {
    metVocab.textContent = `${data.speaking.vocabulary_size} words`;
  }

  if (typeof data.running === "boolean") {
    isPaused = !data.running;
    labelPlayPause.textContent = isPaused ? "Resume" : "Pause";
    iconPlayPause.textContent = isPaused ? "▶" : "⏸";
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// =================================================================================================
// HTTP & SSE DATA SYNC
// =================================================================================================

function applyTelemetry(data) {
  if (!data) return;
  renderTickerAndMetrics(data);
  renderSeeing(data.seeing);
  renderThinking(data.thinking);
  renderSpeaking(data.speaking);
  renderListening(data.listening);
}

// Connect to Server via SSE with automatic HTTP fallback
function initDataStream() {
  const evtSource = new EventSource("/api/stream");

  evtSource.onopen = () => {
    statusBadge.classList.remove("offline");
    statusBadge.classList.add("online");
    statusText.textContent = "HTTP STREAM ACTIVE";
  };

  evtSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      applyTelemetry(data);
    } catch (err) {
      console.error("Failed to parse telemetry:", err);
    }
  };

  evtSource.onerror = () => {
    statusBadge.classList.remove("online");
    statusBadge.classList.add("offline");
    statusText.textContent = "POLLING HTTP FALLBACK";
    evtSource.close();
    // Fall back to periodic HTTP GET polling until stream recovers
    pollHttpFallback();
  };
}

let pollingTimer = null;
async function pollHttpFallback() {
  if (pollingTimer) return;
  pollingTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/telemetry");
      if (res.ok) {
        const data = await res.json();
        applyTelemetry(data);
        // Try reconnecting SSE
        clearInterval(pollingTimer);
        pollingTimer = null;
        initDataStream();
      }
    } catch (err) {
      // Server down, keep retrying
    }
  }, 300);
}

// =================================================================================================
// INTERACTION & CONTROL DISPATCHERS
// =================================================================================================

async function sendControl(action, value = null) {
  try {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value }),
    });
    return await res.json();
  } catch (err) {
    console.error("Control API error:", err);
  }
}

async function speakToOrganism(text) {
  const clean = text.trim();
  if (!clean) return;
  try {
    const res = await fetch("/api/listen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: clean }),
    });
    return await res.json();
  } catch (err) {
    console.error("Listen API error:", err);
  }
}

async function applyHyperparameters() {
  const payload = {
    k_sparsity: parseInt(paramK.value),
    encoder_eta: parseFloat(paramEncEta.value),
    policy_eta: parseFloat(paramPolEta.value),
    policy_gamma: parseFloat(paramGamma.value),
    policy_lam: parseFloat(paramLam.value),
    memory_alloc: parseFloat(paramAlloc.value),
    step_delay_ms: parseInt(paramDelay.value),
    sleep_every: parseInt(paramSleep.value),
  };

  try {
    const res = await fetch("/api/hyperparameters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    btnApplyHyperparams.innerHTML = "<span>&check; Applied!</span>";
    setTimeout(() => {
      btnApplyHyperparams.innerHTML = "<span>Apply Hyperparameters</span>";
    }, 1200);
  } catch (err) {
    console.error("Hyperparameters API error:", err);
  }
}

// Wire Sliders
paramK.oninput = () => (valK.textContent = paramK.value);
paramEncEta.oninput = () => (valEncEta.textContent = parseFloat(paramEncEta.value).toFixed(3));
paramPolEta.oninput = () => (valPolEta.textContent = parseFloat(paramPolEta.value).toFixed(2));
paramGamma.oninput = () => (valGamma.textContent = parseFloat(paramGamma.value).toFixed(2));
paramLam.oninput = () => (valLam.textContent = parseFloat(paramLam.value).toFixed(2));
paramAlloc.oninput = () => (valAlloc.textContent = parseFloat(paramAlloc.value).toFixed(2));
paramDelay.oninput = () => (valDelay.textContent = `${paramDelay.value} ms`);
paramSleep.oninput = () => (valSleep.textContent = `${paramSleep.value} steps`);

// Wire Buttons
btnPausePlay.onclick = () => sendControl(isPaused ? "resume" : "pause");
btnStep.onclick = () => sendControl("step");
btnSleep.onclick = () => sendControl("sleep");
btnReward.onclick = () => sendControl("reward");
btnPunish.onclick = () => sendControl("punish");
btnReset.onclick = () => {
  const seed = Math.floor(Math.random() * 10000);
  sendControl("reset", seed);
};
btnApplyHyperparams.onclick = applyHyperparameters;

// Wire Speech Intercom Form
formSpeak.onsubmit = async (e) => {
  e.preventDefault();
  const text = inputSpeech.value.trim();
  if (!text) return;
  inputSpeech.value = "";
  await speakToOrganism(text);
};

// Wire Quick Presets
presetButtons.forEach((btn) => {
  btn.onclick = () => {
    const phrase = btn.getAttribute("data-phrase");
    if (phrase) {
      speakToOrganism(phrase);
    }
  };
});

// Start Stream
initDataStream();
