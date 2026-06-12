const elements = {
  healthCheckButton: document.querySelector("#healthCheckButton"),
  healthStatus: document.querySelector("#healthStatus"),
  apiBaseUrl: document.querySelector("#apiBaseUrl"),
  betaUserKey: document.querySelector("#betaUserKey"),
  analysisForm: document.querySelector("#analysisForm"),
  youtubeUrl: document.querySelector("#youtubeUrl"),
  takeAudio: document.querySelector("#takeAudio"),
  referenceAudio: document.querySelector("#referenceAudio"),
  takeFileMeta: document.querySelector("#takeFileMeta"),
  referenceFileMeta: document.querySelector("#referenceFileMeta"),
  submitButton: document.querySelector("#submitButton"),
  clearButton: document.querySelector("#clearButton"),
  runtimePill: document.querySelector("#runtimePill"),
  jobStatus: document.querySelector("#jobStatus"),
  jobStatusText: document.querySelector("#jobStatusText"),
  emptyResult: document.querySelector("#emptyResult"),
  resultLayout: document.querySelector("#resultLayout"),
  overallScore: document.querySelector("#overallScore"),
  metricGrid: document.querySelector("#metricGrid"),
  feedbackList: document.querySelector("#feedbackList"),
  warningBox: document.querySelector("#warningBox"),
  warningList: document.querySelector("#warningList"),
  feedbackForm: document.querySelector("#feedbackForm"),
  feedbackButton: document.querySelector("#feedbackButton"),
  feedbackStatus: document.querySelector("#feedbackStatus"),
  feedbackRating: document.querySelector("#feedbackRating"),
  feedbackAnswer: document.querySelector("#feedbackAnswer"),
};

const storageKeys = {
  apiBaseUrl: "konopro.apiBaseUrl",
  betaUserKey: "konopro.betaUserKey",
};

let pollTimer = null;
let activeSessionId = null;

init();

function init() {
  elements.takeFileMeta.dataset.defaultText = elements.takeFileMeta.textContent;
  elements.referenceFileMeta.dataset.defaultText = elements.referenceFileMeta.textContent;
  elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
  elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
  persistSettings();
  bindEvents();
}

function bindEvents() {
  elements.healthCheckButton.addEventListener("click", checkHealth);
  elements.analysisForm.addEventListener("submit", submitAnalysis);
  elements.clearButton.addEventListener("click", resetUi);
  elements.feedbackForm.addEventListener("submit", submitFeedback);
  elements.apiBaseUrl.addEventListener("change", persistSettings);
  elements.betaUserKey.addEventListener("change", persistSettings);
  elements.takeAudio.addEventListener("change", () => updateFileMeta(elements.takeAudio, elements.takeFileMeta));
  elements.referenceAudio.addEventListener("change", () => updateFileMeta(elements.referenceAudio, elements.referenceFileMeta));
}

function defaultTesterId() {
  return `tester-${Math.random().toString(36).slice(2, 8)}`;
}

function persistSettings() {
  localStorage.setItem(storageKeys.apiBaseUrl, apiBaseUrl());
  localStorage.setItem(storageKeys.betaUserKey, elements.betaUserKey.value.trim());
  elements.runtimePill.textContent = apiBaseUrl().replace(/^https?:\/\//, "");
}

function apiBaseUrl() {
  return elements.apiBaseUrl.value.trim().replace(/\/+$/, "");
}

function betaUserKey() {
  return elements.betaUserKey.value.trim();
}

function updateFileMeta(input, meta) {
  const file = input.files?.[0];
  meta.textContent = file ? `${file.name} · ${formatBytes(file.size)}` : meta.dataset.defaultText || meta.textContent;
}

async function checkHealth() {
  persistSettings();
  setHealth("Checking backend...", "working");
  try {
    const response = await fetch(`${apiBaseUrl()}/health`);
    const payload = await parseResponse(response);
    setHealth(`Backend: ${payload.status} (${payload.environment})`, "good");
  } catch (error) {
    setHealth(`Backend error: ${error.message}`, "error");
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  persistSettings();
  resetResult();

  const takeFile = elements.takeAudio.files?.[0];
  const referenceFile = elements.referenceAudio.files?.[0];
  if (!betaUserKey()) {
    setJobStatus("Tester ID is required.", "error");
    return;
  }
  if (!takeFile) {
    setJobStatus("내 노래 녹음 파일을 선택하세요.", "error");
    return;
  }

  const data = new FormData();
  data.append("youtube_url", elements.youtubeUrl.value.trim());
  data.append("take_audio", takeFile);
  if (referenceFile) {
    data.append("reference_audio", referenceFile);
  }

  setBusy(true);
  setJobStatus("Uploading audio...", "working");
  try {
    const response = await fetch(`${apiBaseUrl()}/v1/scoring-jobs`, {
      method: "POST",
      headers: {"X-Konopro-Beta-User": betaUserKey()},
      body: data,
    });
    const payload = await parseResponse(response);
    activeSessionId = payload.session.id;
    setJobStatus(statusText(payload), "working");
    pollScoringJob(payload.job.id);
  } catch (error) {
    setBusy(false);
    setJobStatus(error.message, "error");
  }
}

function pollScoringJob(jobId) {
  clearPollTimer();
  pollTimer = window.setInterval(async () => {
    try {
      const response = await fetch(`${apiBaseUrl()}/v1/scoring-jobs/${jobId}`, {
        headers: {"X-Konopro-Beta-User": betaUserKey()},
      });
      const payload = await parseResponse(response);
      activeSessionId = payload.session.id;
      const jobStatus = payload.job.status;
      const runStatus = payload.scoring_run.status;

      if (jobStatus === "completed" || runStatus === "completed") {
        clearPollTimer();
        setBusy(false);
        setJobStatus("Analysis completed.", "good");
        renderResult(payload.scoring_run);
      } else if (jobStatus === "failed" || runStatus === "failed") {
        clearPollTimer();
        setBusy(false);
        setJobStatus(payload.scoring_run.error_message || payload.job.error_message || "Analysis failed.", "error");
      } else {
        setJobStatus(statusText(payload), "working");
      }
    } catch (error) {
      clearPollTimer();
      setBusy(false);
      setJobStatus(error.message, "error");
    }
  }, 2200);
}

async function submitFeedback(event) {
  event.preventDefault();
  if (!activeSessionId) {
    elements.feedbackStatus.textContent = "분석 결과가 먼저 필요합니다.";
    return;
  }

  const helpedReview = new FormData(elements.feedbackForm).get("helpedReview");
  const payload = {
    helped_review: helpedReview,
    rating: Number(elements.feedbackRating.value),
    answer_text: elements.feedbackAnswer.value.trim() || null,
    context: "web_mvp_scoring_result",
  };

  elements.feedbackButton.disabled = true;
  elements.feedbackStatus.textContent = "Saving feedback...";
  try {
    const response = await fetch(`${apiBaseUrl()}/v1/sessions/${activeSessionId}/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Konopro-Beta-User": betaUserKey(),
      },
      body: JSON.stringify(payload),
    });
    await parseResponse(response);
    elements.feedbackStatus.textContent = "Feedback saved.";
  } catch (error) {
    elements.feedbackButton.disabled = false;
    elements.feedbackStatus.textContent = error.message;
  }
}

function renderResult(scoringRun) {
  const scores = scoringRun.scores || {};
  const metrics = [
    ["Overall", scores.overall_score, "전체 가중 점수"],
    ["Pitch", scores.pitch_accuracy_score, `${scores.mean_pitch_error_cents ?? "--"} cents avg error`],
    ["Timing", scores.timing_score, `${scores.timing_offset_s ?? "--"}s offset`],
    ["Stability", scores.stability_score, `${scores.pitch_stability_cents ?? "--"} cents spread`],
    ["Coverage", scores.coverage_score, `${scores.note_coverage_pct ?? "--"}% voiced coverage`],
    ["Confidence", scores.recording_confidence_score, scores.recording_confidence_level || "unknown"],
  ];

  elements.overallScore.textContent = numericScore(scores.overall_score);
  elements.metricGrid.innerHTML = metrics.map(metricCard).join("");
  renderList(elements.feedbackList, scoringRun.feedback || []);
  renderWarnings(scoringRun.warnings || []);

  elements.emptyResult.classList.add("is-hidden");
  elements.resultLayout.classList.remove("is-hidden");
  elements.feedbackButton.disabled = false;
}

function metricCard([label, value, detail]) {
  return `
    <article class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong>${numericScore(value)}</strong>
      <small>${escapeHtml(String(detail || ""))}</small>
    </article>
  `;
}

function renderList(target, items) {
  const safeItems = items.length ? items : ["No feedback generated."];
  target.innerHTML = safeItems.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderWarnings(warnings) {
  if (!warnings.length) {
    elements.warningBox.classList.add("is-hidden");
    elements.warningList.innerHTML = "";
    return;
  }
  elements.warningBox.classList.remove("is-hidden");
  elements.warningList.innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
}

function setBusy(isBusy) {
  elements.submitButton.disabled = isBusy;
  elements.clearButton.disabled = isBusy;
}

function setHealth(message, state) {
  elements.healthStatus.textContent = message;
  elements.healthStatus.dataset.state = state;
}

function setJobStatus(message, state) {
  elements.jobStatusText.textContent = message;
  elements.jobStatus.classList.remove("is-working", "is-good", "is-error");
  if (state === "working") {
    elements.jobStatus.classList.add("is-working");
  } else if (state === "good") {
    elements.jobStatus.classList.add("is-good");
  } else if (state === "error") {
    elements.jobStatus.classList.add("is-error");
  }
}

function statusText(payload) {
  const source = payload.scoring_run.reference_source === "upload" ? "uploaded reference" : "YouTube reference";
  return `${payload.job.status} · ${payload.scoring_run.status} · ${source}`;
}

function resetUi() {
  clearPollTimer();
  setBusy(false);
  elements.analysisForm.reset();
  elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
  elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
  elements.takeFileMeta.textContent = "WAV, MP3, M4A, AAC, OGG, FLAC";
  elements.referenceFileMeta.textContent = "YouTube가 실패할 때 사용";
  activeSessionId = null;
  resetResult();
  setJobStatus("대기 중", "idle");
  elements.feedbackButton.disabled = true;
  elements.feedbackStatus.textContent = "";
}

function resetResult() {
  elements.overallScore.textContent = "--";
  elements.metricGrid.innerHTML = "";
  elements.feedbackList.innerHTML = "";
  elements.warningList.innerHTML = "";
  elements.warningBox.classList.add("is-hidden");
  elements.resultLayout.classList.add("is-hidden");
  elements.emptyResult.classList.remove("is-hidden");
}

function clearPollTimer() {
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function parseResponse(response) {
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  return payload;
}

function errorMessage(payload, status) {
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg).join("; ");
  }
  return `Request failed with status ${status}`;
}

function numericScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return Math.round(Number(value)).toString();
}

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
