(function () {
  "use strict";

  var elements = {};
  var pollTimer = null;
  var activeSessionId = null;
  var audioPreviewUrls = {
    take: null
  };
  var storageKeys = {
    apiBaseUrl: "konopro.apiBaseUrl",
    betaUserKey: "konopro.betaUserKey"
  };

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    initLandingInteractions();
    initScoringConsole();
  });

  function cacheElements() {
    elements = {
      hamburger: document.getElementById("hamburger"),
      sidebar: document.getElementById("sidebar"),
      sidebarOverlay: document.getElementById("sidebarOverlay"),
      modal: document.getElementById("revealModal"),
      waitlistForm: document.getElementById("waitlistForm"),
      formFeedback: document.getElementById("formFeedback"),
      healthCheckButton: document.getElementById("healthCheckButton"),
      healthStatus: document.getElementById("healthStatus"),
      apiBaseUrl: document.getElementById("apiBaseUrl"),
      betaUserKey: document.getElementById("betaUserKey"),
      analysisForm: document.getElementById("analysisForm"),
      youtubeUrl: document.getElementById("youtubeUrl"),
      takeAudioCard: document.querySelector('[data-audio-card="take"]'),
      takeAudio: document.getElementById("takeAudio"),
      takeFileMeta: document.getElementById("takeFileMeta"),
      takeAudioPreview: document.getElementById("takeAudioPreview"),
      takeAudioPlayer: document.getElementById("takeAudioPlayer"),
      takeFileName: document.getElementById("takeFileName"),
      takeFileSize: document.getElementById("takeFileSize"),
      submitButton: document.getElementById("submitButton"),
      clearButton: document.getElementById("clearButton"),
      jobStatus: document.getElementById("jobStatus"),
      jobStatusText: document.getElementById("jobStatusText"),
      emptyResult: document.getElementById("emptyResult"),
      resultLayout: document.getElementById("resultLayout"),
      overallScore: document.getElementById("overallScore"),
      metricGrid: document.getElementById("metricGrid"),
      feedbackList: document.getElementById("feedbackList"),
      warningBox: document.getElementById("warningBox"),
      warningList: document.getElementById("warningList"),
      feedbackForm: document.getElementById("feedbackForm"),
      feedbackButton: document.getElementById("feedbackButton"),
      feedbackStatus: document.getElementById("feedbackStatus"),
      feedbackRating: document.getElementById("feedbackRating"),
      feedbackAnswer: document.getElementById("feedbackAnswer")
    };
  }

  function initLandingInteractions() {
    initSidebar();
    initRevealAnimations();
    initBuilderPreview();
    initModal();
    initWaitlistForm();
    initActiveNav();
  }

  function initSidebar() {
    if (!elements.hamburger || !elements.sidebar || !elements.sidebarOverlay) {
      return;
    }

    function setOpen(open) {
      elements.hamburger.classList.toggle("is-active", open);
      elements.sidebar.classList.toggle("is-open", open);
      elements.sidebarOverlay.classList.toggle("is-visible", open);
      elements.hamburger.setAttribute("aria-expanded", open ? "true" : "false");
    }

    elements.hamburger.addEventListener("click", function () {
      setOpen(!elements.sidebar.classList.contains("is-open"));
    });
    elements.sidebarOverlay.addEventListener("click", function () {
      setOpen(false);
    });
    elements.sidebar.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        setOpen(false);
      });
    });
  }

  function initRevealAnimations() {
    var revealNodes = document.querySelectorAll(".reveal");
    if (!revealNodes.length) {
      return;
    }
    if (!("IntersectionObserver" in window)) {
      revealNodes.forEach(function (node) {
        node.classList.add("is-visible");
      });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    revealNodes.forEach(function (node) {
      observer.observe(node);
    });
  }

  function initBuilderPreview() {
    var steps = document.querySelectorAll(".builder-step");
    var images = document.querySelectorAll(".builder__preview-img");
    if (!steps.length || !images.length) {
      return;
    }

    function activate(previewId) {
      steps.forEach(function (step) {
        step.classList.toggle("is-active", step.getAttribute("data-preview") === previewId);
      });
      images.forEach(function (image) {
        image.classList.toggle("is-active", image.getAttribute("data-preview-img") === previewId);
      });
    }

    steps.forEach(function (step) {
      step.addEventListener("mouseenter", function () {
        activate(step.getAttribute("data-preview"));
      });
      step.addEventListener("focus", function () {
        activate(step.getAttribute("data-preview"));
      });
      step.addEventListener("click", function () {
        activate(step.getAttribute("data-preview"));
      });
    });
  }

  function initModal() {
    if (!elements.modal) {
      return;
    }

    var openers = document.querySelectorAll(".js-open-reveal");
    var closers = elements.modal.querySelectorAll("[data-close-modal]");
    var emailInput = elements.modal.querySelector("#email");

    function open() {
      elements.modal.classList.add("is-open");
      elements.modal.setAttribute("aria-hidden", "false");
      document.body.style.overflow = "hidden";
      if (emailInput) {
        setTimeout(function () {
          emailInput.focus();
        }, 120);
      }
    }

    function close() {
      elements.modal.classList.remove("is-open");
      elements.modal.setAttribute("aria-hidden", "true");
      document.body.style.overflow = "";
    }

    openers.forEach(function (opener) {
      opener.addEventListener("click", open);
    });
    closers.forEach(function (closer) {
      closer.addEventListener("click", close);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && elements.modal.classList.contains("is-open")) {
        close();
      }
    });
  }

  function initWaitlistForm() {
    if (!elements.waitlistForm || !elements.formFeedback) {
      return;
    }

    elements.waitlistForm.addEventListener("submit", function (event) {
      event.preventDefault();
      elements.formFeedback.classList.remove("is-error", "is-loading");
      elements.formFeedback.classList.add("is-success");
      elements.formFeedback.textContent = "베타 등록이 완료되었습니다. 아래 분석 체험도 바로 사용해볼 수 있습니다.";
      elements.waitlistForm.reset();
    });
  }

  function initActiveNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".sidebar__nav a"));
    var sections = links
      .map(function (link) {
        var id = link.getAttribute("href");
        return id && id.charAt(0) === "#" ? document.querySelector(id) : null;
      })
      .filter(Boolean);

    if (!links.length || !sections.length || !("IntersectionObserver" in window)) {
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) {
          return;
        }
        links.forEach(function (link) {
          link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-35% 0px -55% 0px", threshold: 0.01 });

    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  function initScoringConsole() {
    if (!elements.analysisForm) {
      return;
    }

    elements.takeFileMeta.dataset.defaultText = elements.takeFileMeta.textContent;
    elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
    elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    persistSettings();

    elements.healthCheckButton.addEventListener("click", checkHealth);
    elements.analysisForm.addEventListener("submit", submitAnalysis);
    elements.clearButton.addEventListener("click", resetScoringUi);
    elements.feedbackForm.addEventListener("submit", submitFeedback);
    elements.apiBaseUrl.addEventListener("change", persistSettings);
    elements.betaUserKey.addEventListener("change", persistSettings);
    initAudioPicker({
      key: "take",
      card: elements.takeAudioCard,
      input: elements.takeAudio,
      meta: elements.takeFileMeta,
      preview: elements.takeAudioPreview,
      player: elements.takeAudioPlayer,
      fileName: elements.takeFileName,
      fileSize: elements.takeFileSize
    });
  }

  function defaultTesterId() {
    return "tester-" + Math.random().toString(36).slice(2, 8);
  }

  function persistSettings() {
    localStorage.setItem(storageKeys.apiBaseUrl, apiBaseUrl());
    localStorage.setItem(storageKeys.betaUserKey, betaUserKey());
  }

  function apiBaseUrl() {
    return elements.apiBaseUrl.value.trim().replace(/\/+$/, "");
  }

  function betaUserKey() {
    return elements.betaUserKey.value.trim();
  }

  function initAudioPicker(config) {
    if (!config.card || !config.input || !config.meta || !config.preview || !config.player) {
      return;
    }

    var dropzone = config.card.querySelector("[data-audio-dropzone]");
    config.input.addEventListener("change", function () {
      updateAudioPreview(config);
    });

    if (!dropzone) {
      return;
    }

    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        config.input.click();
      }
    });

    ["dragenter", "dragover"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        config.card.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function () {
        config.card.classList.remove("is-dragging");
      });
    });

    dropzone.addEventListener("drop", function (event) {
      event.preventDefault();
      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (!file) {
        return;
      }
      if (!isLikelyAudio(file)) {
        config.meta.textContent = "오디오 파일만 업로드할 수 있습니다.";
        return;
      }
      if (assignFileToInput(config.input, file)) {
        updateAudioPreview(config);
      } else {
        config.meta.textContent = "드래그 업로드가 안 되면 클릭해서 선택하세요.";
      }
    });
  }

  function updateAudioPreview(config) {
    var input = config.input;
    if (!input) {
      return;
    }
    var file = input.files && input.files[0];
    if (!file) {
      resetAudioPicker(config);
      return;
    }
    if (!isLikelyAudio(file)) {
      input.value = "";
      resetAudioPicker(config);
      config.meta.textContent = "오디오 파일만 업로드할 수 있습니다.";
      return;
    }

    revokeAudioPreview(config.key);
    audioPreviewUrls[config.key] = URL.createObjectURL(file);
    config.player.src = audioPreviewUrls[config.key];
    config.player.load();
    config.card.classList.add("is-ready");
    config.preview.classList.remove("is-hidden");
    config.meta.textContent = file.name + " · " + formatBytes(file.size);
    if (config.fileName) {
      config.fileName.textContent = file.name;
    }
    if (config.fileSize) {
      config.fileSize.textContent = formatBytes(file.size);
    }
  }

  function resetAudioPicker(config) {
    revokeAudioPreview(config.key);
    if (config.input) {
      config.input.value = "";
    }
    if (config.card) {
      config.card.classList.remove("is-ready", "is-dragging");
    }
    if (config.preview) {
      config.preview.classList.add("is-hidden");
    }
    if (config.player) {
      config.player.removeAttribute("src");
      config.player.load();
    }
    if (config.meta) {
      config.meta.textContent = config.meta.dataset.defaultText;
    }
    if (config.fileName) {
      config.fileName.textContent = "No file selected";
    }
    if (config.fileSize) {
      config.fileSize.textContent = "";
    }
  }

  function resetAudioPickers() {
    resetAudioPicker({
      key: "take",
      card: elements.takeAudioCard,
      input: elements.takeAudio,
      meta: elements.takeFileMeta,
      preview: elements.takeAudioPreview,
      player: elements.takeAudioPlayer,
      fileName: elements.takeFileName,
      fileSize: elements.takeFileSize
    });
  }

  function revokeAudioPreview(key) {
    if (audioPreviewUrls[key]) {
      URL.revokeObjectURL(audioPreviewUrls[key]);
      audioPreviewUrls[key] = null;
    }
  }

  function assignFileToInput(input, file) {
    if (typeof DataTransfer === "undefined") {
      return false;
    }
    try {
      var transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      return true;
    } catch (_error) {
      return false;
    }
  }

  function isLikelyAudio(file) {
    if (file.type && file.type.indexOf("audio/") === 0) {
      return true;
    }
    return /\.(aac|aif|aiff|flac|m4a|mp3|ogg|opus|wav|webm)$/i.test(file.name);
  }

  function checkHealth() {
    persistSettings();
    setHealth("Checking backend...", "working");
    fetch(apiBaseUrl() + "/health")
      .then(parseResponse)
      .then(function (payload) {
        setHealth("Backend: " + payload.status + " (" + payload.environment + ")", "good");
      })
      .catch(function (error) {
        setHealth("Backend error: " + error.message, "error");
      });
  }

  function submitAnalysis(event) {
    event.preventDefault();
    persistSettings();
    resetResult();

    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];
    if (!betaUserKey()) {
      setJobStatus("Tester ID is required.", "error");
      return;
    }
    if (!takeFile) {
      setJobStatus("내 노래 녹음 파일을 선택하세요.", "error");
      return;
    }

    var data = new FormData();
    data.append("youtube_url", elements.youtubeUrl.value.trim());
    data.append("take_audio", takeFile);

    setBusy(true);
    setJobStatus("Uploading audio...", "working");
    fetch(apiBaseUrl() + "/v1/scoring-jobs", {
      method: "POST",
      headers: { "X-Konopro-Beta-User": betaUserKey() },
      body: data
    })
      .then(parseResponse)
      .then(function (payload) {
        activeSessionId = payload.session.id;
        setJobStatus(statusText(payload), "working");
        pollScoringJob(payload.job.id);
      })
      .catch(function (error) {
        setBusy(false);
        setJobStatus(error.message, "error");
      });
  }

  function pollScoringJob(jobId) {
    clearPollTimer();
    pollTimer = window.setInterval(function () {
      fetch(apiBaseUrl() + "/v1/scoring-jobs/" + jobId, {
        headers: { "X-Konopro-Beta-User": betaUserKey() }
      })
        .then(parseResponse)
        .then(function (payload) {
          activeSessionId = payload.session.id;
          if (payload.job.status === "completed" || payload.scoring_run.status === "completed") {
            clearPollTimer();
            setBusy(false);
            setJobStatus("Analysis completed.", "good");
            renderResult(payload.scoring_run);
            return;
          }
          if (payload.job.status === "failed" || payload.scoring_run.status === "failed") {
            clearPollTimer();
            setBusy(false);
            setJobStatus(
              payload.scoring_run.error_message || payload.job.error_message || "Analysis failed.",
              "error"
            );
            return;
          }
          setJobStatus(statusText(payload), "working");
        })
        .catch(function (error) {
          clearPollTimer();
          setBusy(false);
          setJobStatus(error.message, "error");
        });
    }, 2200);
  }

  function submitFeedback(event) {
    event.preventDefault();
    if (!activeSessionId) {
      elements.feedbackStatus.textContent = "분석 결과가 먼저 필요합니다.";
      return;
    }

    var formData = new FormData(elements.feedbackForm);
    var payload = {
      helped_review: formData.get("helpedReview"),
      rating: Number(elements.feedbackRating.value),
      answer_text: elements.feedbackAnswer.value.trim() || null,
      context: "web_mvp_scoring_result"
    };

    elements.feedbackButton.disabled = true;
    elements.feedbackStatus.textContent = "Saving feedback...";
    fetch(apiBaseUrl() + "/v1/sessions/" + activeSessionId + "/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Konopro-Beta-User": betaUserKey()
      },
      body: JSON.stringify(payload)
    })
      .then(parseResponse)
      .then(function () {
        elements.feedbackStatus.textContent = "Feedback saved.";
      })
      .catch(function (error) {
        elements.feedbackButton.disabled = false;
        elements.feedbackStatus.textContent = error.message;
      });
  }

  function renderResult(scoringRun) {
    var scores = scoringRun.scores || {};
    var metrics = [
      ["Overall", scores.overall_score, "전체 가중 점수"],
      ["Pitch", scores.pitch_accuracy_score, (scores.mean_pitch_error_cents || "--") + " cents avg error"],
      ["Timing", scores.timing_score, (scores.timing_offset_s || "--") + "s offset"],
      ["Stability", scores.stability_score, (scores.pitch_stability_cents || "--") + " cents spread"],
      ["Coverage", scores.coverage_score, (scores.note_coverage_pct || "--") + "% voiced coverage"],
      ["Confidence", scores.recording_confidence_score, scores.recording_confidence_level || "unknown"]
    ];

    elements.overallScore.textContent = numericScore(scores.overall_score);
    elements.metricGrid.innerHTML = metrics.map(metricCard).join("");
    renderList(elements.feedbackList, scoringRun.feedback || []);
    renderWarnings(scoringRun.warnings || []);
    elements.emptyResult.classList.add("is-hidden");
    elements.resultLayout.classList.remove("is-hidden");
    elements.feedbackButton.disabled = false;
  }

  function metricCard(metric) {
    return (
      '<article class="metric-card">' +
      "<span>" + escapeHtml(metric[0]) + "</span>" +
      "<strong>" + numericScore(metric[1]) + "</strong>" +
      "<small>" + escapeHtml(String(metric[2] || "")) + "</small>" +
      "</article>"
    );
  }

  function renderList(target, items) {
    var safeItems = items.length ? items : ["No feedback generated."];
    target.innerHTML = safeItems.map(function (item) {
      return "<li>" + escapeHtml(item) + "</li>";
    }).join("");
  }

  function renderWarnings(warnings) {
    if (!warnings.length) {
      elements.warningBox.classList.add("is-hidden");
      elements.warningList.innerHTML = "";
      return;
    }
    elements.warningBox.classList.remove("is-hidden");
    elements.warningList.innerHTML = warnings.map(function (warning) {
      return "<li>" + escapeHtml(warning) + "</li>";
    }).join("");
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
    var source = payload.scoring_run.reference_source === "upload" ? "uploaded reference" : "YouTube reference";
    return payload.job.status + " · " + payload.scoring_run.status + " · " + source;
  }

  function resetScoringUi() {
    clearPollTimer();
    setBusy(false);
    elements.analysisForm.reset();
    elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
    elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    resetAudioPickers();
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

  function parseResponse(response) {
    return response.text().then(function (text) {
      var payload = text ? JSON.parse(text) : {};
      if (!response.ok) {
        throw new Error(errorMessage(payload, response.status));
      }
      return payload;
    });
  }

  function errorMessage(payload, status) {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      return payload.detail.map(function (item) {
        return item.msg;
      }).join("; ");
    }
    return "Request failed with status " + status;
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
    var units = ["B", "KB", "MB", "GB"];
    var index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    var value = bytes / Math.pow(1024, index);
    return value.toFixed(value >= 10 || index === 0 ? 0 : 1) + " " + units[index];
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
