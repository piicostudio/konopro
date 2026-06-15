(function () {
  "use strict";

  var elements = {};
  var pollTimer = null;
  var processingStageTimer = null;
  var processingStageIndex = 0;
  var activeSessionId = null;
  var audioPreviewUrls = {
    take: null
  };
  var storageKeys = {
    apiBaseUrl: "konopro.apiBaseUrl",
    betaUserKey: "konopro.betaUserKey",
    workflowAnswers: "konopro.workflowQuestionAnswers"
  };
  var questionNames = ["karaokeUse", "deviceContext", "appInstall", "resultPriority"];
  var processingStages = [
    {
      title: "Loading YouTube reference",
      text: "Downloading the original singer's track so your take can be compared against the same song."
    },
    {
      title: "Preparing both audio files",
      text: "Normalizing the upload and reference so pitch and timing can be measured on the same footing."
    },
    {
      title: "Tracing pitch differences",
      text: "Following the melody contour and finding where your notes drift sharp, flat, or stay centered."
    },
    {
      title: "Checking timing alignment",
      text: "Matching phrases between your recording and the reference to estimate timing offset."
    },
    {
      title: "Measuring stability and coverage",
      text: "Checking how steady the voiced sections are and how much usable singing was detected."
    },
    {
      title: "Building final feedback",
      text: "Combining the scores into practice notes and warnings you can actually use."
    }
  ];
  var flowState = createInitialFlowState();

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
      uploadStage: document.getElementById("uploadStage"),
      confirmStage: document.getElementById("confirmStage"),
      processingStage: document.getElementById("processingStage"),
      flowStepUpload: document.getElementById("flowStepUpload"),
      flowStepReference: document.getElementById("flowStepReference"),
      flowStepAnalyze: document.getElementById("flowStepAnalyze"),
      youtubeModal: document.getElementById("youtubeModal"),
      youtubeUrlForm: document.getElementById("youtubeUrlForm"),
      youtubeModalUrl: document.getElementById("youtubeModalUrl"),
      youtubeModalError: document.getElementById("youtubeModalError"),
      confirmYoutubeEmbed: document.getElementById("confirmYoutubeEmbed"),
      confirmYoutubeLink: document.getElementById("confirmYoutubeLink"),
      confirmAnalyzeButton: document.getElementById("confirmAnalyzeButton"),
      changeYoutubeButton: document.getElementById("changeYoutubeButton"),
      workflowQuestionForm: document.getElementById("workflowQuestionForm"),
      questionProgress: document.getElementById("questionProgress"),
      resultGate: document.getElementById("resultGate"),
      processingHint: document.getElementById("processingHint"),
      processingStepTitle: document.getElementById("processingStepTitle"),
      processingStepText: document.getElementById("processingStepText"),
      processingStepList: document.getElementById("processingStepList"),
      jobStatus: document.getElementById("jobStatus"),
      jobStatusText: document.getElementById("jobStatusText"),
      result: document.getElementById("result"),
      emptyResult: document.getElementById("emptyResult"),
      resultLayout: document.getElementById("resultLayout"),
      overallScore: document.getElementById("overallScore"),
      metricGrid: document.getElementById("metricGrid"),
      feedbackList: document.getElementById("feedbackList"),
      warningBox: document.getElementById("warningBox"),
      warningList: document.getElementById("warningList"),
      resultTakeFileName: document.getElementById("resultTakeFileName"),
      resultTakeMeta: document.getElementById("resultTakeMeta"),
      resultTakeAudioPlayer: document.getElementById("resultTakeAudioPlayer"),
      resultYoutubeEmbed: document.getElementById("resultYoutubeEmbed"),
      resultYoutubeLink: document.getElementById("resultYoutubeLink")
    };
  }

  function createInitialFlowState() {
    return {
      youtubeUrl: "",
      youtubeVideoId: "",
      scoringRun: null,
      analysisReady: false,
      analysisFailed: false,
      questionsComplete: false,
      questionAnswers: {},
      questionFeedbackSent: false,
      resultRendered: false
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
      elements.formFeedback.textContent = "등록되었습니다. 출시 또는 베타테스트를 진행하면 소식을 보내드릴게요.";
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
    elements.analysisForm.addEventListener("submit", handleUploadStep);
    elements.clearButton.addEventListener("click", resetScoringUi);
    elements.apiBaseUrl.addEventListener("change", persistSettings);
    elements.betaUserKey.addEventListener("change", persistSettings);
    if (elements.youtubeUrlForm) {
      elements.youtubeUrlForm.addEventListener("submit", handleYoutubeUrlSubmit);
    }
    if (elements.youtubeModal) {
      elements.youtubeModal.querySelectorAll("[data-close-youtube-modal]").forEach(function (closer) {
        closer.addEventListener("click", closeYoutubeModal);
      });
    }
    if (elements.confirmAnalyzeButton) {
      elements.confirmAnalyzeButton.addEventListener("click", startConfirmedAnalysis);
    }
    if (elements.changeYoutubeButton) {
      elements.changeYoutubeButton.addEventListener("click", function () {
        openYoutubeModal(flowState.youtubeUrl);
      });
    }
    if (elements.workflowQuestionForm) {
      elements.workflowQuestionForm.addEventListener("change", updateQuestionProgress);
    }
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && elements.youtubeModal && elements.youtubeModal.classList.contains("is-open")) {
        closeYoutubeModal();
      }
    });
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
    renderProcessingStages();
    setProcessingStage(0, "idle");
    showFlowPanel("upload");
    updateQuestionProgress();
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

  function renderProcessingStages() {
    if (!elements.processingStepList) {
      return;
    }
    elements.processingStepList.innerHTML = processingStages.map(function (stage, index) {
      return (
        '<li class="processing-step" data-processing-step="' + index + '">' +
        '<span class="processing-step__copy">' +
        '<span class="processing-step__title">' + escapeHtml(stage.title) + "</span>" +
        '<span class="processing-step__desc">' + escapeHtml(stage.text) + "</span>" +
        "</span>" +
        "</li>"
      );
    }).join("");
  }

  function startProcessingNarrative() {
    clearProcessingStageTimer();
    processingStageIndex = 0;
    setProcessingStage(0, "active");
    processingStageTimer = window.setInterval(function () {
      var nextIndex = Math.min(processingStageIndex + 1, processingStages.length - 1);
      if (nextIndex !== processingStageIndex) {
        setProcessingStage(nextIndex, "active");
      }
    }, 3600);
  }

  function completeProcessingNarrative() {
    clearProcessingStageTimer();
    setProcessingStage(processingStages.length - 1, "done");
    elements.processingStepTitle.textContent = "Analysis complete";
    elements.processingStepText.textContent = "Scores and practice feedback are ready. Finish the quick questions to unlock them.";
  }

  function failProcessingNarrative(message) {
    clearProcessingStageTimer();
    setProcessingStage(processingStageIndex, "failed");
    elements.processingStepTitle.textContent = "Analysis stopped";
    elements.processingStepText.textContent = message || "The backend could not finish this analysis.";
  }

  function setProcessingStage(index, state) {
    if (!elements.processingStepList || !elements.processingStepTitle || !elements.processingStepText) {
      return;
    }
    processingStageIndex = Math.max(0, Math.min(index, processingStages.length - 1));
    var activeStage = processingStages[processingStageIndex];
    elements.processingStepTitle.textContent = activeStage.title;
    elements.processingStepText.textContent = activeStage.text;

    elements.processingStepList.querySelectorAll("[data-processing-step]").forEach(function (item) {
      var itemIndex = Number(item.getAttribute("data-processing-step"));
      item.classList.remove("is-active", "is-done", "is-failed");
      if (state === "idle") {
        return;
      }
      if (state === "done" || itemIndex < processingStageIndex) {
        item.classList.add("is-done");
      } else if (state === "failed" && itemIndex === processingStageIndex) {
        item.classList.add("is-failed");
      } else if (itemIndex === processingStageIndex) {
        item.classList.add("is-active");
      }
    });
  }

  function clearProcessingStageTimer() {
    if (processingStageTimer) {
      window.clearInterval(processingStageTimer);
      processingStageTimer = null;
    }
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

  function handleUploadStep(event) {
    event.preventDefault();
    persistSettings();
    resetResult();
    flowState.analysisReady = false;
    flowState.analysisFailed = false;
    flowState.scoringRun = null;
    flowState.questionFeedbackSent = false;
    flowState.resultRendered = false;

    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];
    if (!betaUserKey()) {
      setJobStatus("Tester ID is required.", "error");
      return;
    }
    if (!takeFile) {
      setJobStatus("내 노래 녹음 파일을 선택하세요.", "error");
      return;
    }

    setJobStatus("Cover uploaded locally. Add the original YouTube link next.", "good");
    openYoutubeModal(flowState.youtubeUrl);
  }

  function openYoutubeModal(defaultUrl) {
    if (!elements.youtubeModal || !elements.youtubeModalUrl) {
      return;
    }
    elements.youtubeModal.classList.add("is-open");
    elements.youtubeModal.setAttribute("aria-hidden", "false");
    elements.youtubeModalUrl.value = defaultUrl || elements.youtubeUrl.value || "";
    elements.youtubeModalError.textContent = "";
    elements.youtubeModalError.classList.remove("is-error", "is-success");
    document.body.style.overflow = "hidden";
    setTimeout(function () {
      elements.youtubeModalUrl.focus();
    }, 120);
  }

  function closeYoutubeModal() {
    if (!elements.youtubeModal) {
      return;
    }
    elements.youtubeModal.classList.remove("is-open");
    elements.youtubeModal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }

  function handleYoutubeUrlSubmit(event) {
    event.preventDefault();
    var url = elements.youtubeModalUrl.value.trim();
    var videoId = extractYoutubeVideoId(url);
    if (!videoId) {
      elements.youtubeModalError.classList.add("is-error");
      elements.youtubeModalError.textContent = "Paste a valid YouTube watch, youtu.be, shorts, live, or embed URL.";
      return;
    }

    flowState.youtubeUrl = url;
    flowState.youtubeVideoId = videoId;
    elements.youtubeUrl.value = url;
    renderReferenceConfirmation();
    closeYoutubeModal();
    showFlowPanel("confirm");
    setJobStatus("Reference loaded. Confirm this is the original song you sang.", "good");
  }

  function renderReferenceConfirmation() {
    var embedUrl = youtubeEmbedUrl(flowState.youtubeVideoId);
    elements.confirmYoutubeEmbed.src = embedUrl;
    elements.confirmYoutubeLink.href = flowState.youtubeUrl;
    elements.confirmYoutubeLink.textContent = flowState.youtubeUrl;
    elements.resultYoutubeEmbed.src = embedUrl;
    elements.resultYoutubeLink.href = flowState.youtubeUrl;
    elements.resultYoutubeLink.textContent = flowState.youtubeUrl;
  }

  function startConfirmedAnalysis() {
    persistSettings();
    resetResult();

    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];
    if (!takeFile) {
      showFlowPanel("upload");
      setJobStatus("내 노래 녹음 파일을 선택하세요.", "error");
      return;
    }
    if (!flowState.youtubeUrl || !flowState.youtubeVideoId) {
      openYoutubeModal("");
      setJobStatus("원곡 YouTube URL을 먼저 입력하세요.", "error");
      return;
    }

    var data = new FormData();
    data.append("youtube_url", flowState.youtubeUrl);
    data.append("take_audio", takeFile);

    activeSessionId = null;
    flowState.analysisReady = false;
    flowState.analysisFailed = false;
    flowState.scoringRun = null;
    flowState.questionFeedbackSent = false;
    flowState.resultRendered = false;
    flowState.questionAnswers = {};
    flowState.questionsComplete = false;
    resetQuestionInputs();
    updateQuestionProgress();
    showFlowPanel("processing");
    startProcessingNarrative();
    setBusy(true);
    setJobStatus("Uploading audio and starting analysis...", "working");
    fetch(apiBaseUrl() + "/v1/scoring-jobs", {
      method: "POST",
      headers: { "X-Konopro-Beta-User": betaUserKey() },
      body: data
    })
      .then(parseResponse)
      .then(function (payload) {
        activeSessionId = payload.session.id;
        setJobStatus(statusText(payload), "working");
        submitWorkflowFeedback();
        pollScoringJob(payload.job.id);
      })
      .catch(function (error) {
        setBusy(false);
        setJobStatus(error.message, "error");
        elements.processingHint.textContent = "The backend could not start analysis. Check the URL, backend status, and upload format.";
        failProcessingNarrative("The backend could not start analysis. Check the URL, backend status, and upload format.");
        flowState.analysisFailed = true;
        tryUnlockResult();
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
            setJobStatus("Analysis is ready.", "good");
            flowState.analysisReady = true;
            flowState.scoringRun = payload.scoring_run;
            elements.processingHint.textContent = "Analysis finished. Complete the quick questions to unlock the result.";
            completeProcessingNarrative();
            tryUnlockResult();
            return;
          }
          if (payload.job.status === "failed" || payload.scoring_run.status === "failed") {
            clearPollTimer();
            setBusy(false);
            setJobStatus(
              payload.scoring_run.error_message || payload.job.error_message || "Analysis failed.",
              "error"
            );
            elements.processingHint.textContent = "Analysis failed before a score could be generated.";
            failProcessingNarrative(payload.scoring_run.error_message || payload.job.error_message || "Analysis failed before a score could be generated.");
            flowState.analysisFailed = true;
            tryUnlockResult();
            return;
          }
          setJobStatus(statusText(payload), "working");
        })
        .catch(function (error) {
          clearPollTimer();
          setBusy(false);
          setJobStatus(error.message, "error");
          elements.processingHint.textContent = "Polling failed. Check whether the backend is still running.";
          failProcessingNarrative("Polling failed. Check whether the backend is still running.");
          flowState.analysisFailed = true;
          tryUnlockResult();
        });
    }, 2200);
  }

  function updateQuestionProgress() {
    flowState.questionAnswers = collectQuestionAnswers();
    var answered = Object.keys(flowState.questionAnswers).length;
    flowState.questionsComplete = answered === questionNames.length;
    if (elements.questionProgress) {
      elements.questionProgress.textContent = answered + " of " + questionNames.length + " answered";
    }
    storeWorkflowAnswers();
    submitWorkflowFeedback();
    tryUnlockResult();
  }

  function collectQuestionAnswers() {
    var answers = {};
    questionNames.forEach(function (name) {
      var checked = elements.workflowQuestionForm && elements.workflowQuestionForm.querySelector(
        'input[name="' + name + '"]:checked'
      );
      if (checked) {
        answers[name] = checked.value;
      }
    });
    return answers;
  }

  function storeWorkflowAnswers() {
    try {
      localStorage.setItem(storageKeys.workflowAnswers, JSON.stringify({
        created_at: new Date().toISOString(),
        tester_id: betaUserKey(),
        youtube_url: flowState.youtubeUrl,
        answers: flowState.questionAnswers
      }));
    } catch (_error) {
      // Local feedback capture is best-effort; backend scoring should not depend on it.
    }
  }

  function submitWorkflowFeedback() {
    if (!activeSessionId || !flowState.questionsComplete || flowState.questionFeedbackSent) {
      return;
    }

    flowState.questionFeedbackSent = true;
    var answerText = questionNames.map(function (name) {
      return name + "=" + (flowState.questionAnswers[name] || "");
    }).join(" | ");
    var payload = {
      helped_review: "not_sure",
      rating: 3,
      answer_text: answerText.slice(0, 500),
      context: "web_mvp_processing_questions"
    };

    fetch(apiBaseUrl() + "/v1/sessions/" + activeSessionId + "/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Konopro-Beta-User": betaUserKey()
      },
      body: JSON.stringify(payload)
    })
      .then(parseResponse)
      .catch(function () {
        flowState.questionFeedbackSent = false;
      });
  }

  function tryUnlockResult() {
    if (!elements.resultGate) {
      return;
    }
    if (flowState.analysisFailed) {
      elements.resultGate.classList.add("is-hidden");
      return;
    }
    if (flowState.analysisReady && !flowState.questionsComplete) {
      elements.resultGate.classList.remove("is-hidden");
      elements.resultGate.textContent = "Analysis is ready. Finish the quick questions to unlock your result.";
      return;
    }
    if (!flowState.analysisReady && flowState.questionsComplete) {
      elements.resultGate.classList.remove("is-hidden");
      elements.resultGate.textContent = "Questions complete. Your result will unlock as soon as analysis finishes.";
      return;
    }
    if (!flowState.analysisReady || !flowState.questionsComplete || !flowState.scoringRun) {
      elements.resultGate.classList.add("is-hidden");
      return;
    }
    if (flowState.resultRendered) {
      elements.resultGate.classList.add("is-hidden");
      return;
    }

    elements.resultGate.classList.add("is-hidden");
    showFlowPanel("processing");
    flowState.resultRendered = true;
    renderResult(flowState.scoringRun);
    submitWorkflowFeedback();
  }

  function renderResult(scoringRun) {
    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];
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
    if (takeFile) {
      elements.resultTakeFileName.textContent = takeFile.name;
      elements.resultTakeMeta.textContent = formatBytes(takeFile.size);
      elements.resultTakeAudioPlayer.src = audioPreviewUrls.take || "";
      elements.resultTakeAudioPlayer.load();
    }
    if (flowState.youtubeVideoId) {
      elements.resultYoutubeEmbed.src = youtubeEmbedUrl(flowState.youtubeVideoId);
      elements.resultYoutubeLink.href = flowState.youtubeUrl;
      elements.resultYoutubeLink.textContent = flowState.youtubeUrl;
    }
    elements.metricGrid.innerHTML = metrics.map(metricCard).join("");
    renderList(elements.feedbackList, scoringRun.feedback || []);
    renderWarnings(scoringRun.warnings || []);
    elements.emptyResult.classList.add("is-hidden");
    elements.resultLayout.classList.remove("is-hidden");
    elements.processingHint.textContent = "Result unlocked.";
    elements.result.scrollIntoView({ behavior: "smooth", block: "start" });
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

  function showFlowPanel(panelName) {
    toggleHidden(elements.uploadStage, panelName !== "upload");
    toggleHidden(elements.confirmStage, panelName !== "confirm");
    toggleHidden(elements.processingStage, panelName !== "processing");

    setFlowStep(elements.flowStepUpload, panelName === "upload", panelName === "confirm" || panelName === "processing");
    setFlowStep(elements.flowStepReference, panelName === "confirm", panelName === "processing");
    setFlowStep(elements.flowStepAnalyze, panelName === "processing", flowState.analysisReady && flowState.questionsComplete);
  }

  function setFlowStep(step, isActive, isDone) {
    if (!step) {
      return;
    }
    step.classList.toggle("is-active", isActive);
    step.classList.toggle("is-done", isDone);
  }

  function toggleHidden(node, hidden) {
    if (node) {
      node.classList.toggle("is-hidden", hidden);
    }
  }

  function setBusy(isBusy) {
    elements.submitButton.disabled = isBusy;
    elements.clearButton.disabled = isBusy;
    if (elements.confirmAnalyzeButton) {
      elements.confirmAnalyzeButton.disabled = isBusy;
    }
    if (elements.changeYoutubeButton) {
      elements.changeYoutubeButton.disabled = isBusy;
    }
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
    clearProcessingStageTimer();
    setBusy(false);
    elements.analysisForm.reset();
    elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
    elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    elements.youtubeUrl.value = "";
    if (elements.youtubeModalUrl) {
      elements.youtubeModalUrl.value = "";
    }
    resetAudioPickers();
    resetQuestionInputs();
    closeYoutubeModal();
    flowState = createInitialFlowState();
    activeSessionId = null;
    resetResult();
    showFlowPanel("upload");
    updateQuestionProgress();
    setProcessingStage(0, "idle");
    setJobStatus("대기 중", "idle");
    elements.processingHint.textContent = "The backend is fetching the original song and comparing it against your recording.";
  }

  function resetResult() {
    elements.overallScore.textContent = "--";
    elements.metricGrid.innerHTML = "";
    elements.feedbackList.innerHTML = "";
    elements.warningList.innerHTML = "";
    elements.warningBox.classList.add("is-hidden");
    elements.resultLayout.classList.add("is-hidden");
    elements.emptyResult.classList.remove("is-hidden");
    elements.resultTakeFileName.textContent = "Cover recording";
    elements.resultTakeMeta.textContent = "Selected audio appears here.";
    elements.resultTakeAudioPlayer.removeAttribute("src");
    elements.resultTakeAudioPlayer.load();
    elements.resultYoutubeEmbed.removeAttribute("src");
    elements.resultYoutubeLink.href = "#";
    elements.resultYoutubeLink.textContent = "Open YouTube";
  }

  function resetQuestionInputs() {
    if (!elements.workflowQuestionForm) {
      return;
    }
    elements.workflowQuestionForm.querySelectorAll("input[type='radio']").forEach(function (input) {
      input.checked = false;
    });
    elements.resultGate.classList.add("is-hidden");
    elements.questionProgress.textContent = "0 of " + questionNames.length + " answered";
  }

  function clearPollTimer() {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function extractYoutubeVideoId(rawUrl) {
    if (!rawUrl) {
      return "";
    }
    try {
      var url = new URL(rawUrl);
      var host = url.hostname.replace(/^www\./, "").replace(/^m\./, "");
      var segments = url.pathname.split("/").filter(Boolean);
      var videoId = "";
      if (host === "youtu.be") {
        videoId = segments[0] || "";
      } else if (host === "youtube.com" || host === "music.youtube.com") {
        if (url.pathname === "/watch") {
          videoId = url.searchParams.get("v") || "";
        } else if (segments[0] === "embed" || segments[0] === "shorts" || segments[0] === "live") {
          videoId = segments[1] || "";
        }
      }
      return /^[A-Za-z0-9_-]{6,}$/.test(videoId) ? videoId : "";
    } catch (_error) {
      return "";
    }
  }

  function youtubeEmbedUrl(videoId) {
    return "https://www.youtube.com/embed/" + encodeURIComponent(videoId);
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
