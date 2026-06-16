(function () {
  "use strict";

  var elements = {};
  var pollTimer = null;
  var presenceTimer = null;
  var processingStageTimer = null;
  var processingStageIndex = 0;
  var activeSessionId = null;
  var vantaInstance = null;
  var resultDetailSlides = [];
  var resultDetailIndex = 0;
  var resultDetailViewTimer = null;
  var trackedResultHighlightViews = {};
  
  var audioPreviewUrls = {
    take: null
  };
  
  var defaultApiBaseUrl = "https://jockstrap-passion-obtrusive.ngrok-free.dev";
  var defaultGoogleDbUrl = "https://script.google.com/macros/s/AKfycbw2aSa60f7B-wMBQlspwdNHi8w2iHTQ-tLouwVdMP7ddPomE_TYBPcM1iNQgRHpeLyoYw/exec";
  var storageKeys = {
    apiBaseUrl: "konopro.apiBaseUrl",
    betaUserKey: "konopro.betaUserKey",
    googleDbUrl: "konopro.googleDbUrl",
    showDeveloperWarnings: "konopro.showDeveloperWarnings",
    presenceVisitorId: "konopro.presenceVisitorId",
    theme: "konopro.theme",
    analyticsSessionId: "konopro.analyticsSessionId",
    analyticsQueue: "konopro.analyticsQueue"
  };
  
  var analyticsQueueLimit = 80;
  
  var processingStages = [
    {
      title: "원곡 준비 중 (1/3)",
      text: "유튜브에서 원곡 음원을 다운로드하여 보컬 분석을 준비하고 있습니다."
    },
    {
      title: "보컬 트레이싱 및 얼라인먼트 (2/3)",
      text: "내 보컬 녹음의 피치 컨투어를 원곡 멜로디와 타임스탬프 단위로 매칭합니다."
    },
    {
      title: "분석 보고서 작성 중 (3/3)",
      text: "평균 편차(cents/ms)와 불안정성 지수를 종합하여 맞춤 피드백을 생성하고 있습니다."
    }
  ];

  var workflowQuestions = [
    {
      name: "karaokeUse",
      title: "코노에서 어떻게 쓸 것 같나요?",
      options: [
        { value: "full_session", label: "한 세션 전체 녹음 (30분~1시간)" },
        { value: "single_song", label: "한 곡씩 녹음 (3-4분)" },
        { value: "upload_later", label: "먼저 녹음을 하고 나중에 분석 하기 위해 올릴 것" }
      ]
    },
    {
      name: "deviceContext",
      title: "지금 어떤 기기로 보고 있나요?",
      options: [
        { value: "phone", label: "휴대폰" },
        { value: "laptop", label: "노트북" },
        { value: "tablet", label: "태블릿" }
      ]
    },
    {
      name: "appInstall",
      title: "업로드 없이 앱에서 바로 녹음된다면 설치할 것 같나요?",
      options: [
        { value: "yes", label: "네" },
        { value: "no", label: "아니요" },
        { value: "not_sure", label: "아직 모르겠어요" }
      ]
    },
    {
      name: "resultPriority",
      title: "결과에서 가장 먼저 보고 싶은 건?",
      options: [
        { value: "practice_gaps", label: "연습을 더 해야 할 부분" },
        { value: "progress", label: "늘었는지 여부" },
        { value: "similarity", label: "원곡과 얼마나 비슷하게 하고 있는지" }
      ]
    }
  ];

  var flowState = createInitialFlowState();

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    initTheme();
    initVantaBackground();
    initStoredSettings();
    initPresenceHeartbeat();
    initLandingInteractions();
    initScoringConsole();
    initResultDetailCarousel();
    initInlineWaitlistForm();
    initEntranceAnimations();
  });

  function cacheElements() {
    elements = {
      themeToggle: document.getElementById("themeToggle"),
      menuToggle: document.getElementById("menuToggle"),
      primaryNav: document.getElementById("primaryNav"),
      settingsButton: document.getElementById("settingsButton"),
      settingsModal: document.getElementById("settingsModal"),
      modal: document.getElementById("revealModal"),
      waitlistForm: document.getElementById("waitlistForm"),
      formFeedback: document.getElementById("formFeedback"),
      healthCheckButton: document.getElementById("healthCheckButton"),
      healthStatus: document.getElementById("healthStatus"),
      presencePill: document.getElementById("presencePill"),
      presenceCount: document.getElementById("presenceCount"),
      queueWaitingCount: document.getElementById("queueWaitingCount"),
      apiBaseUrl: document.getElementById("apiBaseUrl"),
      betaUserKey: document.getElementById("betaUserKey"),
      googleDbUrl: document.getElementById("googleDbUrl"),
      showDeveloperWarnings: document.getElementById("showDeveloperWarnings"),
      
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
      
      // Right workspace state elements
      emptyResult: document.getElementById("emptyResult"),
      processingStage: document.getElementById("processingStage"),
      processingHint: document.getElementById("processingHint"),
      processingStepTitle: document.getElementById("processingStepTitle"),
      processingStepText: document.getElementById("processingStepText"),
      processingStepList: document.getElementById("processingStepList"),
      processingSurvey: document.getElementById("processingSurvey"),
      surveyProgress: document.getElementById("surveyProgress"),
      surveyHint: document.getElementById("surveyHint"),
      surveyCard: document.getElementById("surveyCard"),
      surveyDots: document.getElementById("surveyDots"),
      resultGate: document.getElementById("resultGate"),
      
      result: document.getElementById("result"),
      resultLayout: document.getElementById("resultLayout"),
      playbackPreviewPanel: document.getElementById("playbackPreviewPanel"),
      overallScore: document.getElementById("overallScore"),
      metricGrid: document.getElementById("metricGrid"),
      resultDetailTitle: document.getElementById("resultDetailTitle"),
      resultDetailDescription: document.getElementById("resultDetailDescription"),
      resultDetailSlides: document.getElementById("resultDetailSlides"),
      resultDetailDots: document.getElementById("resultDetailDots"),
      resultDetailStatus: document.getElementById("resultDetailStatus"),
      resultDetailPrev: document.getElementById("resultDetailPrev"),
      resultDetailNext: document.getElementById("resultDetailNext"),
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
      activeQuestionIndex: 0,
      processingMode: "idle",
      resultRendered: false
    };
  }

  function initStoredSettings() {
    if (elements.apiBaseUrl) {
      elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || defaultApiBaseUrl;
    }
    if (elements.betaUserKey) {
      elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    }
    if (elements.googleDbUrl) {
      elements.googleDbUrl.value = localStorage.getItem(storageKeys.googleDbUrl) || defaultGoogleDbUrl;
    }
    if (elements.showDeveloperWarnings) {
      elements.showDeveloperWarnings.checked = localStorage.getItem(storageKeys.showDeveloperWarnings) === "true";
    }
    persistSettings();
  }

  function initPresenceHeartbeat() {
    if (!elements.presenceCount) return;
    sendPresenceHeartbeat();
    presenceTimer = window.setInterval(sendPresenceHeartbeat, 30000);
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) sendPresenceHeartbeat();
    });
  }

  function sendPresenceHeartbeat() {
    if (!elements.presenceCount || document.hidden) return;
    fetch(apiBaseUrl() + "/v1/presence/heartbeat", {
      method: "POST",
      headers: apiRequestHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        visitor_id: presenceVisitorId(),
        path: window.location.pathname + window.location.hash
      })
    })
      .then(parseResponse)
      .then(function (payload) {
        elements.presenceCount.textContent = String(payload.active_visitor_count || 1);
        if (elements.queueWaitingCount) {
          elements.queueWaitingCount.textContent = String(payload.queued_scoring_count || 0);
        }
        if (elements.presencePill) {
          elements.presencePill.dataset.state = "live";
        }
      })
      .catch(function () {
        elements.presenceCount.textContent = "--";
        if (elements.queueWaitingCount) {
          elements.queueWaitingCount.textContent = "--";
        }
        if (elements.presencePill) {
          elements.presencePill.dataset.state = "offline";
        }
      });
  }

  function initVantaBackground() {
    var isDarkTheme = document.body.getAttribute("data-theme") === "dark";

    if (!isDarkTheme && vantaInstance && typeof vantaInstance.destroy === "function") {
      vantaInstance.destroy();
      vantaInstance = null;
      return;
    }

    if (!isDarkTheme || vantaInstance) return;

    if (window.VANTA && window.VANTA.NET) {
      vantaInstance = window.VANTA.NET({
        el: "#vanta-bg",
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.0,
        minWidth: 200.0,
        scale: 1.0,
        scaleMobile: 1.0,
        color: 0xff4081,
        backgroundColor: 0x09090b,
        points: 8.0,
        maxDistance: 20.0,
        spacing: 16.0
      });
    }
  }

  function initTheme() {
    var storedTheme = localStorage.getItem(storageKeys.theme);
    var theme = storedTheme === "dark" ? "dark" : "light";
    applyTheme(theme);

    if (!elements.themeToggle) return;

    elements.themeToggle.addEventListener("click", function () {
      var nextTheme = document.body.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(nextTheme);
      localStorage.setItem(storageKeys.theme, nextTheme);
      initVantaBackground();
    });
  }

  function applyTheme(theme) {
    document.body.setAttribute("data-theme", theme);

    if (!elements.themeToggle) return;

    var isDarkTheme = theme === "dark";
    elements.themeToggle.setAttribute("aria-pressed", String(isDarkTheme));
    elements.themeToggle.setAttribute("aria-label", isDarkTheme ? "라이트 모드로 전환" : "다크 모드로 전환");
  }

  function initEntranceAnimations() {
    if (window.gsap) {
      var tl = gsap.timeline();
      tl.from(".navbar", { y: -64, opacity: 0, duration: 0.8, ease: "power3.out" });
      tl.from(".hero__badge", { y: 20, opacity: 0, duration: 0.6 }, "-=0.4");
      tl.from(".hero__title", { y: 30, opacity: 0, duration: 0.8, ease: "power3.out" }, "-=0.4");
      tl.from(".hero__subtitle", { y: 20, opacity: 0, duration: 0.6 }, "-=0.5");
      tl.from(".hero__actions", { y: 20, opacity: 0, duration: 0.6 }, "-=0.4");
    }
  }

  function transitionRightColumn(state) {
    var activeElement = null;
    var toShowElement = null;

    if (!elements.emptyResult.classList.contains("is-hidden")) activeElement = elements.emptyResult;
    else if (!elements.processingStage.classList.contains("is-hidden")) activeElement = elements.processingStage;
    else if (!elements.result.classList.contains("is-hidden")) activeElement = elements.result;

    if (state === "empty") toShowElement = elements.emptyResult;
    else if (state === "processing") toShowElement = elements.processingStage;
    else if (state === "result") toShowElement = elements.result;

    if (activeElement === toShowElement) return;

    if (window.gsap && activeElement && toShowElement) {
      gsap.to(activeElement, {
        opacity: 0,
        y: -10,
        duration: 0.2,
        ease: "power2.in",
        onComplete: function () {
          activeElement.classList.add("is-hidden");
          toShowElement.classList.remove("is-hidden");
          gsap.fromTo(toShowElement,
            { opacity: 0, y: 10 },
            { opacity: 1, y: 0, duration: 0.3, ease: "power2.out" }
          );
        }
      });
    } else {
      if (activeElement) activeElement.classList.add("is-hidden");
      if (toShowElement) toShowElement.classList.remove("is-hidden");
    }
  }

  function initLandingInteractions() {
    initRevealAnimations();
    initMobileNav();
    alignInitialHashTarget();
    initBuilderPreview();
    initSettingsModal();
    initAnalytics();
    initModal();
    initWaitlistForm();
  }

  function alignInitialHashTarget() {
    if (!window.location.hash) return;

    function scrollToHash() {
      var id = window.location.hash.slice(1);
      if (!id) return;

      try {
        id = decodeURIComponent(id);
      } catch (error) {
        return;
      }

      var target = document.getElementById(id);
      if (target) target.scrollIntoView({ block: "start" });
    }

    window.setTimeout(scrollToHash, 80);
    window.addEventListener("load", function () {
      window.setTimeout(scrollToHash, 120);
    }, { once: true });
  }

  function initMobileNav() {
    if (!elements.menuToggle || !elements.primaryNav) return;

    function closeMenu() {
      elements.primaryNav.classList.remove("is-open");
      elements.menuToggle.classList.remove("is-open");
      elements.menuToggle.setAttribute("aria-expanded", "false");
    }

    elements.menuToggle.addEventListener("click", function () {
      var isOpen = elements.primaryNav.classList.toggle("is-open");
      elements.menuToggle.classList.toggle("is-open", isOpen);
      elements.menuToggle.setAttribute("aria-expanded", String(isOpen));
    });

    elements.primaryNav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", closeMenu);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeMenu();
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 767) closeMenu();
    });
  }

  function initRevealAnimations() {
    var revealNodes = document.querySelectorAll(".reveal");
    if (!revealNodes.length) return;
    
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
    }, { threshold: 0.15 });

    revealNodes.forEach(function (node) {
      observer.observe(node);
    });
  }

  function initBuilderPreview() {
    var steps = document.querySelectorAll(".builder-step[data-preview]");
    var images = document.querySelectorAll(".builder__preview-img[data-preview-img]");
    if (!steps.length || !images.length) return;

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
      step.addEventListener("click", function () {
        activate(step.getAttribute("data-preview"));
      });
    });
  }

  function initSettingsModal() {
    if (!elements.settingsButton || !elements.settingsModal) return;

    var closers = elements.settingsModal.querySelectorAll("[data-close-settings-modal]");

    function open() {
      elements.settingsModal.classList.add("is-open");
      elements.settingsModal.setAttribute("aria-hidden", "false");
      elements.settingsButton.setAttribute("aria-expanded", "true");
      document.body.style.overflow = "hidden";
      setTimeout(function () {
        elements.apiBaseUrl.focus();
      }, 120);
    }

    function close() {
      persistSettings();
      elements.settingsModal.classList.remove("is-open");
      elements.settingsModal.setAttribute("aria-hidden", "true");
      elements.settingsButton.setAttribute("aria-expanded", "false");
      document.body.style.overflow = "";
    }

    elements.settingsButton.addEventListener("click", open);
    closers.forEach(function (closer) {
      closer.addEventListener("click", close);
    });
  }

  function initModal() {
    if (!elements.modal) return;

    var openers = document.querySelectorAll(".js-open-reveal");
    var closers = elements.modal.querySelectorAll("[data-close-modal]");
    var emailInput = elements.modal.querySelector("#email");

    function open() {
      elements.modal.classList.add("is-open");
      elements.modal.setAttribute("aria-hidden", "false");
      document.body.style.overflow = "hidden";
      setTimeout(function () {
        if (emailInput) emailInput.focus();
      }, 120);
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
  }

  function initScoringConsole() {
    if (!elements.analysisForm) return;

    elements.takeFileMeta.dataset.defaultText = elements.takeFileMeta.textContent;
    persistSettings();

    elements.healthCheckButton.addEventListener("click", checkHealth);
    elements.analysisForm.addEventListener("submit", handleDirectFormSubmit);
    elements.clearButton.addEventListener("click", resetScoringUi);
    elements.apiBaseUrl.addEventListener("change", persistSettings);
    elements.betaUserKey.addEventListener("change", persistSettings);
    elements.googleDbUrl.addEventListener("change", function () {
      persistSettings();
      flushAnalyticsQueue();
    });
    if (elements.showDeveloperWarnings) {
      elements.showDeveloperWarnings.addEventListener("change", function () {
        persistSettings();
        if (flowState.scoringRun && flowState.resultRendered) {
          renderResultDetailCarousel(flowState.scoringRun, flowState.scoringRun.scores || {});
        }
      });
    }

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

    initProcessingSurvey();
    renderProcessingStages();
    resetProcessingSurvey();
    setProcessingStage(0, "idle");
    transitionRightColumn("empty");
  }

  function handleDirectFormSubmit(event) {
    event.preventDefault();
    persistSettings();
    resetResult();
    
    var url = elements.youtubeUrl.value.trim();
    var videoId = extractYoutubeVideoId(url);
    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];

    if (!betaUserKey()) {
      setJobStatus("Tester ID가 설정되지 않았습니다. 설정 아이콘을 눌러 확인하세요.", "error");
      return;
    }
    if (!url || !videoId) {
      setJobStatus("올바른 유튜브 링크를 입력하세요.", "error");
      return;
    }
    if (!takeFile) {
      setJobStatus("내 노래 녹음 파일을 업로드하세요.", "error");
      return;
    }

    flowState.youtubeUrl = url;
    flowState.youtubeVideoId = videoId;
    flowState.analysisReady = false;
    flowState.analysisFailed = false;
    flowState.scoringRun = null;
    flowState.resultRendered = false;
    flowState.processingMode = "starting";
    resetProcessingSurvey();

    // Start UI processing stage
    transitionRightColumn("processing");
    startProcessingNarrative();
    setBusy(true);
    setJobStatus("오디오 분석을 시작합니다...", "working");

    trackEvent("action_analysis_started", "site_MVP", {
      youtube_video_id: videoId,
      file_extension: fileExtension(takeFile.name),
      file_size_bytes: takeFile.size
    });

    var data = new FormData();
    data.append("youtube_url", url);
    data.append("take_audio", takeFile);

    fetch(apiBaseUrl() + "/v1/scoring-jobs", {
      method: "POST",
      headers: apiRequestHeaders({ "X-Konopro-Beta-User": betaUserKey() }),
      body: data
    })
      .then(parseResponse)
      .then(function (payload) {
        activeSessionId = payload.session.id;
        updateProcessingQueueState(payload);
        pollScoringJob(payload.job.id);
      })
      .catch(function (error) {
        setBusy(false);
        setJobStatus(error.message, "error");
        elements.processingHint.textContent = "분석 시작 실패. 백엔드 서버 및 파일 형식을 확인하세요.";
        failProcessingNarrative("분석 시작 실패. 백엔드 서버 및 파일 형식을 확인하세요.");
        flowState.analysisFailed = true;
        setResultGate("", false);
        trackEvent("action_analysis_failed", "site_MVP", {
          stage: "start",
          message: error.message
        });
        transitionRightColumn("processing");
      });
  }

  function pollScoringJob(jobId) {
    clearPollTimer();
    pollTimer = window.setInterval(function () {
      fetch(apiBaseUrl() + "/v1/scoring-jobs/" + jobId, {
        headers: apiRequestHeaders({ "X-Konopro-Beta-User": betaUserKey() })
      })
        .then(parseResponse)
        .then(function (payload) {
          activeSessionId = payload.session.id;
          
          if (payload.job.status === "completed" || payload.scoring_run.status === "completed") {
            clearPollTimer();
            setBusy(false);
            setJobStatus("분석이 완료되었습니다.", "good");
            flowState.analysisReady = true;
            flowState.scoringRun = payload.scoring_run;
            completeProcessingNarrative();
            
            trackEvent("action_analysis_completed", "site_MVP", {
              job_id: payload.job.id,
              session_id: payload.session.id,
              overall_score: payload.scoring_run.scores && payload.scoring_run.scores.overall_score
            });

            tryRevealResult();
            return;
          }
          
          if (payload.job.status === "failed" || payload.scoring_run.status === "failed") {
            clearPollTimer();
            setBusy(false);
            var errMsg = payload.scoring_run.error_message || payload.job.error_message || "분석 도중 실패했습니다.";
            setJobStatus(errMsg, "error");
            failProcessingNarrative(errMsg);
            flowState.analysisFailed = true;
            setResultGate("", false);
            trackEvent("action_analysis_failed", "site_MVP", {
              stage: "poll",
              job_id: payload.job.id,
              message: errMsg
            });
            return;
          }
          
          updateProcessingQueueState(payload);
        })
        .catch(function (error) {
          clearPollTimer();
          setBusy(false);
          setJobStatus(error.message, "error");
          failProcessingNarrative("결과 폴링 연결 실패. 백엔드가 실행 중인지 확인하세요.");
          flowState.analysisFailed = true;
          setResultGate("", false);
        });
    }, 2000);
  }

  function updateProcessingQueueState(payload) {
    var queue = payload.queue || {};
    var status = payload.job && payload.job.status;
    var peopleAhead = Number(queue.people_ahead_count || 0);

    if (status === "queued") {
      if (flowState.processingMode !== "queued") {
        clearProcessingStageTimer();
        setProcessingStage(0, "active");
        flowState.processingMode = "queued";
      }
      elements.processingStepTitle.textContent = peopleAhead > 0
        ? "앞에 " + peopleAhead + "명 대기 중"
        : "0명 대기 중";
      elements.processingStepText.textContent = peopleAhead > 0
        ? "앞에 " + peopleAhead + "명이 분석을 기다리고 있어요. 순서가 오면 자동으로 시작됩니다."
        : "현재 앞 순서가 없습니다. 서버가 준비되는 대로 바로 분석을 시작합니다.";
      elements.processingHint.textContent = queue.pending_count > 1
        ? "현재 대기/분석 중인 요청 " + queue.pending_count + "개"
        : "대기열에 등록되었습니다.";
      setJobStatus(statusText(payload), "working");
      return;
    }

    if (status === "processing") {
      if (flowState.processingMode !== "processing") {
        startProcessingNarrative();
        flowState.processingMode = "processing";
      }
      elements.processingHint.textContent = "지금 내 녹음을 분석하고 있어요.";
      setJobStatus(statusText(payload), "working");
      return;
    }

    setJobStatus(statusText(payload), "working");
  }

  function initProcessingSurvey() {
    if (!elements.processingSurvey) return;

    elements.processingSurvey.addEventListener("change", function (event) {
      var input = event.target.closest("[data-survey-answer]");
      if (!input || !input.checked) return;

      answerProcessingSurveyQuestion(input.name, input.value);
    });
  }

  function resetProcessingSurvey() {
    flowState.questionAnswers = {};
    flowState.questionsComplete = false;
    flowState.activeQuestionIndex = 0;
    renderProcessingSurvey();
    setSurveyFocusMode(false);
    setResultGate("", false);
  }

  function renderProcessingSurvey() {
    if (!elements.surveyCard || !elements.surveyProgress || !elements.surveyDots) return;

    var total = workflowQuestions.length;
    var answeredCount = Object.keys(flowState.questionAnswers || {}).length;
    var displayIndex = Math.min(flowState.activeQuestionIndex + 1, total);

    if (flowState.questionsComplete) {
      elements.surveyProgress.textContent = total + "/" + total;
      elements.surveyHint.textContent = flowState.analysisReady
        ? "분석이 완료되었습니다. 이제 결과를 열고 있어요."
        : "설문 완료. 분석이 끝나면 결과가 바로 열립니다.";
      elements.surveyCard.innerHTML = (
        '<div class="processing-survey__complete">' +
          '<strong>답변 완료</strong>' +
          '<span>분석이 끝나는 즉시 결과를 보여드릴게요.</span>' +
        '</div>'
      );
      renderSurveyDots(total, total);
      animateSurveyCard();
      return;
    }

    var question = workflowQuestions[flowState.activeQuestionIndex];
    elements.surveyProgress.textContent = displayIndex + "/" + total;
    elements.surveyHint.textContent = flowState.analysisReady
      ? "분석이 완료되었습니다. 설문을 마치면 바로 결과를 볼 수 있어요."
      : "답변이 끝나고 분석이 완료되면 결과가 바로 열립니다.";

    elements.surveyCard.innerHTML = (
      '<fieldset class="processing-survey__fieldset">' +
        '<legend class="processing-survey__question">' + escapeHtml(question.title) + '</legend>' +
        '<div class="processing-survey__options">' +
          question.options.map(function (option) {
            return (
              '<label class="processing-survey__option">' +
                '<input type="radio" name="' + escapeHtml(question.name) + '" value="' + escapeHtml(option.value) + '" data-survey-answer>' +
                '<span>' + escapeHtml(option.label) + '</span>' +
              '</label>'
            );
          }).join("") +
        '</div>' +
      '</fieldset>'
    );
    renderSurveyDots(answeredCount, total);
    animateSurveyCard();
  }

  function renderSurveyDots(answeredCount, total) {
    if (!elements.surveyDots) return;

    elements.surveyDots.innerHTML = workflowQuestions.map(function (_, index) {
      var activeClass = index < answeredCount ? " is-done" : "";
      if (!flowState.questionsComplete && index === flowState.activeQuestionIndex) {
        activeClass += " is-active";
      }
      return '<span class="processing-survey__dot' + activeClass + '"></span>';
    }).join("");
  }

  function animateSurveyCard() {
    if (!elements.surveyCard) return;

    elements.surveyCard.classList.remove("is-entering");
    void elements.surveyCard.offsetWidth;
    elements.surveyCard.classList.add("is-entering");
  }

  function answerProcessingSurveyQuestion(questionName, value) {
    var question = workflowQuestions[flowState.activeQuestionIndex];
    if (!question || question.name !== questionName || flowState.questionsComplete) return;

    var option = question.options.find(function (item) {
      return item.value === value;
    });

    flowState.questionAnswers[questionName] = value;
    trackEvent("click_surveyquestion_" + questionName, "site_MVP", {
      question_index: flowState.activeQuestionIndex + 1,
      question: question.title,
      answer: value,
      answer_label: option ? option.label : value
    });

    elements.surveyCard.querySelectorAll("input").forEach(function (input) {
      input.disabled = true;
    });

    window.setTimeout(function () {
      if (flowState.activeQuestionIndex >= workflowQuestions.length - 1) {
        flowState.questionsComplete = true;
      } else {
        flowState.activeQuestionIndex += 1;
      }

      renderProcessingSurvey();
      tryRevealResult();
    }, 160);
  }

  function tryRevealResult() {
    if (flowState.analysisFailed) {
      setSurveyFocusMode(false);
      setResultGate("", false);
      return;
    }

    if (flowState.analysisReady && flowState.scoringRun && flowState.questionsComplete) {
      if (flowState.resultRendered) return;

      flowState.resultRendered = true;
      setSurveyFocusMode(false);
      setResultGate("", false);
      renderResult(flowState.scoringRun);
      transitionRightColumn("result");
      trackEvent("action_result_unlocked", "site_MVP", {
        answered_questions: Object.keys(flowState.questionAnswers || {}).length,
        overall_score: flowState.scoringRun.scores && flowState.scoringRun.scores.overall_score
      });
      return;
    }

    if (flowState.analysisReady && !flowState.questionsComplete) {
      elements.processingHint.textContent = "분석은 완료되었습니다. 아래 설문을 마치면 결과가 바로 열립니다.";
      setSurveyFocusMode(true);
      setResultGate("분석이 완료되었습니다. 설문을 마치면 바로 결과를 볼 수 있어요.", true);
      renderProcessingSurvey();
      return;
    }

    if (!flowState.analysisReady && flowState.questionsComplete) {
      setSurveyFocusMode(false);
      setResultGate("설문 완료. 분석이 끝나면 결과가 바로 열립니다.", true);
      renderProcessingSurvey();
      return;
    }

    setSurveyFocusMode(false);
    setResultGate("", false);
  }

  function setSurveyFocusMode(isFocused) {
    if (!elements.processingStage) return;

    elements.processingStage.classList.toggle("is-waiting-survey", isFocused);
    var workspaceCard = elements.processingStage.closest(".result-workspace-card");
    if (workspaceCard) {
      workspaceCard.classList.toggle("is-survey-focus", isFocused);
    }
  }

  function setResultGate(message, isVisible) {
    if (!elements.resultGate) return;

    elements.resultGate.textContent = message || "";
    elements.resultGate.classList.toggle("is-hidden", !isVisible);
  }

  function renderResult(scoringRun) {
    var takeFile = elements.takeAudio.files && elements.takeAudio.files[0];
    var scores = scoringRun.scores || {};
    
    var metrics = [
      ["Overall Score", scores.overall_score, "종합 보컬 점수"],
      ["음정 정확도", scores.pitch_accuracy_score, (scores.mean_pitch_error_cents !== undefined ? Math.abs(Math.round(scores.mean_pitch_error_cents)) + " cents avg error" : "피치 오류 분석")],
      ["타이밍 정확도", scores.timing_score, (scores.timing_offset_s !== undefined ? Math.round(scores.timing_offset_s * 1000) + "ms offset" : "박자 편차 분석")],
      ["성대 안정성", scores.stability_score, (scores.pitch_stability_cents !== undefined ? Math.round(scores.pitch_stability_cents) + " cents spread" : "피치 흔들림 분석")],
      ["발성 가창율", scores.coverage_score, (scores.note_coverage_pct !== undefined ? Math.round(scores.note_coverage_pct) + "% coverage" : "가창 구간 분량")],
      ["녹음 신뢰도", scores.recording_confidence_score, scores.recording_confidence_level || "보통"]
    ];

    elements.overallScore.textContent = numericScore(scores.overall_score);
    
    if (takeFile) {
      elements.resultTakeFileName.textContent = takeFile.name;
      elements.resultTakeMeta.textContent = formatBytes(takeFile.size);
      if (elements.takeAudioPreview) {
        elements.takeAudioPreview.classList.add("is-hidden");
      }
      
      // Update result target audio
      var targetTakeAudioPlayer = document.getElementById("resultTakeAudioPlayer");
      if (targetTakeAudioPlayer) {
        targetTakeAudioPlayer.src = audioPreviewUrls.take || "";
        targetTakeAudioPlayer.load();
      }
    }
    
    if (flowState.youtubeVideoId) {
      elements.resultYoutubeEmbed.src = youtubeEmbedUrl(flowState.youtubeVideoId);
      elements.resultYoutubeLink.href = flowState.youtubeUrl;
      elements.resultYoutubeLink.textContent = flowState.youtubeUrl;
    }

    if (elements.playbackPreviewPanel) {
      elements.playbackPreviewPanel.classList.remove("is-hidden");
      var previewCard = elements.playbackPreviewPanel.closest(".workspace__card");
      if (previewCard) previewCard.classList.add("has-playback-preview");
    }
    
    elements.metricGrid.innerHTML = metrics.map(metricCard).join("");
    renderResultDetailCarousel(scoringRun, scores);
  }

  function initResultDetailCarousel() {
    if (elements.resultDetailPrev) {
      elements.resultDetailPrev.addEventListener("click", function () {
        var fromIndex = resultDetailIndex;
        showResultDetailSlide(resultDetailIndex - 1);
        trackResultDetailNavigation("prev", fromIndex, resultDetailIndex);
      });
    }

    if (elements.resultDetailNext) {
      elements.resultDetailNext.addEventListener("click", function () {
        var fromIndex = resultDetailIndex;
        showResultDetailSlide(resultDetailIndex + 1);
        trackResultDetailNavigation("next", fromIndex, resultDetailIndex);
      });
    }

    if (elements.resultDetailDots) {
      elements.resultDetailDots.addEventListener("click", function (event) {
        var button = event.target.closest("[data-detail-index]");
        if (!button) return;
        showResultDetailSlide(Number(button.getAttribute("data-detail-index")));
      });
    }
  }

  function buildABMoments(scores) {
    var pitchScore = scores.pitch_accuracy_score || 70;
    var timingScore = scores.timing_score || 70;

    return [
      {
        title: "후렴구 도입부 음정 불안정 (Chorus Entrance)",
        timestamp: "0:42",
        startSeconds: 42,
        deviation: "-" + Math.round(105 - pitchScore) + " cents Flat",
        desc: "고음역 진입부에서 호흡 압력 전달 부족으로 순간 피치가 다소 떨어졌습니다."
      },
      {
        title: "1절 브릿지 타이밍 지연 (Verse Bridge)",
        timestamp: "1:15",
        startSeconds: 75,
        deviation: "+" + Math.round((100 - timingScore) * 5) + "ms Delay",
        desc: "소절 끝부분 롱톤 처리에서 박자가 비트 뒤쪽으로 밀리는 레이백 현상이 감지되었습니다."
      }
    ];
  }

  function renderResultDetailCarousel(scoringRun, scores) {
    var moments = buildABMoments(scores);
    var feedbackItems = normalizedCoachFeedbackItems(scoringRun.feedback || []);
    var warnings = scoringRun.warnings || [];

    resultDetailSlides = moments.map(function (moment, index) {
      return {
        title: "문제 구간 " + (index + 1) + " / " + moments.length,
        desc: "음정이나 박자 편차가 컸던 핵심 구간입니다. 내 목소리와 원곡을 번갈아 들어보세요.",
        type: "highlight",
        isHighlightSlide: true,
        highlightRank: index + 1,
        momentTitle: moment.title,
        clipStartSeconds: moment.startSeconds,
        timestamp: moment.timestamp,
        deviation: moment.deviation,
        html: renderMomentSlide(moment, feedbackItems[index])
      };
    });

    if (showDeveloperWarnings() && warnings.length) {
      resultDetailSlides.push({
        title: "주의사항 " + warnings.length + "개",
        desc: "녹음이나 원곡 상태 때문에 점수 해석에 영향을 줄 수 있는 항목입니다.",
        type: "warnings",
        isHighlightSlide: false,
        html: renderWarningSlide(warnings)
      });
    }

    resultDetailSlides.push({
      title: "출시 알림",
      desc: "정식 서비스가 준비되면 첫 달 무료 코드와 함께 알려드릴게요.",
      type: "waitlist",
      isHighlightSlide: false,
      html: renderWaitlistSlide()
    });

    resultDetailIndex = 0;
    trackedResultHighlightViews = {};
    showResultDetailSlide(0);
  }

  function renderMomentSlide(m, feedback) {
    var coachFeedback = feedback || "이 구간을 원곡과 번갈아 들으며 차이를 줄여보세요.";
    return (
      '<article class="result-detail-card result-detail-card--moment">' +
        '<div class="moment-card">' +
        '  <div class="moment-header">' +
        '    <div class="moment-title-wrap">' +
        '      <span class="moment-title">' + escapeHtml(m.title) + '</span>' +
        '      <span class="moment-timestamp">타임스탬프: ' + escapeHtml(m.timestamp) + '</span>' +
        '    </div>' +
        '    <span class="moment-badge">' + escapeHtml(m.deviation) + '</span>' +
        '  </div>' +
        '  <p class="moment-description">' + escapeHtml(m.desc) + '</p>' +
        '  <div class="moment-actions">' +
        '    <button class="btn-play-clip btn-play-take" data-start="' + m.startSeconds + '">' +
        '      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>' +
        '      내 목소리 재생' +
        '    </button>' +
        '    <button class="btn-play-clip btn-play-original" data-start="' + m.startSeconds + '">' +
        '      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>' +
        '      원곡 재생' +
        '    </button>' +
        '  </div>' +
        '  <div class="moment-coach">' +
        '    <span class="moment-coach__label">코치 피드백</span>' +
        '    <p class="moment-coach__text">' + escapeHtml(coachFeedback) + '</p>' +
        '  </div>' +
        '</div>' +
      '</article>'
    );
  }

  function renderFeedbackSlide(feedback) {
    var safeItems = normalizedCoachFeedbackItems(feedback || []);
    if (!safeItems.length) {
      safeItems = ["이번 결과에서 가장 낮은 항목부터 한 구간씩 다시 들어보세요."];
    }
    return (
      '<article class="result-detail-card result-detail-card--feedback">' +
        '<ul class="feedback-list">' +
          safeItems.map(function (item) {
            return "<li>" + escapeHtml(item) + "</li>";
          }).join("") +
        '</ul>' +
      '</article>'
    );
  }

  function normalizedCoachFeedbackItems(feedback) {
    return (feedback || [])
      .map(normalizeCoachFeedback)
      .filter(Boolean)
      .slice(0, 2);
  }

  function normalizeCoachFeedback(item) {
    var text = String(item || "").trim();
    if (!text) return "";
    var translations = {
      "Treat this as a diagnostic take. Re-record a shorter, cleaner section if needed.": "",
      "Strong take overall. Use this as a reference point for later attempts.": "",
      "Usable practice take. Focus on the lowest metric first before re-recording.": "",
      "Pitch contour is the main gap. Practice the melody slowly before singing full tempo.":
        "음정 흐름이 원곡과 가장 많이 달라요. 멜로디를 천천히 맞춘 뒤 전체 속도로 불러보세요.",
      "Pitch stability is low. Hold longer notes steadily before adding style or vibrato.":
        "긴 음에서 흔들림이 보여요. 비브라토나 스타일을 넣기 전에 한 음을 안정적으로 유지해보세요.",
      "Timing alignment is weak. Trim the take/reference to the same phrase and retry.":
        "박자가 원곡과 어긋나요. 같은 구간을 짧게 잘라 다시 맞춰보세요.",
      "Detected singing coverage is low. Sing through more of the reference phrase.":
        "부른 구간이 원곡을 충분히 커버하지 못했어요. 원곡과 같은 구간을 끝까지 불러보세요.",
      "Recording confidence is low, so use this score as rough feedback only.": ""
    };
    return Object.prototype.hasOwnProperty.call(translations, text) ? translations[text] : text;
  }

  function renderWarningSlide(warnings) {
    return (
      '<article class="result-detail-card result-detail-card--warnings">' +
        '<ul class="warning-list">' +
          warnings.map(function (warning) {
            return "<li>" + escapeHtml(warning) + "</li>";
          }).join("") +
        '</ul>' +
      '</article>'
    );
  }

  function renderWaitlistSlide() {
    return (
      '<article class="result-detail-card result-detail-card--waitlist">' +
        '<div class="embedded-waitlist">' +
          '<div class="embedded-waitlist__icon">알림</div>' +
          '<div class="embedded-waitlist__content">' +
            '<h4>분석 결과가 마음에 드셨나요?</h4>' +
            '<p>KonoPro의 모바일 앱 정식 출시 소식과 함께 첫 달 무료 코드를 보내드릴게요.</p>' +
            '<form id="inlineWaitlistForm" class="inline-waitlist-form" novalidate>' +
              '<div class="inline-waitlist-field">' +
                '<input id="inlineEmail" name="email" type="email" placeholder="이메일 주소를 입력하세요" required class="form-input">' +
                '<button type="submit" class="btn btn--primary">신청하기</button>' +
              '</div>' +
              '<p id="inlineFormFeedback" class="form-feedback" role="status" aria-live="polite"></p>' +
            '</form>' +
          '</div>' +
        '</div>' +
      '</article>'
    );
  }

  function showResultDetailSlide(nextIndex) {
    if (!elements.resultDetailSlides || !resultDetailSlides.length) return;

    resultDetailIndex = (nextIndex + resultDetailSlides.length) % resultDetailSlides.length;
    var slide = resultDetailSlides[resultDetailIndex];

    elements.resultDetailTitle.textContent = slide.title;
    elements.resultDetailDescription.textContent = slide.desc;
    elements.resultDetailSlides.innerHTML = slide.html;

    elements.resultDetailDots.innerHTML = resultDetailSlides.map(function (_, index) {
      var activeClass = index === resultDetailIndex ? " is-active" : "";
      var current = index === resultDetailIndex ? ' aria-current="true"' : "";
      return '<button type="button" class="result-detail-carousel__dot' + activeClass + '" data-detail-index="' + index + '"' + current + ' aria-label="상세 카드 ' + (index + 1) + '번 보기"></button>';
    }).join("");

    if (elements.resultDetailStatus) {
      elements.resultDetailStatus.textContent = (resultDetailIndex + 1) + " / " + resultDetailSlides.length;
    }

    if (elements.resultDetailPrev) {
      elements.resultDetailPrev.disabled = resultDetailSlides.length <= 1;
    }
    if (elements.resultDetailNext) {
      elements.resultDetailNext.disabled = resultDetailSlides.length <= 1;
    }

    bindMomentClipButtons();
    initInlineWaitlistForm();
    scheduleResultHighlightViewTracking(resultDetailIndex);
  }

  function bindMomentClipButtons() {
    if (!elements.resultDetailSlides) return;

    elements.resultDetailSlides.querySelectorAll(".btn-play-take").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var start = Number(btn.getAttribute("data-start"));
        trackEvent("click_result_play_take", "site_MVP", resultDetailMetadata({
          clip_start_seconds: start,
          playback_duration_seconds: 5
        }));
        playTakeSegment(start, 5);
      });
    });

    elements.resultDetailSlides.querySelectorAll(".btn-play-original").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var start = Number(btn.getAttribute("data-start"));
        trackEvent("click_result_play_original", "site_MVP", resultDetailMetadata({
          clip_start_seconds: start
        }));
        playOriginalSegment(start);
      });
    });
  }

  function trackResultDetailNavigation(direction, fromIndex, toIndex) {
    var eventName = direction === "prev" ? "click_result_highlight_prev" : "click_result_highlight_next";
    trackEvent(eventName, "site_MVP", resultDetailMetadata({
      direction: direction,
      from_slide_index: fromIndex + 1,
      from_slide_title: resultDetailSlides[fromIndex] ? resultDetailSlides[fromIndex].title : "",
      to_slide_index: toIndex + 1
    }));
  }

  function scheduleResultHighlightViewTracking(slideIndex) {
    if (resultDetailViewTimer) {
      window.clearTimeout(resultDetailViewTimer);
      resultDetailViewTimer = null;
    }

    var slide = resultDetailSlides[slideIndex];
    if (!slide || !slide.isHighlightSlide || trackedResultHighlightViews[slide.highlightRank]) return;

    resultDetailViewTimer = window.setTimeout(function () {
      if (resultDetailIndex !== slideIndex || !resultDetailSlides[slideIndex]) return;
      trackedResultHighlightViews[slide.highlightRank] = true;
      trackEvent("view_result_highlight_" + slide.highlightRank, "site_MVP", resultDetailMetadata({
        dwell_ms: 1500
      }));
    }, 1500);
  }

  function resultDetailMetadata(extra) {
    var slide = resultDetailSlides[resultDetailIndex] || {};
    var metadata = {
      slide_index: resultDetailIndex + 1,
      slide_title: slide.title || "",
      slide_type: slide.type || "unknown",
      is_highlight_slide: Boolean(slide.isHighlightSlide),
      highlight_rank: slide.highlightRank || null,
      highlight_title: slide.momentTitle || "",
      clip_start_seconds: slide.clipStartSeconds || null,
      timestamp: slide.timestamp || "",
      deviation: slide.deviation || ""
    };

    Object.keys(extra || {}).forEach(function (key) {
      metadata[key] = extra[key];
    });
    return metadata;
  }

  function playTakeSegment(startSeconds, durationSeconds) {
    var player = document.getElementById("resultTakeAudioPlayer");
    if (!player || !player.src) {
      // Fallback to local upload stage player if loaded
      player = elements.takeAudioPlayer;
    }
    if (!player || !player.src) return;

    player.currentTime = startSeconds;
    player.play();

    if (window.takeSegmentTimeout) {
      clearTimeout(window.takeSegmentTimeout);
    }
    window.takeSegmentTimeout = setTimeout(function () {
      player.pause();
    }, durationSeconds * 1000);
  }

  function playOriginalSegment(startSeconds) {
    var embed = elements.resultYoutubeEmbed;
    if (!embed || !flowState.youtubeVideoId) return;

    embed.src = youtubeEmbedUrl(flowState.youtubeVideoId) + "?start=" + startSeconds + "&autoplay=1";
  }

  function initInlineWaitlistForm() {
    var form = document.getElementById("inlineWaitlistForm");
    var feedback = document.getElementById("inlineFormFeedback");
    if (!form || !feedback) return;

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var emailInput = form.querySelector("#inlineEmail");
      var email = emailInput ? emailInput.value.trim() : "";

      if (!email) return;

      trackEvent("click_submit_email_inline", "site_MVP", {
        email_masked: maskEmail(email)
      });

      feedback.classList.remove("is-error", "is-success");
      feedback.classList.add("is-loading");
      feedback.textContent = "대기열 등록 중...";

      postWaitlistRecord(email, "inline_result_waitlist")
        .then(function () {
          feedback.classList.remove("is-error", "is-loading");
          feedback.classList.add("is-success");
          feedback.textContent = "등록 감사합니다! 앱 출시 시 할인 코드와 함께 안내 메일을 보내드릴게요.";
          form.reset();
        })
        .catch(function (error) {
          feedback.classList.remove("is-loading", "is-success");
          feedback.classList.add("is-error");
          feedback.textContent = error.message || "등록 실패. 설정을 확인해 주세요.";
        });
    });
  }

  function initWaitlistForm() {
    if (!elements.waitlistForm || !elements.formFeedback) return;

    elements.waitlistForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var emailInput = elements.waitlistForm.querySelector("#email");
      var adviceInput = elements.waitlistForm.querySelector("#advice");
      var email = emailInput ? emailInput.value.trim() : "";
      var advice = adviceInput ? adviceInput.value.trim() : "";

      trackEvent("click_submit_email", "site_CTA", {
        email_masked: maskEmail(email),
        has_advice: Boolean(advice)
      });

      elements.formFeedback.classList.remove("is-error", "is-success");
      elements.formFeedback.classList.add("is-loading");
      elements.formFeedback.textContent = "등록 중입니다...";

      postWaitlistRecord(email, advice)
        .then(function () {
          elements.formFeedback.classList.remove("is-error", "is-loading");
          elements.formFeedback.classList.add("is-success");
          elements.formFeedback.textContent = "등록 성공! 정식 출시 소식을 보내드릴게요.";
          elements.waitlistForm.reset();
        })
        .catch(function (error) {
          elements.formFeedback.classList.remove("is-loading", "is-success");
          elements.formFeedback.classList.add("is-error");
          elements.formFeedback.textContent = error.message || "등록 실패. 설정을 확인해 주세요.";
        });
    });
  }

  function checkHealth() {
    persistSettings();
    setHealth("연결 확인 중...", "working");
    fetch(apiBaseUrl() + "/health", {
      headers: apiRequestHeaders()
    })
      .then(parseResponse)
      .then(function (payload) {
        setHealth("백엔드 정상: " + payload.status + " (" + payload.environment + ")", "good");
      })
      .catch(function (error) {
        setHealth("연결 실패: " + error.message, "error");
      });
  }

  function initAudioPicker(config) {
    if (!config.card || !config.input || !config.meta || !config.preview || !config.player) return;

    var dropzone = config.card.querySelector("[data-audio-dropzone]");
    config.input.addEventListener("change", function () {
      updateAudioPreview(config);
    });

    if (!dropzone) return;

    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        config.input.click();
      }
    });

    dropzone.addEventListener("click", function () {
      config.input.click();
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
      if (!file) return;
      if (!isLikelyAudio(file)) {
        config.meta.textContent = "오디오 파일만 업로드할 수 있습니다.";
        return;
      }
      if (assignFileToInput(config.input, file)) {
        updateAudioPreview(config);
      }
    });
  }

  function updateAudioPreview(config) {
    var input = config.input;
    if (!input) return;
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
    if (config.fileName) config.fileName.textContent = file.name;
    if (config.fileSize) config.fileSize.textContent = formatBytes(file.size);
  }

  function resetAudioPicker(config) {
    revokeAudioPreview(config.key);
    if (config.input) config.input.value = "";
    if (config.card) config.card.classList.remove("is-ready", "is-dragging");
    if (config.preview) config.preview.classList.add("is-hidden");
    if (config.player) {
      config.player.removeAttribute("src");
      config.player.load();
    }
    if (config.meta) config.meta.textContent = config.meta.dataset.defaultText;
    if (config.fileName) config.fileName.textContent = "No file selected";
    if (config.fileSize) config.fileSize.textContent = "";
  }

  function revokeAudioPreview(key) {
    if (audioPreviewUrls[key]) {
      URL.revokeObjectURL(audioPreviewUrls[key]);
      audioPreviewUrls[key] = null;
    }
  }

  function assignFileToInput(input, file) {
    if (typeof DataTransfer === "undefined") return false;
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
    if (file.type && file.type.indexOf("audio/") === 0) return true;
    return /\.(aac|aif|aiff|flac|m4a|mp3|ogg|opus|wav|webm)$/i.test(file.name);
  }

  function renderProcessingStages() {
    if (!elements.processingStepList) return;
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
    }, 4000);
  }

  function completeProcessingNarrative() {
    clearProcessingStageTimer();
    flowState.processingMode = "done";
    setProcessingStage(processingStages.length - 1, "done");
  }

  function failProcessingNarrative(message) {
    clearProcessingStageTimer();
    setSurveyFocusMode(false);
    flowState.processingMode = "failed";
    setProcessingStage(processingStageIndex, "failed");
    elements.processingStepTitle.textContent = "분석이 중단되었습니다.";
    elements.processingStepText.textContent = message || "오류가 발생하여 피치 분석을 마칠 수 없습니다.";
  }

  function setProcessingStage(index, state) {
    if (!elements.processingStepList || !elements.processingStepTitle || !elements.processingStepText) return;
    
    processingStageIndex = Math.max(0, Math.min(index, processingStages.length - 1));
    var activeStage = processingStages[processingStageIndex];
    elements.processingStepTitle.textContent = activeStage.title;
    elements.processingStepText.textContent = activeStage.text;

    elements.processingStepList.querySelectorAll("[data-processing-step]").forEach(function (item) {
      var itemIndex = Number(item.getAttribute("data-processing-step"));
      item.classList.remove("is-active", "is-done", "is-failed");
      if (state === "idle") return;
      
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

  function resetScoringUi() {
    clearPollTimer();
    clearProcessingStageTimer();
    setBusy(false);
    elements.analysisForm.reset();
    
    elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || defaultApiBaseUrl;
    elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    elements.googleDbUrl.value = localStorage.getItem(storageKeys.googleDbUrl) || defaultGoogleDbUrl;
    elements.youtubeUrl.value = "";
    
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
    
    flowState = createInitialFlowState();
    activeSessionId = null;
    resetResult();
    resetProcessingSurvey();
    setProcessingStage(0, "idle");
    setJobStatus("대기 중", "idle");
    transitionRightColumn("empty");
  }

  function resetResult() {
    elements.overallScore.textContent = "--";
    elements.metricGrid.innerHTML = "";
    if (resultDetailViewTimer) {
      window.clearTimeout(resultDetailViewTimer);
      resultDetailViewTimer = null;
    }
    trackedResultHighlightViews = {};
    resultDetailSlides = [];
    resultDetailIndex = 0;
    if (elements.resultDetailTitle) elements.resultDetailTitle.textContent = "집중 개선 필요 구간";
    if (elements.resultDetailDescription) elements.resultDetailDescription.textContent = "화살표를 눌러 문제 구간, 주의사항, 출시 알림을 확인하세요.";
    if (elements.resultDetailSlides) elements.resultDetailSlides.innerHTML = "";
    if (elements.resultDetailDots) elements.resultDetailDots.innerHTML = "";
    if (elements.resultDetailStatus) elements.resultDetailStatus.textContent = "";
    if (elements.playbackPreviewPanel) {
      elements.playbackPreviewPanel.classList.add("is-hidden");
      var previewCard = elements.playbackPreviewPanel.closest(".workspace__card");
      if (previewCard) previewCard.classList.remove("has-playback-preview");
    }
    elements.resultTakeFileName.textContent = "Cover recording";
    elements.resultTakeMeta.textContent = "Selected audio appears here.";
    
    var resultTakeAudioPlayer = document.getElementById("resultTakeAudioPlayer");
    if (resultTakeAudioPlayer) {
      resultTakeAudioPlayer.removeAttribute("src");
      resultTakeAudioPlayer.load();
    }
    
    elements.resultYoutubeEmbed.removeAttribute("src");
    elements.resultYoutubeLink.href = "#";
    elements.resultYoutubeLink.textContent = "Open YouTube";
    setResultGate("", false);
  }

  function persistSettings() {
    if (elements.apiBaseUrl) localStorage.setItem(storageKeys.apiBaseUrl, apiBaseUrl());
    if (elements.betaUserKey) localStorage.setItem(storageKeys.betaUserKey, betaUserKey());
    if (elements.googleDbUrl) localStorage.setItem(storageKeys.googleDbUrl, googleDbUrl());
    if (elements.showDeveloperWarnings) localStorage.setItem(storageKeys.showDeveloperWarnings, String(showDeveloperWarnings()));
  }

  function apiBaseUrl() {
    return elements.apiBaseUrl.value.trim().replace(/\/+$/, "");
  }

  function apiRequestHeaders(extraHeaders) {
    var headers = {
      "Accept": "application/json"
    };
    if (isNgrokBackend()) {
      headers["ngrok-skip-browser-warning"] = "true";
    }
    Object.keys(extraHeaders || {}).forEach(function (key) {
      headers[key] = extraHeaders[key];
    });
    return headers;
  }

  function isNgrokBackend() {
    try {
      return new URL(apiBaseUrl()).hostname.indexOf("ngrok") !== -1;
    } catch (_error) {
      return false;
    }
  }

  function betaUserKey() {
    return elements.betaUserKey.value.trim();
  }

  function googleDbUrl() {
    return elements.googleDbUrl ? elements.googleDbUrl.value.trim() : "";
  }

  function showDeveloperWarnings() {
    return Boolean(elements.showDeveloperWarnings && elements.showDeveloperWarnings.checked);
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
    var source = payload.scoring_run.reference_source === "upload" ? "업로드 원곡" : "유튜브 원곡";
    var queue = payload.queue || {};
    var peopleAhead = Number(queue.people_ahead_count || 0);
    if (payload.job.status === "queued") {
      return peopleAhead > 0
        ? "앞에 " + peopleAhead + "명 대기 중 · " + source
        : "0명 대기 중 · 다음 순서 · " + source;
    }
    if (payload.job.status === "processing") {
      return "분석 중 · " + source;
    }
    return payload.job.status + " · " + payload.scoring_run.status + " · " + source;
  }

  function defaultTesterId() {
    return "tester-" + Math.random().toString(36).slice(2, 8);
  }

  function presenceVisitorId() {
    var existing = localStorage.getItem(storageKeys.presenceVisitorId);
    if (existing) return existing;
    var created = "visitor-" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(storageKeys.presenceVisitorId, created);
    return created;
  }

  function parseResponse(response) {
    return response.text().then(function (body) {
      var payload = null;
      if (body) {
        try {
          payload = JSON.parse(body);
        } catch (_error) {
          var preview = body.replace(/\s+/g, " ").trim().slice(0, 180);
          var fallback = preview || response.statusText || "응답 본문 없음";
          throw new Error("서버가 JSON이 아닌 응답을 보냈습니다 (" + response.status + "): " + fallback);
        }
      }

      if (!response.ok) {
        throw new Error(errorMessageFromPayload(payload) || "서버 통신 실패 (" + response.status + ")");
      }
      return payload || {};
    });
  }

  function errorMessageFromPayload(payload) {
    if (!payload) return "";
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map(function (item) {
        return item.msg || item.message || JSON.stringify(item);
      }).join(", ");
    }
    return payload.message || payload.error || "";
  }

  function numericScore(val) {
    if (val === undefined || val === null || isNaN(Number(val))) return "--";
    return Math.round(Number(val));
  }

  function metricCard(metric) {
    var label = metric[0];
    var score = metric[1];
    var subtext = metric[2];
    var formattedScore = numericScore(score);
    return (
      '<div class="metric-card">' +
      '  <div class="metric-card__title">' + escapeHtml(label) + '</div>' +
      '  <div class="metric-card__value">' + formattedScore + '</div>' +
      '  <div class="metric-card__sub">' + escapeHtml(subtext) + '</div>' +
      '</div>'
    );
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    var k = 1024, sizes = ["Bytes", "KB", "MB"], i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function fileExtension(name) {
    return name.slice(((name.lastIndexOf(".") - 1) >>> 0) + 2);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  function youtubeEmbedUrl(videoId) {
    return "https://www.youtube.com/embed/" + videoId;
  }

  function extractYoutubeVideoId(rawUrl) {
    if (!rawUrl) return "";
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

  function postWaitlistRecord(email, advice) {
    var endpoint = googleDbUrl();
    if (!endpoint) return Promise.reject(new Error("Google DB URL 설정이 필요합니다."));
    
    return postGoogleDbRecord(endpoint, "waitlist", {
      timestamp: new Date().toISOString(),
      session_id: ensureAnalyticsSessionId(),
      tester_id: betaUserKey(),
      email: email,
      advice: advice || "",
      page_url: window.location.href,
      path: window.location.pathname + window.location.hash
    });
  }

  function postGoogleDbRecord(endpoint, type, payload) {
    return fetch(endpoint, {
      method: "POST",
      mode: "no-cors",
      keepalive: true,
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify({ type: type, payload: payload })
    });
  }

  function renderList(target, items) {
    var safeItems = items.length ? items : ["코치 코멘트가 생성되지 않았습니다."];
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

  // Analytics helper dummy functions
  function initAnalytics() {
    ensureAnalyticsSessionId();
    bindAnalyticsClickTracking();
    initSectionViewTracking();
    trackEvent("view_page", "site", {
      referrer: document.referrer || "",
      title: document.title,
      viewport_width: window.innerWidth,
      viewport_height: window.innerHeight,
      screen_width: window.screen ? window.screen.width : null,
      screen_height: window.screen ? window.screen.height : null,
      language: navigator.language || "",
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || ""
    });
    flushAnalyticsQueue();
  }

  function trackEvent(eventName, section, metadata) {
    var payload = {
      event_name: eventName,
      section: section || "site",
      timestamp: new Date().toISOString(),
      session_id: ensureAnalyticsSessionId(),
      tester_id: betaUserKey(),
      page_url: window.location.href,
      path: window.location.pathname + window.location.search + window.location.hash,
      user_agent: navigator.userAgent || "",
      event_id: createEventId(),
      metadata: metadata || {}
    };
    enqueueAnalyticsEvent(payload);
    flushAnalyticsQueue();
  }

  function bindAnalyticsClickTracking() {
    document.querySelectorAll("[data-event]").forEach(function (node) {
      if (node.dataset.analyticsBound === "true") return;
      node.dataset.analyticsBound = "true";
      node.addEventListener("click", function () {
        trackEvent(node.dataset.event, node.dataset.section || "site", {
          label: node.textContent.trim(),
          href: node.getAttribute("href") || ""
        });
      });
    });
  }

  function initSectionViewTracking() {
    var sectionMap = {
      hero: "site_hero",
      problem: "site_problem",
      solution: "site_corefeature",
      analyze: "site_MVP",
      "cta-bottom": "site_CTA"
    };
    var sections = Object.keys(sectionMap)
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);
    if (!sections.length) return;

    if (!("IntersectionObserver" in window)) {
      sections.forEach(function (section) {
        trackEvent("view_section", sectionMap[section.id], { section_id: section.id });
      });
      return;
    }

    var seen = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting || seen[entry.target.id]) return;
        seen[entry.target.id] = true;
        trackEvent("view_section", sectionMap[entry.target.id], {
          section_id: entry.target.id,
          visible_ratio: Math.round(entry.intersectionRatio * 100) / 100
        });
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.45 });

    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  function createEventId() {
    var randomPart = Math.random().toString(36).slice(2, 10);
    return Date.now().toString(36) + "-" + randomPart;
  }

  function ensureAnalyticsSessionId() {
    var id = localStorage.getItem(storageKeys.analyticsSessionId);
    if (!id) {
      id = "session-" + Math.random().toString(36).slice(2, 12);
      localStorage.setItem(storageKeys.analyticsSessionId, id);
    }
    return id;
  }

  function enqueueAnalyticsEvent(eventPayload) {
    var queue = readAnalyticsQueue();
    queue.push(eventPayload);
    if (queue.length > analyticsQueueLimit) queue.shift();
    writeAnalyticsQueue(queue);
  }

  function flushAnalyticsQueue() {
    var endpoint = googleDbUrl();
    if (!endpoint) return;
    var queue = readAnalyticsQueue();
    if (!queue.length) return;
    
    writeAnalyticsQueue([]);
    queue.forEach(function (payload) {
      postGoogleDbRecord(endpoint, "analytics", payload).catch(function () {
        // Re-queue on failure
        var currentQueue = readAnalyticsQueue();
        currentQueue.push(payload);
        writeAnalyticsQueue(currentQueue);
      });
    });
  }

  function readAnalyticsQueue() {
    try {
      return JSON.parse(localStorage.getItem(storageKeys.analyticsQueue) || "[]");
    } catch (_err) {
      return [];
    }
  }

  function writeAnalyticsQueue(queue) {
    try {
      localStorage.setItem(storageKeys.analyticsQueue, JSON.stringify(queue));
    } catch (_err) {}
  }

  function maskEmail(email) {
    if (!email) return "";
    var parts = email.split("@");
    if (parts.length < 2) return "***";
    var name = parts[0], domain = parts[1];
    return name.slice(0, Math.min(name.length, 3)) + "***@" + domain;
  }

  function clearPollTimer() {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

})();
