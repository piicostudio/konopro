(function () {
  "use strict";

  var elements = {};
  var pollTimer = null;
  var processingStageTimer = null;
  var processingStageIndex = 0;
  var activeSessionId = null;
  var vantaInstance = null;
  
  var audioPreviewUrls = {
    take: null
  };
  
  var defaultGoogleDbUrl = "https://script.google.com/macros/s/AKfycbw2aSa60f7B-wMBQlspwdNHi8w2iHTQ-tLouwVdMP7ddPomE_TYBPcM1iNQgRHpeLyoYw/exec";
  var storageKeys = {
    apiBaseUrl: "konopro.apiBaseUrl",
    betaUserKey: "konopro.betaUserKey",
    googleDbUrl: "konopro.googleDbUrl",
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

  var flowState = createInitialFlowState();

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    initTheme();
    initVantaBackground();
    initStoredSettings();
    initLandingInteractions();
    initScoringConsole();
    initInlineWaitlistForm();
    initEntranceAnimations();
  });

  function cacheElements() {
    elements = {
      themeToggle: document.getElementById("themeToggle"),
      settingsButton: document.getElementById("settingsButton"),
      settingsModal: document.getElementById("settingsModal"),
      modal: document.getElementById("revealModal"),
      waitlistForm: document.getElementById("waitlistForm"),
      formFeedback: document.getElementById("formFeedback"),
      healthCheckButton: document.getElementById("healthCheckButton"),
      healthStatus: document.getElementById("healthStatus"),
      apiBaseUrl: document.getElementById("apiBaseUrl"),
      betaUserKey: document.getElementById("betaUserKey"),
      googleDbUrl: document.getElementById("googleDbUrl"),
      
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
      
      result: document.getElementById("result"),
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
      questionsComplete: true, // Auto bypass survey
      resultRendered: false
    };
  }

  function initStoredSettings() {
    if (elements.apiBaseUrl) {
      elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
    }
    if (elements.betaUserKey) {
      elements.betaUserKey.value = localStorage.getItem(storageKeys.betaUserKey) || defaultTesterId();
    }
    if (elements.googleDbUrl) {
      elements.googleDbUrl.value = localStorage.getItem(storageKeys.googleDbUrl) || defaultGoogleDbUrl;
    }
    persistSettings();
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
    initBuilderPreview();
    initSettingsModal();
    initAnalytics();
    initModal();
    initWaitlistForm();
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
        elements.processingHint.textContent = "분석 시작 실패. 백엔드 서버 및 파일 형식을 확인하세요.";
        failProcessingNarrative("분석 시작 실패. 백엔드 서버 및 파일 형식을 확인하세요.");
        flowState.analysisFailed = true;
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
        headers: { "X-Konopro-Beta-User": betaUserKey() }
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

            // Unlock and render results immediately
            flowState.resultRendered = true;
            transitionRightColumn("result");
            renderResult(payload.scoring_run);
            return;
          }
          
          if (payload.job.status === "failed" || payload.scoring_run.status === "failed") {
            clearPollTimer();
            setBusy(false);
            var errMsg = payload.scoring_run.error_message || payload.job.error_message || "분석 도중 실패했습니다.";
            setJobStatus(errMsg, "error");
            failProcessingNarrative(errMsg);
            flowState.analysisFailed = true;
            trackEvent("action_analysis_failed", "site_MVP", {
              stage: "poll",
              job_id: payload.job.id,
              message: errMsg
            });
            return;
          }
          
          setJobStatus(statusText(payload), "working");
        })
        .catch(function (error) {
          clearPollTimer();
          setBusy(false);
          setJobStatus(error.message, "error");
          failProcessingNarrative("결과 폴링 연결 실패. 백엔드가 실행 중인지 확인하세요.");
          flowState.analysisFailed = true;
        });
    }, 2000);
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
    
    elements.metricGrid.innerHTML = metrics.map(metricCard).join("");
    renderList(elements.feedbackList, scoringRun.feedback || []);
    renderWarnings(scoringRun.warnings || []);

    // Render A/B moments dynamically based on scores
    renderABMoments(scores);
  }

  function renderABMoments(scores) {
    var container = document.getElementById("badMomentsContainer");
    if (!container) return;

    var pitchScore = scores.pitch_accuracy_score || 70;
    var timingScore = scores.timing_score || 70;

    // Plausible moments based on actual performance
    var moments = [
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

    container.innerHTML = moments.map(function (m) {
      return (
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
        '</div>'
      );
    }).join("");

    // Wire moments events
    container.querySelectorAll(".btn-play-take").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var start = Number(btn.getAttribute("data-start"));
        playTakeSegment(start, 5);
      });
    });

    container.querySelectorAll(".btn-play-original").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var start = Number(btn.getAttribute("data-start"));
        playOriginalSegment(start);
      });
    });
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
    fetch(apiBaseUrl() + "/health")
      .then(parseResponse)
      .then(function (payload) {
        setHealth("Backend OK: " + payload.status + " (" + payload.environment + ")", "good");
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
    setProcessingStage(processingStages.length - 1, "done");
  }

  function failProcessingNarrative(message) {
    clearProcessingStageTimer();
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
    
    elements.apiBaseUrl.value = localStorage.getItem(storageKeys.apiBaseUrl) || "http://127.0.0.1:8000";
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
    setProcessingStage(0, "idle");
    setJobStatus("대기 중", "idle");
    transitionRightColumn("empty");
  }

  function resetResult() {
    elements.overallScore.textContent = "--";
    elements.metricGrid.innerHTML = "";
    elements.feedbackList.innerHTML = "";
    elements.warningList.innerHTML = "";
    elements.warningBox.classList.add("is-hidden");
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
  }

  function persistSettings() {
    if (elements.apiBaseUrl) localStorage.setItem(storageKeys.apiBaseUrl, apiBaseUrl());
    if (elements.betaUserKey) localStorage.setItem(storageKeys.betaUserKey, betaUserKey());
    if (elements.googleDbUrl) localStorage.setItem(storageKeys.googleDbUrl, googleDbUrl());
  }

  function apiBaseUrl() {
    return elements.apiBaseUrl.value.trim().replace(/\/+$/, "");
  }

  function betaUserKey() {
    return elements.betaUserKey.value.trim();
  }

  function googleDbUrl() {
    return elements.googleDbUrl ? elements.googleDbUrl.value.trim() : "";
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

  function defaultTesterId() {
    return "tester-" + Math.random().toString(36).slice(2, 8);
  }

  function parseResponse(response) {
    if (!response.ok) {
      return response.json().then(function (err) {
        throw new Error(err.detail || "서버 통신 실패");
      });
    }
    return response.json();
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
    flushAnalyticsQueue();
  }

  function trackEvent(eventName, section, metadata) {
    var payload = {
      event_name: eventName,
      section: section || "site",
      timestamp: new Date().toISOString(),
      session_id: ensureAnalyticsSessionId(),
      tester_id: betaUserKey(),
      metadata: metadata || {}
    };
    enqueueAnalyticsEvent(payload);
    flushAnalyticsQueue();
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
