const SHEETS = {
  events: "events_raw",
  waitlist: "waitlist",
  eventSummary: "event_summary",
  sessionSummary: "session_summary",
  readme: "readme"
};

function setupKonoproDb() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  ensureSheet_(spreadsheet, SHEETS.events, [
    "timestamp",
    "received_at",
    "session_id",
    "tester_id",
    "event_name",
    "section",
    "page_url",
    "path",
    "user_agent",
    "metadata_json",
    "event_id"
  ]);
  ensureSheet_(spreadsheet, SHEETS.waitlist, [
    "submitted_at",
    "received_at",
    "session_id",
    "tester_id",
    "email",
    "email_masked",
    "email_hash",
    "advice",
    "page_url",
    "path"
  ]);
  ensureSummarySheets_(spreadsheet);
  return { ok: true, spreadsheet_url: spreadsheet.getUrl() };
}

function doGet() {
  const result = setupKonoproDb();
  return json_(Object.assign({ status: "ready" }, result));
}

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(5000);
  try {
    setupKonoproDb();
    const body = parseBody_(e);
    if (body.type === "waitlist") {
      appendWaitlist_(body.payload || {});
      return json_({ ok: true, type: "waitlist" });
    }
    appendEvent_(body.payload || {});
    return json_({ ok: true, type: "event" });
  } catch (error) {
    return json_({ ok: false, error: String(error && error.message || error) });
  } finally {
    lock.releaseLock();
  }
}

function appendEvent_(payload) {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEETS.events);
  sheet.appendRow([
    payload.timestamp || new Date().toISOString(),
    new Date().toISOString(),
    payload.session_id || "",
    payload.tester_id || "",
    payload.event_name || "",
    payload.section || "",
    payload.page_url || "",
    payload.path || "",
    payload.user_agent || "",
    JSON.stringify(payload.metadata || {}),
    payload.event_id || ""
  ]);
}

function appendWaitlist_(payload) {
  const email = String(payload.email || "").trim();
  if (!email) {
    throw new Error("Email is required.");
  }

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEETS.waitlist);
  sheet.appendRow([
    payload.timestamp || new Date().toISOString(),
    new Date().toISOString(),
    payload.session_id || "",
    payload.tester_id || "",
    email,
    maskEmail_(email),
    sha256_(email.toLowerCase()),
    payload.advice || "",
    payload.page_url || "",
    payload.path || ""
  ]);
}

function ensureSheet_(spreadsheet, name, headers) {
  const sheet = spreadsheet.getSheetByName(name) || spreadsheet.insertSheet(name);
  const currentHeaders = sheet.getRange(1, 1, 1, headers.length).getValues()[0];
  const missingHeaders = headers.some((header, index) => currentHeaders[index] !== header);
  if (missingHeaders) {
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
    sheet.setFrozenRows(1);
    sheet.autoResizeColumns(1, headers.length);
  }
  return sheet;
}

function ensureSummarySheets_(spreadsheet) {
  const eventSummary = spreadsheet.getSheetByName(SHEETS.eventSummary) || spreadsheet.insertSheet(SHEETS.eventSummary);
  const eventSummaryFormula = "=QUERY(events_raw!E2:F,\"select E, F, count(E) where E is not null group by E, F label E 'event_name', F 'section', count(E) 'count'\",0)";
  if (eventSummary.getRange("A1").getFormula() !== eventSummaryFormula) {
    eventSummary.clear();
    eventSummary.getRange("A1").setFormula(eventSummaryFormula);
  }
  eventSummary.setFrozenRows(1);

  const sessionSummary = spreadsheet.getSheetByName(SHEETS.sessionSummary) || spreadsheet.insertSheet(SHEETS.sessionSummary);
  const sessionSummaryFormula = "=QUERY(events_raw!A2:E,\"select C, count(E), min(A), max(A) where C is not null group by C label C 'session_id', count(E) 'event_count', min(A) 'first_seen', max(A) 'last_seen'\",0)";
  if (sessionSummary.getRange("A1").getFormula() !== sessionSummaryFormula) {
    sessionSummary.clear();
    sessionSummary.getRange("A1").setFormula(sessionSummaryFormula);
  }
  sessionSummary.setFrozenRows(1);

  const readme = spreadsheet.getSheetByName(SHEETS.readme) || spreadsheet.insertSheet(SHEETS.readme);
  if (readme.getRange("A1").getValue() !== "Tab") {
    readme.clear();
    readme.getRange(1, 1, 12, 2).setValues([
      ["Tab", "Purpose"],
      ["events_raw", "Append-only log of section views, clicks, hover interest, demo workflow, survey answers, and CTA actions."],
      ["waitlist", "Email signups from the release notification modal. Do not share publicly if raw emails should remain private."],
      ["event_summary", "Auto-generated count by event_name and section."],
      ["session_summary", "Auto-generated count and first/last seen timestamps by browser session."],
      ["Key event", "site_hero, site_problem, site_corefeature, site_MVP, site_CTA"],
      ["Feature interest", "hover_corefeature_1/2/3 on desktop; click_corefeature_1/2/3 on touch/click."],
      ["Upload workflow", "click_uploadcover_mp3, action_uploadcover_mp3, click_next_after_uploadcover."],
      ["Reference workflow", "action_paste_youtubelink, click_previeworiginal, click_yesanalyze, click_changeURL."],
      ["Survey workflow", "click_surveyquestion_karaokeUse/deviceContext/appInstall/resultPriority."],
      ["Analysis workflow", "action_analysis_started, action_analysis_completed, action_analysis_failed, action_result_unlocked."],
      ["CTA workflow", "click_cta_button, click_submit_email, action_submit_email_success."]
    ]);
  }
  readme.setFrozenRows(1);
  readme.autoResizeColumns(1, 2);
}

function parseBody_(e) {
  if (!e || !e.postData || !e.postData.contents) {
    throw new Error("Missing request body.");
  }
  return JSON.parse(e.postData.contents);
}

function maskEmail_(email) {
  const parts = String(email || "").split("@");
  if (parts.length !== 2) {
    return "";
  }
  return parts[0].slice(0, 2) + "***@" + parts[1];
}

function sha256_(value) {
  const bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, value);
  return bytes.map(function(byte) {
    const unsigned = byte < 0 ? byte + 256 : byte;
    return ("0" + unsigned.toString(16)).slice(-2);
  }).join("");
}

function json_(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
