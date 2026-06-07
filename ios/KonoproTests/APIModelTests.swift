import XCTest
@testable import Konopro

final class APIModelTests: XCTestCase {
    func testUploadResponseDecodesSnakeCaseAndDates() throws {
        let data = Data(
            """
            {
              "session": {
                "id": "session-1",
                "user_id": "beta-user",
                "original_filename": "karaoke.m4a",
                "content_type": "audio/mp4",
                "sha256": "abc123",
                "size_bytes": 1234,
                "duration_s": 42.5,
                "client_recorded_at": null,
                "source": "ios_app",
                "status": "queued",
                "processing_job_id": "job-1",
                "created_at": "2026-01-02T03:04:05Z",
                "updated_at": "2026-01-02T03:04:05.123Z"
              },
              "job": {
                "id": "job-1",
                "session_id": "session-1",
                "job_type": "fingerprint",
                "status": "queued",
                "attempt_count": 0,
                "error_message": null,
                "created_at": "2026-01-02T03:04:05Z",
                "updated_at": "2026-01-02T03:04:05Z",
                "started_at": null,
                "finished_at": null
              }
            }
            """.utf8
        )

        let response = try BackendJSON.decoder.decode(UploadSessionResponse.self, from: data)

        XCTAssertEqual(response.session.id, "session-1")
        XCTAssertEqual(response.session.status, .queued)
        XCTAssertEqual(response.session.processingJobId, "job-1")
        XCTAssertEqual(response.job.status, .queued)
        XCTAssertEqual(response.job.attemptCount, 0)
    }

    func testUploadResponseDecodesBackendNaiveDates() throws {
        let data = Data(
            """
            {
              "session": {
                "id": "session-1",
                "user_id": "beta-user",
                "original_filename": "karaoke.mp3",
                "content_type": "audio/mpeg",
                "sha256": "abc123",
                "size_bytes": 1234,
                "duration_s": null,
                "client_recorded_at": null,
                "source": "ios_app",
                "status": "queued",
                "processing_job_id": "job-1",
                "created_at": "2026-06-07T14:47:52.137375",
                "updated_at": "2026-06-07T14:47:52.142312"
              },
              "job": {
                "id": "job-1",
                "session_id": "session-1",
                "job_type": "fingerprint_segmentation",
                "status": "queued",
                "attempt_count": 0,
                "error_message": null,
                "created_at": "2026-06-07T14:47:52.140772",
                "updated_at": "2026-06-07T14:47:52.140772",
                "started_at": null,
                "finished_at": null
              }
            }
            """.utf8
        )

        let response = try BackendJSON.decoder.decode(UploadSessionResponse.self, from: data)

        XCTAssertEqual(response.session.id, "session-1")
        XCTAssertEqual(response.session.originalFilename, "karaoke.mp3")
        XCTAssertEqual(response.job.jobType, "fingerprint_segmentation")
    }

    func testAnalysisResponseDecodesNestedExperimentalData() throws {
        let data = Data(
            """
            {
              "run_id": "run-1",
              "session_id": "session-1",
              "job_id": "job-1",
              "provider": "acrcloud",
              "status": "completed",
              "provider_status": "ok",
              "recording_duration_s": 60,
              "window_s": 20,
              "hop_s": 10,
              "max_windows": 6,
              "use_whole": false,
              "summary": {"recognized_windows": 3},
              "interpretations": {"overall": "experimental"},
              "warnings": ["low confidence"],
              "result_summary": {
                "status": "ok",
                "message": "Detected one likely song interval.",
                "confidence_level": "medium",
                "can_segment": true,
                "accepted_interval_count": 1,
                "weak_candidate_count": 0,
                "window_count": 3,
                "limitations": ["Fingerprinting is not singing scoring."]
              },
              "windows": [],
              "intervals": [],
              "weak_candidates": [],
              "diagnostic": {
                "provider": "acrcloud",
                "can_segment": true,
                "confidence_level": "medium",
                "profile": {"match_rate": 0.66},
                "flags": ["mixed_confidence"],
                "recommendations": ["try shorter hop"],
                "recovery_sweeps": []
              }
            }
            """.utf8
        )

        let response = try BackendJSON.decoder.decode(FingerprintAnalysisResponse.self, from: data)

        XCTAssertEqual(response.runId, "run-1")
        XCTAssertTrue(response.resultSummary.canSegment)
        XCTAssertEqual(response.summary["recognized_windows"], .number(3))
        XCTAssertEqual(response.diagnostic?.flags, [.string("mixed_confidence")])
    }

    func testReportRequestDecodesDeliveredArtifacts() throws {
        let data = Data(
            """
            {
              "id": "report-1",
              "session_id": "session-1",
              "user_id": "beta-user",
              "status": "delivered",
              "priority": "normal",
              "request_type": "verified",
              "target_turnaround_hours": 24,
              "due_at": "2026-01-03T03:04:05Z",
              "user_notes": "Please score the chorus.",
              "admin_notes": null,
              "blocker_reason": null,
              "created_at": "2026-01-02T03:04:05Z",
              "updated_at": "2026-01-02T03:04:05Z",
              "delivered_at": "2026-01-02T04:04:05Z",
              "cancelled_at": null,
              "artifacts": [
                {
                  "id": "artifact-1",
                  "report_request_id": "report-1",
                  "artifact_type": "markdown",
                  "title": "Verified Progress Report",
                  "body_text": "Pitch improved on the chorus.",
                  "content_type": "text/markdown",
                  "filename": null,
                  "visibility": "user",
                  "created_at": "2026-01-02T03:04:05Z",
                  "updated_at": "2026-01-02T03:04:05Z",
                  "published_at": "2026-01-02T04:04:05Z"
                }
              ]
            }
            """.utf8
        )

        let report = try BackendJSON.decoder.decode(ReportRequestResponse.self, from: data)

        XCTAssertEqual(report.status, .delivered)
        XCTAssertEqual(report.priority, .normal)
        XCTAssertEqual(report.artifacts.first?.title, "Verified Progress Report")
    }
}
