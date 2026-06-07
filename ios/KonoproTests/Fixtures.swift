import Foundation
@testable import Konopro

enum FixtureError: Error, LocalizedError {
    case failed

    var errorDescription: String? {
        "Fixture failure"
    }
}

enum Fixtures {
    static let date = ISO8601DateFormatter().date(from: "2026-01-02T03:04:05Z")!

    static func session(
        id: String = "session-1",
        status: SessionStatus = .processed,
        processingJobId: String? = "job-1"
    ) -> AudioSessionResponse {
        AudioSessionResponse(
            id: id,
            userId: "beta-user",
            originalFilename: "karaoke.m4a",
            contentType: "audio/mp4",
            sha256: "abc123",
            sizeBytes: 1234,
            durationS: 42,
            clientRecordedAt: nil,
            source: "ios_app",
            status: status,
            processingJobId: processingJobId,
            createdAt: date,
            updatedAt: date
        )
    }

    static func job(status: JobStatus = .completed) -> ProcessingJobResponse {
        ProcessingJobResponse(
            id: "job-1",
            sessionId: "session-1",
            jobType: "fingerprint",
            status: status,
            attemptCount: 1,
            errorMessage: nil,
            createdAt: date,
            updatedAt: date,
            startedAt: date,
            finishedAt: status.isTerminal ? date : nil
        )
    }

    static func upload() -> UploadSessionResponse {
        UploadSessionResponse(session: session(status: .queued), job: job(status: .queued))
    }

    static func analysis() -> FingerprintAnalysisResponse {
        FingerprintAnalysisResponse(
            runId: "run-1",
            sessionId: "session-1",
            jobId: "job-1",
            provider: "acrcloud",
            status: "completed",
            providerStatus: "ok",
            recordingDurationS: 60,
            windowS: 20,
            hopS: 10,
            maxWindows: 6,
            useWhole: false,
            summary: ["recognized_windows": .number(3)],
            interpretations: ["overall": .string("experimental")],
            warnings: ["low confidence"],
            resultSummary: AnalysisResultSummary(
                status: "ok",
                message: "Detected one likely song interval.",
                confidenceLevel: "medium",
                canSegment: true,
                acceptedIntervalCount: 1,
                weakCandidateCount: 1,
                windowCount: 3,
                limitations: ["Fingerprinting is not singing scoring."]
            ),
            windows: [
                FingerprintWindowResponse(
                    windowIndex: 1,
                    provider: "acrcloud",
                    windowStartS: 0,
                    windowEndS: 20,
                    status: "matched",
                    recognized: true,
                    matchedTitle: "Demo Song",
                    matchedArtist: "Demo Artist",
                    identityKey: "demo-song",
                    isrc: "TEST123",
                    confidence: 88,
                    audioFile: "window.wav",
                    error: ""
                )
            ],
            intervals: [
                SongIntervalResponse(
                    intervalIndex: 1,
                    song: "Demo Song",
                    artist: "Demo Artist",
                    identityKey: "demo-song",
                    startS: 0,
                    endS: 40,
                    durationS: 40,
                    confidenceScore: 82,
                    confidenceLevel: "medium",
                    recognizedWindowCount: 2,
                    totalWindowCount: 3,
                    gapWindowCount: 1,
                    conflictWindowCount: 0,
                    providerConfidence: 88,
                    warnings: []
                )
            ],
            weakCandidates: [
                WeakCandidateResponse(
                    candidateIndex: 1,
                    source: "recovery",
                    song: "Demo Song",
                    artist: "Demo Artist",
                    identityKey: "demo-song",
                    startS: 40,
                    endS: 60,
                    durationS: 20,
                    recognizedWindowCount: 1,
                    totalWindowCount: 2,
                    providerConfidence: 65,
                    reason: "below threshold",
                    recoveryStartS: 40,
                    recoveryEndS: 60,
                    warnings: ["weak match"]
                )
            ],
            diagnostic: FingerprintDiagnosticResponse(
                provider: "acrcloud",
                canSegment: true,
                confidenceLevel: "medium",
                profile: ["match_rate": .number(0.66)],
                flags: [.string("mixed_confidence")],
                recommendations: [.string("try shorter hop")],
                recoverySweeps: [.object(["hop_s": .number(5)])]
            )
        )
    }

    static func report(
        id: String = "report-1",
        status: ReportRequestStatus = .delivered,
        priority: ReportPriority = .normal
    ) -> ReportRequestResponse {
        ReportRequestResponse(
            id: id,
            sessionId: "session-1",
            userId: "beta-user",
            status: status,
            priority: priority,
            requestType: "verified",
            targetTurnaroundHours: 24,
            dueAt: date,
            userNotes: "Please score the chorus.",
            adminNotes: nil,
            blockerReason: nil,
            createdAt: date,
            updatedAt: date,
            deliveredAt: status == .delivered ? date : nil,
            cancelledAt: nil,
            artifacts: [
                ReportArtifactResponse(
                    id: "artifact-1",
                    reportRequestId: id,
                    artifactType: "markdown",
                    title: "Verified Progress Report",
                    bodyText: "Pitch improved on the chorus.",
                    contentType: "text/markdown",
                    filename: nil,
                    visibility: "user",
                    createdAt: date,
                    updatedAt: date,
                    publishedAt: date
                )
            ]
        )
    }

    static func tempAudioFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("m4a")
        try Data("fake-audio".utf8).write(to: url)
        return url
    }
}

final class MockAPIClient: KonoproAPIProviding {
    var healthResult: Result<HealthResponse, Error> = .success(
        HealthResponse(status: "ok", environment: "test")
    )
    var uploadResult: Result<UploadSessionResponse, Error> = .success(Fixtures.upload())
    var sessionsResult: Result<[AudioSessionResponse], Error> = .success([Fixtures.session()])
    var jobResult: Result<ProcessingJobResponse, Error> = .success(Fixtures.job())
    var analysisResult: Result<FingerprintAnalysisResponse, Error> = .success(Fixtures.analysis())
    var reportResult: Result<ReportRequestResponse, Error> = .success(Fixtures.report())
    var reportsResult: Result<[ReportRequestResponse], Error> = .success([Fixtures.report()])

    private(set) var uploadedFileURL: URL?
    private(set) var uploadedSource: String?
    private(set) var reportRequest: (sessionId: String, requestType: String, userNotes: String?)?

    func health() async throws -> HealthResponse {
        try healthResult.get()
    }

    func uploadSession(
        fileURL: URL,
        source: String?,
        clientDurationS: Double?
    ) async throws -> UploadSessionResponse {
        uploadedFileURL = fileURL
        uploadedSource = source
        return try uploadResult.get()
    }

    func listSessions() async throws -> [AudioSessionResponse] {
        try sessionsResult.get()
    }

    func getJob(id: String) async throws -> ProcessingJobResponse {
        try jobResult.get()
    }

    func getAnalysis(sessionId: String) async throws -> FingerprintAnalysisResponse {
        try analysisResult.get()
    }

    func createReportRequest(
        sessionId: String,
        requestType: String,
        userNotes: String?
    ) async throws -> ReportRequestResponse {
        reportRequest = (sessionId, requestType, userNotes)
        return try reportResult.get()
    }

    func listReportRequests() async throws -> [ReportRequestResponse] {
        try reportsResult.get()
    }

    func getReportRequest(id: String) async throws -> ReportRequestResponse {
        try reportResult.get()
    }
}
