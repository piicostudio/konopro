import Foundation

enum BackendJSON {
    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            if let date = iso8601WithFractional.date(from: value)
                ?? iso8601.date(from: value)
                ?? backendNaiveWithFractional.date(from: value)
                ?? backendNaive.date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid ISO-8601 date: \(value)"
            )
        }
        return decoder
    }()

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let backendNaiveWithFractional: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        return formatter
    }()

    private static let backendNaive: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter
    }()
}

struct ErrorDetail: Codable, Equatable {
    let detail: String
}

struct HealthResponse: Codable, Equatable {
    let status: String
    let environment: String
}

enum SessionStatus: String, Codable, Equatable {
    case uploaded
    case queued
    case processing
    case processed
    case failed
    case deleted
}

enum JobStatus: String, Codable, Equatable {
    case queued
    case processing
    case completed
    case failed
    case cancelled

    var isTerminal: Bool {
        self == .completed || self == .failed || self == .cancelled
    }
}

enum ReportRequestStatus: String, Codable, Equatable {
    case requested
    case triaged
    case inProgress = "in_progress"
    case blocked
    case ready
    case delivered
    case cancelled
    case unableToComplete = "unable_to_complete"
}

enum ReportPriority: String, Codable, Equatable {
    case low
    case normal
    case high
}

struct AudioSessionResponse: Codable, Identifiable, Equatable {
    let id: String
    let userId: String
    let originalFilename: String
    let contentType: String
    let sha256: String
    let sizeBytes: Int
    let durationS: Double?
    let clientRecordedAt: Date?
    let source: String?
    let status: SessionStatus
    let processingJobId: String?
    let createdAt: Date
    let updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case id
        case userId = "user_id"
        case originalFilename = "original_filename"
        case contentType = "content_type"
        case sha256
        case sizeBytes = "size_bytes"
        case durationS = "duration_s"
        case clientRecordedAt = "client_recorded_at"
        case source
        case status
        case processingJobId = "processing_job_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct ProcessingJobResponse: Codable, Identifiable, Equatable {
    let id: String
    let sessionId: String
    let jobType: String
    let status: JobStatus
    let attemptCount: Int
    let errorMessage: String?
    let createdAt: Date
    let updatedAt: Date
    let startedAt: Date?
    let finishedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id
        case sessionId = "session_id"
        case jobType = "job_type"
        case status
        case attemptCount = "attempt_count"
        case errorMessage = "error_message"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case startedAt = "started_at"
        case finishedAt = "finished_at"
    }
}

struct UploadSessionResponse: Codable, Equatable {
    let session: AudioSessionResponse
    let job: ProcessingJobResponse
}

struct FingerprintAnalysisResponse: Codable, Equatable {
    let runId: String
    let sessionId: String
    let jobId: String?
    let provider: String
    let status: String
    let providerStatus: String?
    let recordingDurationS: Double?
    let windowS: Double?
    let hopS: Double?
    let maxWindows: Int?
    let useWhole: Bool
    let summary: [String: JSONValue]
    let interpretations: [String: JSONValue]
    let warnings: [String]
    let resultSummary: AnalysisResultSummary
    let windows: [FingerprintWindowResponse]
    let intervals: [SongIntervalResponse]
    let weakCandidates: [WeakCandidateResponse]
    let diagnostic: FingerprintDiagnosticResponse?

    enum CodingKeys: String, CodingKey {
        case runId = "run_id"
        case sessionId = "session_id"
        case jobId = "job_id"
        case provider
        case status
        case providerStatus = "provider_status"
        case recordingDurationS = "recording_duration_s"
        case windowS = "window_s"
        case hopS = "hop_s"
        case maxWindows = "max_windows"
        case useWhole = "use_whole"
        case summary
        case interpretations
        case warnings
        case resultSummary = "result_summary"
        case windows
        case intervals
        case weakCandidates = "weak_candidates"
        case diagnostic
    }
}

struct AnalysisResultSummary: Codable, Equatable {
    let status: String
    let message: String
    let confidenceLevel: String
    let canSegment: Bool
    let acceptedIntervalCount: Int
    let weakCandidateCount: Int
    let windowCount: Int
    let limitations: [String]

    enum CodingKeys: String, CodingKey {
        case status
        case message
        case confidenceLevel = "confidence_level"
        case canSegment = "can_segment"
        case acceptedIntervalCount = "accepted_interval_count"
        case weakCandidateCount = "weak_candidate_count"
        case windowCount = "window_count"
        case limitations
    }
}

struct FingerprintWindowResponse: Codable, Identifiable, Equatable {
    var id: Int { windowIndex }
    let windowIndex: Int
    let provider: String
    let windowStartS: Double
    let windowEndS: Double
    let status: String
    let recognized: Bool
    let matchedTitle: String
    let matchedArtist: String
    let identityKey: String
    let isrc: String
    let confidence: Double?
    let audioFile: String
    let error: String

    enum CodingKeys: String, CodingKey {
        case windowIndex = "window_index"
        case provider
        case windowStartS = "window_start_s"
        case windowEndS = "window_end_s"
        case status
        case recognized
        case matchedTitle = "matched_title"
        case matchedArtist = "matched_artist"
        case identityKey = "identity_key"
        case isrc
        case confidence
        case audioFile = "audio_file"
        case error
    }
}

struct SongIntervalResponse: Codable, Identifiable, Equatable {
    var id: Int { intervalIndex }
    let intervalIndex: Int
    let song: String
    let artist: String
    let identityKey: String
    let startS: Double
    let endS: Double
    let durationS: Double
    let confidenceScore: Double
    let confidenceLevel: String
    let recognizedWindowCount: Int
    let totalWindowCount: Int
    let gapWindowCount: Int
    let conflictWindowCount: Int
    let providerConfidence: Double
    let warnings: [String]

    enum CodingKeys: String, CodingKey {
        case intervalIndex = "interval_index"
        case song
        case artist
        case identityKey = "identity_key"
        case startS = "start_s"
        case endS = "end_s"
        case durationS = "duration_s"
        case confidenceScore = "confidence_score"
        case confidenceLevel = "confidence_level"
        case recognizedWindowCount = "recognized_window_count"
        case totalWindowCount = "total_window_count"
        case gapWindowCount = "gap_window_count"
        case conflictWindowCount = "conflict_window_count"
        case providerConfidence = "provider_confidence"
        case warnings
    }
}

struct WeakCandidateResponse: Codable, Identifiable, Equatable {
    var id: Int { candidateIndex }
    let candidateIndex: Int
    let source: String
    let song: String
    let artist: String
    let identityKey: String
    let startS: Double
    let endS: Double
    let durationS: Double
    let recognizedWindowCount: Int
    let totalWindowCount: Int
    let providerConfidence: Double?
    let reason: String
    let recoveryStartS: Double?
    let recoveryEndS: Double?
    let warnings: [String]

    enum CodingKeys: String, CodingKey {
        case candidateIndex = "candidate_index"
        case source
        case song
        case artist
        case identityKey = "identity_key"
        case startS = "start_s"
        case endS = "end_s"
        case durationS = "duration_s"
        case recognizedWindowCount = "recognized_window_count"
        case totalWindowCount = "total_window_count"
        case providerConfidence = "provider_confidence"
        case reason
        case recoveryStartS = "recovery_start_s"
        case recoveryEndS = "recovery_end_s"
        case warnings
    }
}

struct FingerprintDiagnosticResponse: Codable, Equatable {
    let provider: String
    let canSegment: Bool
    let confidenceLevel: String
    let profile: [String: JSONValue]
    let flags: [JSONValue]
    let recommendations: [JSONValue]
    let recoverySweeps: [JSONValue]

    enum CodingKeys: String, CodingKey {
        case provider
        case canSegment = "can_segment"
        case confidenceLevel = "confidence_level"
        case profile
        case flags
        case recommendations
        case recoverySweeps = "recovery_sweeps"
    }
}

struct ReportRequestCreate: Codable, Equatable {
    let requestType: String
    let userNotes: String?

    enum CodingKeys: String, CodingKey {
        case requestType = "request_type"
        case userNotes = "user_notes"
    }
}

struct ReportArtifactResponse: Codable, Identifiable, Equatable {
    let id: String
    let reportRequestId: String
    let artifactType: String
    let title: String
    let bodyText: String?
    let contentType: String?
    let filename: String?
    let visibility: String
    let createdAt: Date
    let updatedAt: Date
    let publishedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id
        case reportRequestId = "report_request_id"
        case artifactType = "artifact_type"
        case title
        case bodyText = "body_text"
        case contentType = "content_type"
        case filename
        case visibility
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case publishedAt = "published_at"
    }
}

struct ReportRequestResponse: Codable, Identifiable, Equatable {
    let id: String
    let sessionId: String
    let userId: String
    let status: ReportRequestStatus
    let priority: ReportPriority
    let requestType: String
    let targetTurnaroundHours: Int
    let dueAt: Date?
    let userNotes: String?
    let adminNotes: String?
    let blockerReason: String?
    let createdAt: Date
    let updatedAt: Date
    let deliveredAt: Date?
    let cancelledAt: Date?
    let artifacts: [ReportArtifactResponse]

    enum CodingKeys: String, CodingKey {
        case id
        case sessionId = "session_id"
        case userId = "user_id"
        case status
        case priority
        case requestType = "request_type"
        case targetTurnaroundHours = "target_turnaround_hours"
        case dueAt = "due_at"
        case userNotes = "user_notes"
        case adminNotes = "admin_notes"
        case blockerReason = "blocker_reason"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case deliveredAt = "delivered_at"
        case cancelledAt = "cancelled_at"
        case artifacts
    }
}

enum JSONValue: Codable, Equatable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}
