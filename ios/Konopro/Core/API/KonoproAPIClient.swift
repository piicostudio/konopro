import Foundation

protocol KonoproAPIProviding {
    func health() async throws -> HealthResponse
    func uploadSession(fileURL: URL, source: String?, clientDurationS: Double?) async throws -> UploadSessionResponse
    func listSessions() async throws -> [AudioSessionResponse]
    func getJob(id: String) async throws -> ProcessingJobResponse
    func getAnalysis(sessionId: String) async throws -> FingerprintAnalysisResponse
    func createReportRequest(sessionId: String, requestType: String, userNotes: String?) async throws -> ReportRequestResponse
    func listReportRequests() async throws -> [ReportRequestResponse]
    func getReportRequest(id: String) async throws -> ReportRequestResponse
}

enum KonoproAPIError: Error, LocalizedError, Equatable {
    case invalidBaseURL
    case invalidResponse
    case server(statusCode: Int, message: String)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Backend URL is invalid."
        case .invalidResponse:
            return "Backend returned an invalid response."
        case .server(_, let message):
            return message
        }
    }
}

final class KonoproAPIClient: KonoproAPIProviding {
    private let baseURL: URL
    private let betaUserKey: String
    private let session: URLSession

    @MainActor
    convenience init(settings: AppSettings, session: URLSession = .shared) {
        self.init(
            baseURL: settings.backendURL ?? URL(string: AppSettings.defaultBackendBaseURL)!,
            betaUserKey: settings.betaUserKey,
            session: session
        )
    }

    init(baseURL: URL, betaUserKey: String, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.betaUserKey = betaUserKey
        self.session = session
    }

    func health() async throws -> HealthResponse {
        try await send(path: "/health")
    }

    func uploadSession(
        fileURL: URL,
        source: String? = "ios_app",
        clientDurationS: Double? = nil
    ) async throws -> UploadSessionResponse {
        var multipart = MultipartFormData()
        if let clientDurationS {
            multipart.addField(name: "client_duration_s", value: String(clientDurationS))
        }
        if let source {
            multipart.addField(name: "source", value: source)
        }
        let data = try Data(contentsOf: fileURL)
        multipart.addFile(
            name: "file",
            filename: fileURL.lastPathComponent,
            contentType: contentType(for: fileURL),
            data: data
        )
        let body = multipart.finalize()
        var request = try makeRequest(path: "/v1/sessions", method: "POST")
        request.setValue(multipart.contentType, forHTTPHeaderField: "Content-Type")
        return try await send(request: request, body: body)
    }

    func listSessions() async throws -> [AudioSessionResponse] {
        try await send(path: "/v1/sessions")
    }

    func getJob(id: String) async throws -> ProcessingJobResponse {
        try await send(path: "/v1/jobs/\(id)")
    }

    func getAnalysis(sessionId: String) async throws -> FingerprintAnalysisResponse {
        try await send(path: "/v1/sessions/\(sessionId)/analysis")
    }

    func createReportRequest(
        sessionId: String,
        requestType: String,
        userNotes: String? = nil
    ) async throws -> ReportRequestResponse {
        let payload = ReportRequestCreate(requestType: requestType, userNotes: userNotes)
        var request = try makeRequest(
            path: "/v1/sessions/\(sessionId)/report-requests",
            method: "POST"
        )
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return try await send(request: request, body: try BackendJSON.encoder.encode(payload))
    }

    func listReportRequests() async throws -> [ReportRequestResponse] {
        try await send(path: "/v1/report-requests")
    }

    func getReportRequest(id: String) async throws -> ReportRequestResponse {
        try await send(path: "/v1/report-requests/\(id)")
    }

    private func send<T: Decodable>(path: String) async throws -> T {
        try await send(request: makeRequest(path: path), body: nil)
    }

    private func send<T: Decodable>(request: URLRequest, body: Data?) async throws -> T {
        let response: (Data, URLResponse)
        if let body {
            response = try await session.upload(for: request, from: body)
        } else {
            response = try await session.data(for: request)
        }
        guard let http = response.1 as? HTTPURLResponse else {
            throw KonoproAPIError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let error = try? BackendJSON.decoder.decode(ErrorDetail.self, from: response.0)
            throw KonoproAPIError.server(
                statusCode: http.statusCode,
                message: error?.detail ?? "Request failed with status \(http.statusCode)."
            )
        }
        return try BackendJSON.decoder.decode(T.self, from: response.0)
    }

    private func makeRequest(path: String, method: String = "GET") throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: baseURL)?.absoluteURL else {
            throw KonoproAPIError.invalidBaseURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if !betaUserKey.isEmpty {
            request.setValue(betaUserKey, forHTTPHeaderField: "X-Konopro-Beta-User")
        }
        return request
    }

    private func contentType(for fileURL: URL) -> String {
        switch fileURL.pathExtension.lowercased() {
        case "m4a", "mp4":
            return "audio/mp4"
        case "wav":
            return "audio/wav"
        case "mp3":
            return "audio/mpeg"
        default:
            return "application/octet-stream"
        }
    }
}

