import Foundation
import XCTest
@testable import Konopro

final class KonoproAPIClientTests: XCTestCase {
    override func tearDown() {
        URLProtocolStub.handler = nil
        super.tearDown()
    }

    func testHealthAddsBetaHeaderAndDecodesResponse() async throws {
        let client = makeClient { request in
            XCTAssertEqual(request.url?.path, "/health")
            XCTAssertEqual(request.httpMethod, "GET")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-Konopro-Beta-User"), "beta-user")
            return Self.response(
                for: request,
                json: #"{"status":"ok","environment":"test"}"#
            )
        }

        let health = try await client.health()

        XCTAssertEqual(health.status, "ok")
        XCTAssertEqual(health.environment, "test")
    }

    func testServerErrorDecodesErrorDetail() async {
        let client = makeClient { request in
            Self.response(
                for: request,
                statusCode: 422,
                json: #"{"detail":"Unsupported file type"}"#
            )
        }

        do {
            _ = try await client.listSessions()
            XCTFail("Expected server error.")
        } catch let error as KonoproAPIError {
            XCTAssertEqual(error, .server(statusCode: 422, message: "Unsupported file type"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testUploadSessionPostsMultipartToSessionsEndpoint() async throws {
        let fileURL = try Fixtures.tempAudioFile()
        let client = makeClient { request in
            XCTAssertEqual(request.url?.path, "/v1/sessions")
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-Konopro-Beta-User"), "beta-user")
            XCTAssertTrue(
                request.value(forHTTPHeaderField: "Content-Type")?
                    .hasPrefix("multipart/form-data; boundary=") == true
            )
            return Self.response(for: request, json: Self.uploadJSON)
        }

        let response = try await client.uploadSession(
            fileURL: fileURL,
            source: "ios_test",
            clientDurationS: 3
        )

        XCTAssertEqual(response.session.id, "session-1")
        XCTAssertEqual(response.job.id, "job-1")
    }

    private func makeClient(
        handler: @escaping (URLRequest) throws -> (HTTPURLResponse, Data)
    ) -> KonoproAPIClient {
        URLProtocolStub.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [URLProtocolStub.self]
        let session = URLSession(configuration: configuration)
        return KonoproAPIClient(
            baseURL: URL(string: "http://konopro.test")!,
            betaUserKey: "beta-user",
            session: session
        )
    }

    private static func response(
        for request: URLRequest,
        statusCode: Int = 200,
        json: String
    ) -> (HTTPURLResponse, Data) {
        (
            HTTPURLResponse(
                url: request.url!,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!,
            Data(json.utf8)
        )
    }

    private static let uploadJSON = """
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
        "updated_at": "2026-01-02T03:04:05Z"
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
    """
}

final class URLProtocolStub: URLProtocol {
    static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: FixtureError.failed)
            return
        }

        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
