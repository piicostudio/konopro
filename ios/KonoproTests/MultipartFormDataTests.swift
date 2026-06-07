import XCTest
@testable import Konopro

final class MultipartFormDataTests: XCTestCase {
    func testMultipartBodyContainsFieldsFileAndClosingBoundary() {
        var form = MultipartFormData(boundary: "test-boundary")
        form.addField(name: "source", value: "ios_app")
        form.addFile(
            name: "file",
            filename: "take.m4a",
            contentType: "audio/mp4",
            data: Data("audio-bytes".utf8)
        )

        let body = String(data: form.finalize(), encoding: .utf8)

        XCTAssertEqual(form.contentType, "multipart/form-data; boundary=test-boundary")
        XCTAssertTrue(body?.contains("name=\"source\"") == true)
        XCTAssertTrue(body?.contains("ios_app") == true)
        XCTAssertTrue(body?.contains("filename=\"take.m4a\"") == true)
        XCTAssertTrue(body?.contains("Content-Type: audio/mp4") == true)
        XCTAssertTrue(body?.contains("audio-bytes") == true)
        XCTAssertTrue(body?.contains("--test-boundary--") == true)
    }
}
