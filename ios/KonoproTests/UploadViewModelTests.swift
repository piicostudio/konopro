import XCTest
@testable import Konopro

@MainActor
final class UploadViewModelTests: XCTestCase {
    func testUploadPublishesUploadedStateAndPassesSource() async throws {
        let mock = MockAPIClient()
        let viewModel = UploadViewModel(api: mock)
        let fileURL = try Fixtures.tempAudioFile()

        await viewModel.upload(fileURL: fileURL, source: "ios_test")

        XCTAssertEqual(mock.uploadedFileURL, fileURL)
        XCTAssertEqual(mock.uploadedSource, "ios_test")
        XCTAssertEqual(viewModel.state, .uploaded(Fixtures.upload()))
    }

    func testUploadFailurePublishesFailedState() async throws {
        let mock = MockAPIClient()
        mock.uploadResult = .failure(FixtureError.failed)
        let viewModel = UploadViewModel(api: mock)

        await viewModel.upload(fileURL: try Fixtures.tempAudioFile())

        XCTAssertEqual(viewModel.state, .failed("Fixture failure"))
    }
}
