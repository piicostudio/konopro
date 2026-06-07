import XCTest
@testable import Konopro

@MainActor
final class SessionViewModelTests: XCTestCase {
    func testSessionListLoadsSessions() async {
        let mock = MockAPIClient()
        let viewModel = SessionListViewModel(api: mock)

        await viewModel.load()

        XCTAssertEqual(viewModel.sessions(), [Fixtures.session()])
        XCTAssertEqual(viewModel.state, .loaded([Fixtures.session()]))
    }

    func testSessionDetailPollLoadsCompletedJobAndAnalysis() async {
        let mock = MockAPIClient()
        mock.jobResult = .success(Fixtures.job(status: .completed))
        mock.analysisResult = .success(Fixtures.analysis())
        let viewModel = SessionDetailViewModel(session: Fixtures.session(), api: mock)

        await viewModel.pollJob(maxAttempts: 1)

        XCTAssertEqual(viewModel.job, Fixtures.job(status: .completed))
        XCTAssertEqual(viewModel.analysis, Fixtures.analysis())
        XCTAssertFalse(viewModel.isPolling)
    }

    func testSessionDetailCapturesJobError() async {
        let mock = MockAPIClient()
        mock.jobResult = .failure(FixtureError.failed)
        let viewModel = SessionDetailViewModel(session: Fixtures.session(), api: mock)

        await viewModel.pollJob(maxAttempts: 1)

        XCTAssertEqual(viewModel.errorMessage, "Fixture failure")
        XCTAssertFalse(viewModel.isPolling)
    }
}
