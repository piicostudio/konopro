import XCTest
@testable import Konopro

@MainActor
final class ReportViewModelTests: XCTestCase {
    func testRequestReportStoresDetailAndArguments() async {
        let mock = MockAPIClient()
        let viewModel = ReportViewModel(api: mock)

        await viewModel.requestReport(
            sessionId: "session-1",
            requestType: "priority",
            userNotes: "Need chorus feedback."
        )

        XCTAssertEqual(mock.reportRequest?.sessionId, "session-1")
        XCTAssertEqual(mock.reportRequest?.requestType, "priority")
        XCTAssertEqual(mock.reportRequest?.userNotes, "Need chorus feedback.")
        XCTAssertEqual(viewModel.state, .detail(Fixtures.report()))
    }

    func testLoadReportsPublishesList() async {
        let mock = MockAPIClient()
        let viewModel = ReportViewModel(api: mock)

        await viewModel.loadReports()

        XCTAssertEqual(viewModel.state, .loaded([Fixtures.report()]))
    }

    func testLoadReportFailurePublishesFailedState() async {
        let mock = MockAPIClient()
        mock.reportResult = .failure(FixtureError.failed)
        let viewModel = ReportViewModel(api: mock)

        await viewModel.loadReport(id: "report-1")

        XCTAssertEqual(viewModel.state, .failed("Fixture failure"))
    }
}
