import XCTest
@testable import Konopro

@MainActor
final class SetupViewModelTests: XCTestCase {
    func testConnectionAndSaveUseNormalizedSettings() async {
        let settings = AppSettings(defaults: makeDefaults())
        let mock = MockAPIClient()
        let viewModel = SetupViewModel(settings: settings) { _, _ in mock }
        viewModel.betaUserKey = " beta-user "
        viewModel.backendBaseURL = " http://localhost:8000 "

        await viewModel.testConnection()
        viewModel.save()

        XCTAssertEqual(viewModel.state, .valid("Settings saved."))
        XCTAssertEqual(settings.betaUserKey, "beta-user")
        XCTAssertEqual(settings.backendBaseURL, "http://localhost:8000")
    }

    func testConnectionFailureShowsError() async {
        let settings = AppSettings(defaults: makeDefaults())
        let mock = MockAPIClient()
        mock.healthResult = .failure(FixtureError.failed)
        let viewModel = SetupViewModel(settings: settings) { _, _ in mock }
        viewModel.betaUserKey = "beta-user"
        viewModel.backendBaseURL = "http://localhost:8000"

        await viewModel.testConnection()

        XCTAssertEqual(viewModel.state, .failed("Fixture failure"))
    }

    private func makeDefaults() -> UserDefaults {
        let suiteName = "KonoproSetupTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
