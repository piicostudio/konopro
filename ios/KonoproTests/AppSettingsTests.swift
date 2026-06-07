import XCTest
@testable import Konopro

@MainActor
final class AppSettingsTests: XCTestCase {
    func testSettingsPersistAndNormalizeValues() {
        let defaults = makeDefaults()
        let settings = AppSettings(defaults: defaults)

        XCTAssertFalse(settings.isConfigured)
        XCTAssertEqual(settings.backendBaseURL, AppSettings.defaultBackendBaseURL)

        settings.update(
            betaUserKey: " beta-user ",
            backendBaseURL: " http://localhost:8000 "
        )

        XCTAssertEqual(settings.betaUserKey, "beta-user")
        XCTAssertEqual(settings.backendBaseURL, "http://localhost:8000")
        XCTAssertTrue(settings.isConfigured)

        let reloaded = AppSettings(defaults: defaults)
        XCTAssertEqual(reloaded.betaUserKey, "beta-user")
        XCTAssertEqual(reloaded.backendURL?.absoluteString, "http://localhost:8000")
    }

    func testResetClearsIdentityAndRestoresDefaultBackend() {
        let settings = AppSettings(defaults: makeDefaults())
        settings.update(betaUserKey: "beta-user", backendBaseURL: "http://localhost:8000")

        settings.reset()

        XCTAssertEqual(settings.betaUserKey, "")
        XCTAssertEqual(settings.backendBaseURL, AppSettings.defaultBackendBaseURL)
        XCTAssertFalse(settings.isConfigured)
    }

    func testAppStateRoutesFromCurrentSettings() {
        let setupSettings = AppSettings(defaults: makeDefaults())
        XCTAssertEqual(AppState(settings: setupSettings).route, .setup)

        let mainSettings = AppSettings(defaults: makeDefaults())
        mainSettings.update(betaUserKey: "beta-user", backendBaseURL: "http://localhost:8000")
        XCTAssertEqual(AppState(settings: mainSettings).route, .main)
    }

    private func makeDefaults() -> UserDefaults {
        let suiteName = "KonoproTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
