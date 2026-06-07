import Foundation

@MainActor
final class AppSettings: ObservableObject {
    private enum Keys {
        static let betaUserKey = "konopro.betaUserKey"
        static let backendBaseURL = "konopro.backendBaseURL"
    }

    static let defaultBackendBaseURL = "http://127.0.0.1:8000"

    private let defaults: UserDefaults

    @Published var betaUserKey: String {
        didSet { defaults.set(betaUserKey, forKey: Keys.betaUserKey) }
    }

    @Published var backendBaseURL: String {
        didSet { defaults.set(backendBaseURL, forKey: Keys.backendBaseURL) }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.betaUserKey = defaults.string(forKey: Keys.betaUserKey) ?? ""
        self.backendBaseURL = defaults.string(forKey: Keys.backendBaseURL) ?? Self.defaultBackendBaseURL
    }

    var isConfigured: Bool {
        !betaUserKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && URL(string: backendBaseURL) != nil
    }

    var backendURL: URL? {
        URL(string: backendBaseURL.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func update(betaUserKey: String, backendBaseURL: String) {
        self.betaUserKey = betaUserKey.trimmingCharacters(in: .whitespacesAndNewlines)
        self.backendBaseURL = backendBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func reset() {
        betaUserKey = ""
        backendBaseURL = Self.defaultBackendBaseURL
    }
}

