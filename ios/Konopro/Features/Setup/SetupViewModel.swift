import Foundation

@MainActor
final class SetupViewModel: ObservableObject {
    enum State: Equatable {
        case idle
        case checking
        case valid(String)
        case failed(String)
    }

    @Published var betaUserKey: String
    @Published var backendBaseURL: String
    @Published private(set) var state: State = .idle

    private let settings: AppSettings
    private let makeClient: (URL, String) -> KonoproAPIProviding

    init(
        settings: AppSettings,
        makeClient: @escaping (URL, String) -> KonoproAPIProviding = {
            KonoproAPIClient(baseURL: $0, betaUserKey: $1)
        }
    ) {
        self.settings = settings
        self.betaUserKey = settings.betaUserKey
        self.backendBaseURL = settings.backendBaseURL
        self.makeClient = makeClient
    }

    var canSave: Bool {
        normalizedBetaKey.isEmpty == false && parsedURL != nil
    }

    func testConnection() async {
        guard let url = parsedURL, !normalizedBetaKey.isEmpty else {
            state = .failed("Enter a beta key and a valid backend URL.")
            return
        }
        state = .checking
        do {
            let health = try await makeClient(url, normalizedBetaKey).health()
            state = .valid("Connected to \(health.environment).")
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func save() {
        guard canSave else {
            state = .failed("Enter a beta key and a valid backend URL.")
            return
        }
        settings.update(betaUserKey: normalizedBetaKey, backendBaseURL: normalizedBackendURL)
        state = .valid("Settings saved.")
    }

    private var normalizedBetaKey: String {
        betaUserKey.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var normalizedBackendURL: String {
        backendBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var parsedURL: URL? {
        URL(string: normalizedBackendURL)
    }
}

