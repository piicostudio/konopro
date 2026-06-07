import Foundation

@MainActor
final class SessionListViewModel: ObservableObject {
    enum State: Equatable {
        case idle
        case loading
        case loaded([AudioSessionResponse])
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    private let api: KonoproAPIProviding

    init(api: KonoproAPIProviding) {
        self.api = api
    }

    func load() async {
        state = .loading
        do {
            state = .loaded(try await api.listSessions())
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func sessions() -> [AudioSessionResponse] {
        if case .loaded(let sessions) = state {
            return sessions
        }
        return []
    }
}

