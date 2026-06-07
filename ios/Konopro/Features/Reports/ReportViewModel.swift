import Foundation

@MainActor
final class ReportViewModel: ObservableObject {
    enum State: Equatable {
        case idle
        case loading
        case loaded([ReportRequestResponse])
        case detail(ReportRequestResponse)
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    private let api: KonoproAPIProviding

    init(api: KonoproAPIProviding) {
        self.api = api
    }

    func childViewModel() -> ReportViewModel {
        ReportViewModel(api: api)
    }

    func requestReport(sessionId: String, requestType: String = "free", userNotes: String? = nil) async {
        state = .loading
        do {
            state = .detail(
                try await api.createReportRequest(
                    sessionId: sessionId,
                    requestType: requestType,
                    userNotes: userNotes
                )
            )
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func loadReports() async {
        state = .loading
        do {
            state = .loaded(try await api.listReportRequests())
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func loadReport(id: String) async {
        state = .loading
        do {
            state = .detail(try await api.getReportRequest(id: id))
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
