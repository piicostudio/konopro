import SwiftUI

@MainActor
final class SessionDetailViewModel: ObservableObject {
    @Published private(set) var session: AudioSessionResponse
    @Published private(set) var job: ProcessingJobResponse?
    @Published private(set) var analysis: FingerprintAnalysisResponse?
    @Published private(set) var errorMessage: String?
    @Published var isPolling = false

    private let api: KonoproAPIProviding

    init(session: AudioSessionResponse, api: KonoproAPIProviding) {
        self.session = session
        self.api = api
    }

    func refresh() async {
        if let jobId = session.processingJobId {
            do {
                job = try await api.getJob(id: jobId)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        await loadAnalysis()
    }

    func pollJob(maxAttempts: Int = 5) async {
        guard let jobId = session.processingJobId else { return }
        isPolling = true
        defer { isPolling = false }
        for _ in 0..<maxAttempts {
            do {
                let current = try await api.getJob(id: jobId)
                job = current
                if current.status.isTerminal {
                    await loadAnalysis()
                    return
                }
            } catch {
                errorMessage = error.localizedDescription
                return
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    func loadAnalysis() async {
        do {
            analysis = try await api.getAnalysis(sessionId: session.id)
        } catch {
            analysis = nil
        }
    }
}

struct SessionDetailView: View {
    @EnvironmentObject private var settings: AppSettings
    @StateObject var viewModel: SessionDetailViewModel

    var body: some View {
        List {
            Section("Session") {
                LabeledContent("File", value: viewModel.session.originalFilename)
                LabeledContent("Status", value: viewModel.session.status.rawValue)
                if let job = viewModel.job {
                    LabeledContent("Job", value: job.status.rawValue)
                }
            }

            if viewModel.isPolling {
                ProgressView("Checking processing status...")
            }

            if let analysis = viewModel.analysis {
                AnalysisSummaryView(analysis: analysis)
            } else {
                Section("Instant Analysis") {
                    Text("Analysis is not ready yet.")
                        .foregroundStyle(.secondary)
                }
            }

            Section("Verified Report") {
                NavigationLink("Request or view report") {
                    ReportRequestView(
                        viewModel: ReportViewModel(api: KonoproAPIClient(
                            settings: settings
                        )),
                        sessionId: viewModel.session.id
                    )
                }
            }

            if let errorMessage = viewModel.errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle("Session")
        .task {
            await viewModel.refresh()
            await viewModel.pollJob()
        }
    }
}
