import SwiftUI

struct ReportRequestView: View {
    @StateObject var viewModel: ReportViewModel
    let sessionId: String
    @State private var requestType = "free"
    @State private var notes = ""

    var body: some View {
        Form {
            Section("Verified Report") {
                Picker("Request type", selection: $requestType) {
                    Text("Free").tag("free")
                    Text("Priority").tag("paid")
                }
                TextField("Notes", text: $notes, axis: .vertical)
                Button("Request Report") {
                    Task {
                        await viewModel.requestReport(
                            sessionId: sessionId,
                            requestType: requestType,
                            userNotes: notes.isEmpty ? nil : notes
                        )
                    }
                }
            }

            reportState
        }
        .navigationTitle("Verified Report")
    }

    @ViewBuilder
    private var reportState: some View {
        switch viewModel.state {
        case .idle:
            EmptyView()
        case .loading:
            ProgressView("Loading...")
        case .loaded:
            EmptyView()
        case .detail(let report):
            ReportDetailContent(report: report)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }
}

