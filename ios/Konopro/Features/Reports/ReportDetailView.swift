import SwiftUI

struct ReportDetailView: View {
    @StateObject private var viewModel: ReportViewModel
    let requestId: String

    init(viewModel: ReportViewModel, requestId: String) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.requestId = requestId
    }

    var body: some View {
        List {
            switch viewModel.state {
            case .idle, .loading:
                ProgressView("Loading report...")
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            case .loaded:
                EmptyView()
            case .detail(let report):
                ReportDetailContent(report: report)
            }
        }
        .navigationTitle("Report")
        .task { await viewModel.loadReport(id: requestId) }
    }
}

struct ReportDetailContent: View {
    let report: ReportRequestResponse

    var body: some View {
        Section("Status") {
            LabeledContent("State", value: report.status.rawValue)
            LabeledContent("Priority", value: report.priority.rawValue)
            if let blocker = report.blockerReason {
                Text(blocker)
                    .foregroundStyle(.secondary)
            }
        }

        if report.artifacts.isEmpty {
            Section("Delivered Report") {
                Text("No delivered report yet.")
                    .foregroundStyle(.secondary)
            }
        } else {
            Section("Delivered Report") {
                ForEach(report.artifacts) { artifact in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(artifact.title)
                            .font(.headline)
                        if let body = artifact.bodyText {
                            Text(body)
                        }
                    }
                }
            }
        }
    }
}
