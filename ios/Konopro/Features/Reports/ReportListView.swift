import SwiftUI

struct ReportListView: View {
    @StateObject var viewModel: ReportViewModel

    var body: some View {
        List {
            switch viewModel.state {
            case .idle, .loading:
                ProgressView("Loading reports...")
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            case .detail(let report):
                ReportDetailContent(report: report)
            case .loaded(let reports) where reports.isEmpty:
                ContentUnavailableView("No report requests", systemImage: "doc.text")
            case .loaded(let reports):
                ForEach(reports) { report in
                    NavigationLink {
                        ReportDetailView(
                            viewModel: viewModel.childViewModel(),
                            requestId: report.id
                        )
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(report.requestType.capitalized)
                                .font(.headline)
                            Text(report.status.rawValue)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .navigationTitle("Reports")
        .task { await viewModel.loadReports() }
        .refreshable { await viewModel.loadReports() }
    }
}
