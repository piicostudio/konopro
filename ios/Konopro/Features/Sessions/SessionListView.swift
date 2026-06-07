import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var settings: AppSettings
    @StateObject var viewModel: SessionListViewModel

    var body: some View {
        List {
            switch viewModel.state {
            case .idle, .loading:
                ProgressView("Loading sessions...")
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            case .loaded(let sessions) where sessions.isEmpty:
                ContentUnavailableView("No sessions yet", systemImage: "music.mic")
            case .loaded(let sessions):
                ForEach(sessions) { session in
                    NavigationLink {
                        SessionDetailView(
                            viewModel: SessionDetailViewModel(
                                session: session,
                                api: KonoproAPIClient(settings: settings)
                            )
                        )
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(session.originalFilename)
                                .font(.headline)
                            Text(session.status.rawValue)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .navigationTitle("Sessions")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                NavigationLink {
                    RecordSessionView(
                        viewModel: RecordSessionViewModel(
                            uploader: UploadViewModel(api: KonoproAPIClient(settings: settings))
                        )
                    )
                } label: {
                    Image(systemName: "plus.circle.fill")
                }
                .accessibilityLabel("New session")
            }
        }
        .task {
            await viewModel.load()
        }
        .refreshable {
            await viewModel.load()
        }
    }
}

