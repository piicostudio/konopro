import SwiftUI
import UniformTypeIdentifiers

struct RecordSessionView: View {
    @EnvironmentObject private var settings: AppSettings
    @StateObject var viewModel: RecordSessionViewModel
    @State private var isImporterPresented = false

    init(viewModel: RecordSessionViewModel? = nil) {
        _viewModel = StateObject(
            wrappedValue: viewModel ?? RecordSessionViewModel(
                uploader: UploadViewModel(api: KonoproAPIClient(
                    baseURL: URL(string: AppSettings.defaultBackendBaseURL)!,
                    betaUserKey: ""
                ))
            )
        )
    }

    var body: some View {
        VStack(spacing: 20) {
            stateText

            HStack {
                Button("Record") {
                    Task { await viewModel.startRecording() }
                }
                .buttonStyle(.borderedProminent)

                Button("Import") {
                    isImporterPresented = true
                }
                .buttonStyle(.bordered)
            }

            HStack {
                Button("Stop") {
                    viewModel.stopRecording()
                }
                Button("Discard") {
                    viewModel.discard()
                }
                Button("Upload") {
                    Task { await viewModel.uploadSelected() }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .navigationTitle("New Session")
        .fileImporter(isPresented: $isImporterPresented, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result {
                viewModel.selectImportedFile(url)
            }
        }
    }

    @ViewBuilder
    private var stateText: some View {
        switch viewModel.state {
        case .ready:
            Text("Record or import a karaoke session.")
        case .recording:
            Label("Recording...", systemImage: "record.circle")
                .foregroundStyle(.red)
        case .selected(let url):
            Text(url.lastPathComponent)
        case .uploading:
            ProgressView("Uploading...")
        case .uploaded(let response):
            Label("Uploaded \(response.session.originalFilename)", systemImage: "checkmark.circle")
                .foregroundStyle(.green)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
        }
    }
}

