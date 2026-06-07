import SwiftUI

struct SetupView: View {
    @StateObject var viewModel: SetupViewModel

    var body: some View {
        Form {
            Section("Beta Access") {
                TextField("Beta key", text: $viewModel.betaUserKey)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField("Backend URL", text: $viewModel.backendBaseURL)
                    .keyboardType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }

            Section {
                Button {
                    Task { await viewModel.testConnection() }
                } label: {
                    Label("Test Connection", systemImage: "network")
                }

                Button {
                    viewModel.save()
                } label: {
                    Label("Save", systemImage: "checkmark.circle")
                }
                .disabled(!viewModel.canSave)
            }

            Section {
                switch viewModel.state {
                case .idle:
                    EmptyView()
                case .checking:
                    ProgressView("Checking backend...")
                case .valid(let message):
                    Label(message, systemImage: "checkmark.circle")
                        .foregroundStyle(.green)
                case .failed(let message):
                    Label(message, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle("Konopro Setup")
    }
}

