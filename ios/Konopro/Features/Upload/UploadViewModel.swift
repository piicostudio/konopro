import Foundation

@MainActor
final class UploadViewModel: ObservableObject {
    enum State: Equatable {
        case idle
        case uploading(Double)
        case uploaded(UploadSessionResponse)
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    private let api: KonoproAPIProviding

    init(api: KonoproAPIProviding) {
        self.api = api
    }

    func upload(fileURL: URL, source: String = "ios_app") async {
        state = .uploading(0.05)
        do {
            let response = try await api.uploadSession(
                fileURL: fileURL,
                source: source,
                clientDurationS: nil
            )
            state = .uploading(1.0)
            state = .uploaded(response)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}

