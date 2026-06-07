import Foundation

@MainActor
final class RecordSessionViewModel: ObservableObject {
    enum State: Equatable {
        case ready
        case recording(URL)
        case selected(URL)
        case uploading
        case uploaded(UploadSessionResponse)
        case failed(String)
    }

    @Published private(set) var state: State = .ready

    private let recorder: AudioRecording
    private let uploader: UploadViewModel

    init(recorder: AudioRecording = AudioRecorder(), uploader: UploadViewModel) {
        self.recorder = recorder
        self.uploader = uploader
    }

    func startRecording() async {
        guard await recorder.requestPermission() else {
            state = .failed("Microphone permission is required.")
            return
        }
        do {
            state = .recording(try recorder.startRecording())
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func stopRecording() {
        do {
            if let url = try recorder.stopRecording() {
                state = .selected(url)
            } else {
                state = .ready
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func selectImportedFile(_ url: URL) {
        state = .selected(url)
    }

    func discard() {
        recorder.discardRecording()
        state = .ready
    }

    func uploadSelected() async {
        guard case .selected(let url) = state else {
            state = .failed("Choose or record audio first.")
            return
        }
        state = .uploading
        await uploader.upload(fileURL: url)
        switch uploader.state {
        case .uploaded(let response):
            state = .uploaded(response)
        case .failed(let message):
            state = .failed(message)
        default:
            break
        }
    }
}

