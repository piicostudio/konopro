import AVFoundation
import Foundation

protocol AudioRecording {
    var isRecording: Bool { get }
    func requestPermission() async -> Bool
    func startRecording() throws -> URL
    func stopRecording() throws -> URL?
    func discardRecording()
}

final class AudioRecorder: NSObject, AudioRecording, AVAudioRecorderDelegate {
    private var recorder: AVAudioRecorder?
    private(set) var currentFileURL: URL?

    var isRecording: Bool {
        recorder?.isRecording == true
    }

    func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording() throws -> URL {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default)
        try session.setActive(true)

        let url = Self.makeRecordingURL()
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44_100,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.delegate = self
        recorder.record()
        self.recorder = recorder
        self.currentFileURL = url
        return url
    }

    func stopRecording() throws -> URL? {
        recorder?.stop()
        recorder = nil
        try AVAudioSession.sharedInstance().setActive(false)
        return currentFileURL
    }

    func discardRecording() {
        recorder?.stop()
        recorder = nil
        if let currentFileURL {
            try? FileManager.default.removeItem(at: currentFileURL)
        }
        currentFileURL = nil
    }

    private static func makeRecordingURL() -> URL {
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return directory.appendingPathComponent("konopro-\(UUID().uuidString).m4a")
    }
}

