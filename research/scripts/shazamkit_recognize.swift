import AVFoundation
import Foundation
import ShazamKit

func emit(_ payload: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

func errorPayload(_ message: String) -> [String: Any] {
    return [
        "status": "error",
        "error": message,
    ]
}

func mediaPayload(_ item: SHMatchedMediaItem) -> [String: Any] {
    var media: [String: Any] = [
        "title": item.title ?? "",
        "artist": item.artist ?? "",
        "subtitle": item.subtitle ?? "",
        "apple_music_id": item.appleMusicID ?? "",
        "shazam_id": item.shazamID ?? "",
        "isrc": item.isrc ?? "",
        "web_url": item.webURL?.absoluteString ?? "",
        "apple_music_url": item.appleMusicURL?.absoluteString ?? "",
        "artwork_url": item.artworkURL?.absoluteString ?? "",
        "match_offset_s": item.matchOffset,
        "frequency_skew": item.frequencySkew,
    ]
    if #available(macOS 15.4, *) {
        media["confidence"] = item.confidence
    }
    if !item.genres.isEmpty {
        media["genres"] = item.genres
    }
    return media
}

func main() async {
    guard CommandLine.arguments.count >= 2 else {
        emit(errorPayload("usage: swift shazamkit_recognize.swift <audio-path>"))
        Foundation.exit(2)
    }

    let audioURL = URL(fileURLWithPath: CommandLine.arguments[1])
    guard FileManager.default.fileExists(atPath: audioURL.path) else {
        emit(errorPayload("audio file not found: \(audioURL.path)"))
        Foundation.exit(2)
    }

    do {
        let asset = AVURLAsset(url: audioURL)
        let signature = try await SHSignatureGenerator.signature(from: asset)
        let result = await SHSession().result(from: signature)

        switch result {
        case .match(let match):
            guard let item = match.mediaItems.first else {
                emit([
                    "status": "no_match",
                    "media_count": 0,
                ])
                return
            }
            emit([
                "status": "matched",
                "media_count": match.mediaItems.count,
                "media": mediaPayload(item),
            ])
        case .noMatch:
            emit([
                "status": "no_match",
                "media_count": 0,
            ])
        case .error(let error, _):
            emit(errorPayload(String(describing: error)))
            Foundation.exit(1)
        @unknown default:
            emit(errorPayload("unknown ShazamKit result"))
            Foundation.exit(1)
        }
    } catch {
        emit(errorPayload(String(describing: error)))
        Foundation.exit(1)
    }
}

await main()
