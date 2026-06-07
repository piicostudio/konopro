import SwiftUI

struct AnalysisSummaryView: View {
    let analysis: FingerprintAnalysisResponse

    var body: some View {
        let display = AnalysisDisplayState(analysis: analysis)
        Section("Instant Experimental Analysis") {
            Label(display.message, systemImage: display.isConfident ? "waveform" : "questionmark.circle")
                .foregroundStyle(display.isConfident ? .primary : .secondary)

            if !analysis.intervals.isEmpty {
                ForEach(analysis.intervals) { interval in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(interval.song.isEmpty ? "Unknown song" : interval.song)
                            .font(.headline)
                        Text("\(interval.startS, specifier: "%.1f")s - \(interval.endS, specifier: "%.1f")s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if !analysis.weakCandidates.isEmpty {
                ForEach(analysis.weakCandidates) { candidate in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(candidate.song.isEmpty ? "Weak clue" : candidate.song)
                            .font(.subheadline)
                        Text(candidate.reason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            ForEach(display.limitations, id: \.self) { limitation in
                Text(limitation)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

