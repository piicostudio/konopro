import Foundation

struct AnalysisDisplayState: Equatable {
    let title: String
    let message: String
    let isConfident: Bool
    let limitations: [String]

    init(analysis: FingerprintAnalysisResponse) {
        title = analysis.resultSummary.status
        message = analysis.resultSummary.message
        isConfident = analysis.resultSummary.acceptedIntervalCount > 0
        limitations = analysis.resultSummary.limitations
    }
}

