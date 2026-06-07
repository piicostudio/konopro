import Foundation

@MainActor
final class AppState: ObservableObject {
    enum Route: Equatable {
        case setup
        case main
    }

    @Published private(set) var route: Route
    let settings: AppSettings

    init(settings: AppSettings) {
        self.settings = settings
        self.route = settings.isConfigured ? .main : .setup
    }

    func refreshRoute() {
        route = settings.isConfigured ? .main : .setup
    }
}
