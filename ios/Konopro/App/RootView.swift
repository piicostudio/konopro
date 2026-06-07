import SwiftUI

struct RootView: View {
    @EnvironmentObject private var settings: AppSettings

    var body: some View {
        Group {
            if settings.isConfigured {
                MainTabView()
                    .environmentObject(settings)
            } else {
                SetupView(viewModel: SetupViewModel(settings: settings))
            }
        }
    }
}

struct MainTabView: View {
    @EnvironmentObject private var settings: AppSettings

    var body: some View {
        TabView {
            NavigationStack {
                SessionListView(
                    viewModel: SessionListViewModel(
                        api: KonoproAPIClient(settings: settings)
                    )
                )
            }
            .tabItem {
                Label("Sessions", systemImage: "music.mic")
            }

            NavigationStack {
                ReportListView(
                    viewModel: ReportViewModel(
                        api: KonoproAPIClient(settings: settings)
                    )
                )
            }
            .tabItem {
                Label("Reports", systemImage: "doc.text")
            }

            NavigationStack {
                SetupView(viewModel: SetupViewModel(settings: settings))
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
        }
    }
}
