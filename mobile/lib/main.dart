import 'package:flutter/material.dart';

void main() {
  runApp(const KonoProApp());
}

class KonoProApp extends StatelessWidget {
  const KonoProApp({super.key});

  @override
  Widget build(BuildContext context) {
    const background = Color(0xFF101114);
    const surface = Color(0xFF1B1D22);
    const text = Color(0xFFF7F2EA);
    const accent = Color(0xFFE8EBD6);
    const secondary = Color(0xFF6DA7A1);

    return MaterialApp(
      title: 'KonoPro',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: background,
        colorScheme: ColorScheme.fromSeed(
          seedColor: secondary,
          brightness: Brightness.dark,
          surface: surface,
          primary: accent,
        ),
        textTheme: ThemeData.dark().textTheme.apply(
          bodyColor: text,
          displayColor: text,
          fontFamily: 'Roboto',
        ),
      ),
      home: const KonoProShell(),
    );
  }
}

class KonoProShell extends StatefulWidget {
  const KonoProShell({super.key});

  @override
  State<KonoProShell> createState() => _KonoProShellState();
}

class _KonoProShellState extends State<KonoProShell> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      const HomeScreen(),
      const RecordFlowScreen(),
      const FeedPlaceholderScreen(),
    ];

    return Scaffold(
      body: SafeArea(child: pages[_selectedIndex]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) =>
            setState(() => _selectedIndex = index),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.mic_none), label: 'Record'),
          NavigationDestination(
            icon: Icon(Icons.dynamic_feed_outlined),
            label: 'Feed',
          ),
        ],
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      children: [
        const HeaderBlock(),
        const SizedBox(height: 18),
        const StatusGrid(),
        const SizedBox(height: 16),
        const LastSessionCard(),
        const SizedBox(height: 22),
        Text(
          'Most practiced',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 10),
        for (final song in demoSongs)
          Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: SongPracticeCard(song: song),
          ),
      ],
    );
  }
}

class HeaderBlock extends StatelessWidget {
  const HeaderBlock({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'KonoPro',
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            fontWeight: FontWeight.w900,
            letterSpacing: 0,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          'Your karaoke practice, saved song by song.',
          style: Theme.of(
            context,
          ).textTheme.bodyMedium?.copyWith(color: Colors.white70),
        ),
      ],
    );
  }
}

class StatusGrid extends StatelessWidget {
  const StatusGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return const Row(
      children: [
        Expanded(
          child: StatTile(value: 'Lv. 4', label: 'mastered'),
        ),
        SizedBox(width: 10),
        Expanded(
          child: StatTile(value: '3 wk', label: 'streak'),
        ),
        SizedBox(width: 10),
        Expanded(
          child: StatTile(value: '2', label: 'credits'),
        ),
      ],
    );
  }
}

class StatTile extends StatelessWidget {
  const StatTile({required this.value, required this.label, super.key});

  final String value;
  final String label;

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900),
          ),
          const SizedBox(height: 3),
          Text(
            label,
            style: Theme.of(
              context,
            ).textTheme.labelMedium?.copyWith(color: Colors.white60),
          ),
        ],
      ),
    );
  }
}

class LastSessionCard extends StatelessWidget {
  const LastSessionCard({super.key});

  @override
  Widget build(BuildContext context) {
    return AppCard(
      emphasized: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Last session',
            style: Theme.of(
              context,
            ).textTheme.labelLarge?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 8),
          Text(
            '7 songs found',
            style: Theme.of(
              context,
            ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w900),
          ),
          const SizedBox(height: 6),
          const Text(
            '48 min session. 3 songs matched your earlier practice history.',
          ),
        ],
      ),
    );
  }
}

class SongPracticeCard extends StatelessWidget {
  const SongPracticeCard({required this.song, super.key});

  final PracticeSong song;

  @override
  Widget build(BuildContext context) {
    return AppCard(
      onTap: () {
        Navigator.of(context).push(
          MaterialPageRoute<void>(builder: (_) => SongPlayerScreen(song: song)),
        );
      },
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: song.color,
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.music_note, color: Color(0xFF101114)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  song.title,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  '${song.takeCount} takes saved',
                  style: const TextStyle(color: Colors.white60),
                ),
              ],
            ),
          ),
          const Icon(Icons.chevron_right),
        ],
      ),
    );
  }
}

class RecordFlowScreen extends StatefulWidget {
  const RecordFlowScreen({super.key});

  @override
  State<RecordFlowScreen> createState() => _RecordFlowScreenState();
}

class _RecordFlowScreenState extends State<RecordFlowScreen> {
  bool _isReady = false;

  @override
  Widget build(BuildContext context) {
    if (!_isReady) {
      return RecordHelper(onStart: () => setState(() => _isReady = true));
    }

    return const RecordingScreen();
  }
}

class RecordHelper extends StatelessWidget {
  const RecordHelper({required this.onStart, super.key});

  final VoidCallback onStart;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Before you sing',
            style: Theme.of(
              context,
            ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w900),
          ),
          const SizedBox(height: 10),
          const Text(
            'Place your phone near the karaoke speaker so KonoPro can split the session by song.',
          ),
          const SizedBox(height: 22),
          Expanded(
            child: AppCard(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 92,
                      height: 150,
                      decoration: BoxDecoration(
                        border: Border.all(
                          color: const Color(0xFFE8EBD6),
                          width: 4,
                        ),
                        borderRadius: BorderRadius.circular(32),
                      ),
                      child: const Icon(Icons.mic, size: 48),
                    ),
                    const SizedBox(height: 18),
                    const Text(
                      'Stable phone. Speaker nearby. Start before the first song.',
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: onStart,
            icon: const Icon(Icons.radio_button_checked),
            label: const Text('OK, start recording'),
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(54),
            ),
          ),
        ],
      ),
    );
  }
}

class RecordingScreen extends StatelessWidget {
  const RecordingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      children: [
        GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          childAspectRatio: 1.7,
          mainAxisSpacing: 10,
          crossAxisSpacing: 10,
          children: const [
            ClockTile(),
            MetricTile(label: 'Session', value: '18:42'),
            MetricTile(label: 'Tracks', value: '3'),
            MetricTile(label: 'Signal', value: 'Good'),
          ],
        ),
        const SizedBox(height: 16),
        AppCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Picked up',
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900),
              ),
              const SizedBox(height: 8),
              const DetectedTrackRow(
                title: 'Every Moment',
                time: '00:42 - 04:12',
                badge: 'live',
              ),
              const DetectedTrackRow(
                title: 'Night Letter',
                time: '06:10 - 09:51',
                badge: 'new',
              ),
              const DetectedTrackRow(
                title: 'Only Then',
                time: '12:16 - 16:44',
                badge: 'match',
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        const WaveformCard(),
      ],
    );
  }
}

class ClockTile extends StatelessWidget {
  const ClockTile({super.key});

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: Center(
        child: Text(
          '21:34',
          style: Theme.of(
            context,
          ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w900),
        ),
      ),
    );
  }
}

class MetricTile extends StatelessWidget {
  const MetricTile({required this.label, required this.value, super.key});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(color: Colors.white60)),
          const SizedBox(height: 5),
          Text(
            value,
            style: Theme.of(
              context,
            ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900),
          ),
        ],
      ),
    );
  }
}

class DetectedTrackRow extends StatelessWidget {
  const DetectedTrackRow({
    required this.title,
    required this.time,
    required this.badge,
    super.key,
  });

  final String title;
  final String time;
  final String badge;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 9),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                Text(time, style: const TextStyle(color: Colors.white60)),
              ],
            ),
          ),
          Chip(label: Text(badge)),
        ],
      ),
    );
  }
}

class WaveformCard extends StatelessWidget {
  const WaveformCard({super.key});

  static const heights = [
    24.0,
    52.0,
    34.0,
    82.0,
    44.0,
    96.0,
    38.0,
    66.0,
    28.0,
    74.0,
    46.0,
  ];

  @override
  Widget build(BuildContext context) {
    return AppCard(
      child: SizedBox(
        height: 124,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            for (final height in heights)
              Container(
                width: 7,
                height: height,
                decoration: BoxDecoration(
                  color: const Color(0xFFE8EBD6),
                  borderRadius: BorderRadius.circular(99),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class SongPlayerScreen extends StatelessWidget {
  const SongPlayerScreen({required this.song, super.key});

  final PracticeSong song;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(song.title)),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 10, 20, 24),
        children: [
          Container(
            height: 180,
            decoration: BoxDecoration(
              color: song.color,
              borderRadius: BorderRadius.circular(18),
            ),
            child: const Icon(
              Icons.graphic_eq,
              size: 72,
              color: Color(0xFF101114),
            ),
          ),
          const SizedBox(height: 18),
          Text(
            song.title,
            style: Theme.of(
              context,
            ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w900),
          ),
          const SizedBox(height: 4),
          Text(
            'All saved takes, newest first',
            style: Theme.of(
              context,
            ).textTheme.bodyMedium?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 18),
          AppCard(
            child: Row(
              children: [
                FilledButton(
                  onPressed: () {},
                  style: FilledButton.styleFrom(
                    shape: const CircleBorder(),
                    padding: const EdgeInsets.all(14),
                  ),
                  child: const Icon(Icons.pause),
                ),
                const SizedBox(width: 12),
                const Expanded(child: LinearProgressIndicator(value: 0.42)),
                const SizedBox(width: 12),
                IconButton(onPressed: () {}, icon: const Icon(Icons.replay_10)),
              ],
            ),
          ),
          const SizedBox(height: 16),
          for (final take in song.takes)
            Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: AppCard(
                child: Row(
                  children: [
                    const Icon(Icons.play_arrow),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            take.label,
                            style: const TextStyle(fontWeight: FontWeight.w800),
                          ),
                          Text(
                            take.subtitle,
                            style: const TextStyle(color: Colors.white60),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class FeedPlaceholderScreen extends StatelessWidget {
  const FeedPlaceholderScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Padding(
        padding: EdgeInsets.all(28),
        child: Text(
          'Feed is out of scope for this beta. The demo focuses on recording, song detection, and playback history.',
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}

class AppCard extends StatelessWidget {
  const AppCard({
    required this.child,
    this.onTap,
    this.emphasized = false,
    super.key,
  });

  final Widget child;
  final VoidCallback? onTap;
  final bool emphasized;

  @override
  Widget build(BuildContext context) {
    final card = Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: emphasized ? const Color(0xFF222A27) : const Color(0xFF1B1D22),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: emphasized ? const Color(0xFF3D514D) : const Color(0xFF30343C),
        ),
      ),
      child: child,
    );

    if (onTap == null) {
      return card;
    }

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(14),
      child: card,
    );
  }
}

class PracticeSong {
  const PracticeSong({
    required this.title,
    required this.takeCount,
    required this.color,
    required this.takes,
  });

  final String title;
  final int takeCount;
  final Color color;
  final List<PracticeTake> takes;
}

class PracticeTake {
  const PracticeTake({required this.label, required this.subtitle});

  final String label;
  final String subtitle;
}

const demoSongs = [
  PracticeSong(
    title: 'Every Moment',
    takeCount: 5,
    color: Color(0xFFE8EBD6),
    takes: [
      PracticeTake(
        label: 'June 12 take',
        subtitle: '4:30, detected chorus confidence high',
      ),
      PracticeTake(label: 'June 04 take', subtitle: '4:22, previous attempt'),
      PracticeTake(label: 'May 28 take', subtitle: '4:18, first saved attempt'),
    ],
  ),
  PracticeSong(
    title: 'Night Letter',
    takeCount: 3,
    color: Color(0xFF6DA7A1),
    takes: [
      PracticeTake(label: 'June 12 take', subtitle: '3:41, clean match'),
      PracticeTake(label: 'June 01 take', subtitle: '3:35, earlier practice'),
    ],
  ),
  PracticeSong(
    title: 'Only Then',
    takeCount: 2,
    color: Color(0xFFE2B36A),
    takes: [
      PracticeTake(label: 'June 12 take', subtitle: '4:28, latest performance'),
      PracticeTake(label: 'May 30 take', subtitle: '4:18, first saved attempt'),
    ],
  ),
];
