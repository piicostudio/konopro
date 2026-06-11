import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:konopro/main.dart';

void main() {
  final sampleSession = BackendSession(
    id: 'session-1',
    originalFilename: 'karaoke-night.m4a',
    status: 'uploaded',
    sizeBytes: 2048,
    durationS: 245,
    processingJobId: 'job-1',
    createdAt: DateTime.parse('2026-06-12T01:00:00'),
  );

  test('BackendSession decodes backend session JSON', () {
    final session = BackendSession.fromJson({
      'id': 'session-1',
      'original_filename': 'room.wav',
      'status': 'processed',
      'size_bytes': 4096,
      'duration_s': 123.4,
      'processing_job_id': 'job-1',
      'created_at': '2026-06-12T01:02:03.123456',
    });

    expect(session.id, 'session-1');
    expect(session.originalFilename, 'room.wav');
    expect(session.status, 'processed');
    expect(session.sizeBytes, 4096);
    expect(session.durationS, 123.4);
    expect(session.processingJobId, 'job-1');
    expect(session.createdAt.year, 2026);
  });

  testWidgets('home shows the practice dashboard', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    expect(find.text('KonoPro'), findsOneWidget);
    expect(find.text('Backend URL'), findsOneWidget);
    expect(find.text('Test connection'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('7 songs found'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('7 songs found'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('Most practiced'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('Most practiced'), findsOneWidget);
    expect(find.text('Every Moment'), findsOneWidget);
  });

  testWidgets('connection panel shows backend health success', (tester) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [sampleSession],
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    expect(find.text('Connected to local'), findsOneWidget);
    expect(find.text('Backend status: ok'), findsOneWidget);
    expect(find.text('karaoke-night.m4a'), findsOneWidget);
    expect(find.text('4:05 • 2.0 KB'), findsOneWidget);
  });

  testWidgets('connection panel shows backend health failure', (tester) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            throw const BackendConnectionException('network failed'),
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    expect(find.text('Connection failed'), findsOneWidget);
    expect(find.text('network failed'), findsOneWidget);
  });

  testWidgets('connection panel validates backend URL', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    await tester.enterText(find.byType(TextField).first, 'not-a-url');
    await tester.tap(find.text('Test connection'));
    await tester.pump();

    expect(
      find.text('Enter a full URL like http://127.0.0.1:8000.'),
      findsOneWidget,
    );
  });

  testWidgets('session list shows empty state after connection', (
    tester,
  ) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    expect(find.text('No sessions yet'), findsOneWidget);
    expect(
      find.text(
        'Upload or record karaoke audio to start your practice history.',
      ),
      findsOneWidget,
    );
  });

  testWidgets('session list shows backend error state', (tester) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async =>
            throw const BackendConnectionException('session load failed'),
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    expect(find.text('Could not load sessions'), findsOneWidget);
    expect(find.text('session load failed'), findsOneWidget);
  });

  testWidgets('song card opens playback history', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    await tester.scrollUntilVisible(
      find.text('Every Moment'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    await tester.tap(find.text('Every Moment'));
    await tester.pumpAndSettle();

    expect(find.text('All saved takes, newest first'), findsOneWidget);
    expect(find.text('June 12 take'), findsOneWidget);
    expect(find.byIcon(Icons.pause), findsOneWidget);
  });

  testWidgets('record tab starts with setup helper', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();

    expect(find.text('Before you sing'), findsOneWidget);
    expect(find.text('OK, start recording'), findsOneWidget);

    await tester.tap(find.text('OK, start recording'));
    await tester.pumpAndSettle();

    expect(find.text('Picked up'), findsOneWidget);
    expect(find.text('Tracks'), findsOneWidget);
  });
}
