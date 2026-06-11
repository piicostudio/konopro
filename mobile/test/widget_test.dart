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
  const sampleJob = BackendJob(
    id: 'job-123456789',
    sessionId: 'session-1',
    status: 'queued',
    jobType: 'fingerprint_segmentation',
  );
  const sampleFile = PickedAudioFile(
    name: 'karaoke.wav',
    bytes: [1, 2, 3, 4],
    contentType: 'audio/wav',
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

  test('UploadSessionResult decodes backend upload JSON', () {
    final result = UploadSessionResult.fromJson({
      'session': {
        'id': 'session-1',
        'original_filename': 'karaoke.wav',
        'status': 'queued',
        'size_bytes': 4,
        'duration_s': null,
        'processing_job_id': 'job-1',
        'created_at': '2026-06-12T01:02:03.123456',
      },
      'job': {
        'id': 'job-1',
        'session_id': 'session-1',
        'status': 'queued',
        'job_type': 'fingerprint_segmentation',
      },
    });

    expect(result.session.id, 'session-1');
    expect(result.session.originalFilename, 'karaoke.wav');
    expect(result.job.id, 'job-1');
    expect(result.job.sessionId, 'session-1');
    expect(result.job.jobType, 'fingerprint_segmentation');
  });

  test('multipart upload body includes file and source parts', () {
    final body = buildUploadSessionBody(
      file: sampleFile,
      boundary: 'test-boundary',
    );
    final text = String.fromCharCodes(body.bytes);

    expect(body.contentType, 'multipart/form-data; boundary=test-boundary');
    expect(text, contains('name="source"'));
    expect(text, contains('flutter_app'));
    expect(text, contains('name="file"; filename="karaoke.wav"'));
    expect(text, contains('Content-Type: audio/wav'));
    expect(body.bytes, containsAll(sampleFile.bytes));
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

    expect(find.text('Tracks'), findsOneWidget);
    expect(find.text('Backend required'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('Picked up'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('Picked up'), findsOneWidget);
  });

  testWidgets('record upload shows success state and returned IDs', (
    tester,
  ) async {
    Uri? uploadedBaseUrl;
    String? uploadedIdentity;
    PickedAudioFile? uploadedFile;

    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        uploadSessionClient: (baseUrl, betaIdentity, file) async {
          uploadedBaseUrl = baseUrl;
          uploadedIdentity = betaIdentity;
          uploadedFile = file;
          return UploadSessionResult(session: sampleSession, job: sampleJob);
        },
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('OK, start recording'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Choose file'));
    await tester.pump();
    await tester.pump();
    expect(find.text('karaoke.wav'), findsOneWidget);

    await tester.tap(find.text('Upload'));
    await tester.pumpAndSettle();

    expect(uploadedBaseUrl.toString(), 'http://127.0.0.1:8000');
    expect(uploadedIdentity, 'peter-demo');
    expect(uploadedFile, sampleFile);
    expect(find.text('Uploaded'), findsOneWidget);
    expect(find.textContaining('Job job-1234'), findsOneWidget);
  });

  testWidgets('record upload shows failure state', (tester) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        uploadSessionClient: (_, _, _) async =>
            throw const BackendConnectionException('upload failed'),
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    await tester.tap(find.text('Record'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('OK, start recording'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose file'));
    await tester.pump();
    await tester.pump();
    await tester.tap(find.text('Upload'));
    await tester.pump();
    await tester.pump();

    expect(find.text('Upload failed'), findsOneWidget);
    expect(find.text('upload failed'), findsOneWidget);
  });
}
