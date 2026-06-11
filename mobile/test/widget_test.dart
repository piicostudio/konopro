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
  const completedJob = BackendJob(
    id: 'job-123456789',
    sessionId: 'session-1',
    status: 'completed',
    jobType: 'fingerprint_segmentation',
  );
  const failedJob = BackendJob(
    id: 'job-123456789',
    sessionId: 'session-1',
    status: 'failed',
    jobType: 'fingerprint_segmentation',
    errorMessage: 'processor failed',
  );
  const sampleFile = PickedAudioFile(
    name: 'karaoke.wav',
    bytes: [1, 2, 3, 4],
    contentType: 'audio/wav',
  );
  const sampleAnalysis = SessionAnalysis(
    status: 'completed',
    provider: 'fake',
    summary: AnalysisSummary(
      status: 'accepted_intervals',
      message: 'Likely song intervals were found.',
      confidenceLevel: 'high',
      acceptedIntervalCount: 1,
      weakCandidateCount: 0,
    ),
    intervals: [
      DetectedSongSegment(
        title: 'Demo Song',
        artist: 'Demo Artist',
        startS: 42,
        endS: 252,
        confidenceLabel: 'high',
        confidenceScore: 91.2,
      ),
    ],
    weakCandidates: [],
  );
  const weakAnalysis = SessionAnalysis(
    status: 'completed',
    provider: 'fake',
    summary: AnalysisSummary(
      status: 'weak_candidates',
      message: 'Only weak song clues were found.',
      confidenceLevel: 'weak_clues',
      acceptedIntervalCount: 0,
      weakCandidateCount: 1,
    ),
    intervals: [],
    weakCandidates: [
      DetectedSongSegment(
        title: 'Maybe Song',
        artist: 'Maybe Artist',
        startS: 0,
        endS: 10,
        confidenceLabel: 'weak',
        confidenceScore: 0.4,
        reason: 'singleton_match',
        isWeak: true,
      ),
    ],
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

  test('BackendJob decodes backend job JSON with failure message', () {
    final job = BackendJob.fromJson({
      'id': 'job-1',
      'session_id': 'session-1',
      'status': 'failed',
      'job_type': 'fingerprint_segmentation',
      'error_message': 'bad audio',
    });

    expect(job.id, 'job-1');
    expect(job.sessionId, 'session-1');
    expect(job.status, 'failed');
    expect(job.errorMessage, 'bad audio');
  });

  test('SessionAnalysis decodes intervals and weak candidates', () {
    final analysis = SessionAnalysis.fromJson({
      'status': 'completed',
      'provider': 'fake',
      'result_summary': {
        'status': 'weak_candidates',
        'message': 'Only weak clues.',
        'confidence_level': 'weak_clues',
        'accepted_interval_count': 0,
        'weak_candidate_count': 1,
      },
      'intervals': [
        {
          'song': 'Demo Song',
          'artist': 'Demo Artist',
          'start_s': 42.0,
          'end_s': 252.0,
          'confidence_level': 'high',
          'confidence_score': 91.2,
        },
      ],
      'weak_candidates': [
        {
          'song': 'Maybe Song',
          'artist': 'Maybe Artist',
          'start_s': 0.0,
          'end_s': 10.0,
          'provider_confidence': 0.4,
          'reason': 'singleton_match',
        },
      ],
    });

    expect(analysis.summary.status, 'weak_candidates');
    expect(analysis.intervals.single.title, 'Demo Song');
    expect(analysis.intervals.single.confidenceScore, 91.2);
    expect(analysis.weakCandidates.single.isWeak, isTrue);
    expect(analysis.weakCandidates.single.reason, 'singleton_match');
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
    final playbackController = FakeSegmentPlaybackController();

    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        jobPollInterval: Duration.zero,
        uploadSessionClient: (baseUrl, betaIdentity, file) async {
          uploadedBaseUrl = baseUrl;
          uploadedIdentity = betaIdentity;
          uploadedFile = file;
          return UploadSessionResult(session: sampleSession, job: sampleJob);
        },
        jobStatusClient: (_, _, _) async => completedJob,
        sessionAnalysisClient: (_, _, _) async => sampleAnalysis,
        segmentPlaybackControllerFactory: () => playbackController,
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

  testWidgets('record upload polls job and displays accepted analysis', (
    tester,
  ) async {
    final playbackController = FakeSegmentPlaybackController();
    final statuses = [
      sampleJob,
      const BackendJob(
        id: 'job-123456789',
        sessionId: 'session-1',
        status: 'processing',
        jobType: 'fingerprint_segmentation',
      ),
      completedJob,
    ];
    var statusIndex = 0;

    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        uploadSessionClient: (_, _, _) async =>
            UploadSessionResult(session: sampleSession, job: sampleJob),
        jobStatusClient: (_, _, _) async =>
            statuses[statusIndex++ % statuses.length],
        sessionAnalysisClient: (_, _, _) async => sampleAnalysis,
        segmentPlaybackControllerFactory: () => playbackController,
        jobPollInterval: Duration.zero,
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
    await tester.pumpAndSettle();

    await tester.scrollUntilVisible(
      find.text('1 detected song segment'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('1 detected song segment'), findsOneWidget);
    expect(find.text('Demo Song - Demo Artist'), findsOneWidget);
    expect(find.text('0:42 - 4:12'), findsOneWidget);
    expect(find.text('high'), findsOneWidget);

    await tester.tap(find.byIcon(Icons.play_arrow).first);
    await tester.pump();
    await tester.pump();

    expect(
      playbackController.audioUri.toString(),
      'http://127.0.0.1:8000/v1/sessions/session-1/audio',
    );
    expect(playbackController.betaIdentity, 'peter-demo');
    expect(playbackController.segment?.title, 'Demo Song');
    expect(find.byIcon(Icons.pause), findsOneWidget);

    await tester.tap(find.byIcon(Icons.pause));
    await tester.pump();

    expect(playbackController.pauseCount, 1);
    expect(find.byIcon(Icons.play_arrow), findsWidgets);
  });

  testWidgets('record upload displays weak analysis without overclaiming', (
    tester,
  ) async {
    final playbackController = FakeSegmentPlaybackController();
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        uploadSessionClient: (_, _, _) async =>
            UploadSessionResult(session: sampleSession, job: sampleJob),
        jobStatusClient: (_, _, _) async => completedJob,
        sessionAnalysisClient: (_, _, _) async => weakAnalysis,
        segmentPlaybackControllerFactory: () => playbackController,
        jobPollInterval: Duration.zero,
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
    await tester.pumpAndSettle();

    await tester.scrollUntilVisible(
      find.text('1 weak clue found'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('1 weak clue found'), findsOneWidget);
    expect(find.text('Maybe Song - Maybe Artist'), findsOneWidget);
    expect(find.text('weak'), findsOneWidget);
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

  testWidgets('record upload shows processing failure from job polling', (
    tester,
  ) async {
    await tester.pumpWidget(
      KonoProApp(
        healthCheckClient: (_) async =>
            const BackendHealth(status: 'ok', environment: 'local'),
        sessionListClient: (_, _) async => [],
        audioFilePicker: () async => sampleFile,
        uploadSessionClient: (_, _, _) async =>
            UploadSessionResult(session: sampleSession, job: sampleJob),
        jobStatusClient: (_, _, _) async => failedJob,
        sessionAnalysisClient: (_, _, _) async => sampleAnalysis,
        jobPollInterval: Duration.zero,
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
    await tester.pumpAndSettle();

    await tester.scrollUntilVisible(
      find.text('Processing failed'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('Processing failed'), findsOneWidget);
    expect(find.text('processor failed'), findsOneWidget);
  });
}

class FakeSegmentPlaybackController implements SegmentPlaybackController {
  Uri? audioUri;
  String? betaIdentity;
  DetectedSongSegment? segment;
  int pauseCount = 0;
  int disposeCount = 0;

  @override
  Future<void> playSegment({
    required Uri audioUri,
    required String betaIdentity,
    required DetectedSongSegment segment,
  }) async {
    this.audioUri = audioUri;
    this.betaIdentity = betaIdentity;
    this.segment = segment;
  }

  @override
  Future<void> pause() async {
    pauseCount += 1;
  }

  @override
  Future<void> dispose() async {
    disposeCount += 1;
  }
}
