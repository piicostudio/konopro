import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';

void main() {
  runApp(const KonoProApp());
}

typedef HealthCheckClient = Future<BackendHealth> Function(Uri baseUrl);
typedef SessionListClient =
    Future<List<BackendSession>> Function(Uri baseUrl, String betaIdentity);
typedef UploadSessionClient =
    Future<UploadSessionResult> Function(
      Uri baseUrl,
      String betaIdentity,
      PickedAudioFile file,
    );
typedef AudioFilePicker = Future<PickedAudioFile?> Function();
typedef JobStatusClient =
    Future<BackendJob> Function(Uri baseUrl, String betaIdentity, String jobId);
typedef SessionAnalysisClient =
    Future<SessionAnalysis> Function(
      Uri baseUrl,
      String betaIdentity,
      String sessionId,
    );
typedef SegmentPlaybackControllerFactory = SegmentPlaybackController Function();

abstract class SegmentPlaybackController {
  Future<void> playSegment({
    required Uri audioUri,
    required String betaIdentity,
    required DetectedSongSegment segment,
  });

  Future<void> pause();

  Future<void> dispose();
}

class JustAudioSegmentPlaybackController implements SegmentPlaybackController {
  JustAudioSegmentPlaybackController({AudioPlayer? player})
    : _player = player ?? AudioPlayer();

  final AudioPlayer _player;

  @override
  Future<void> playSegment({
    required Uri audioUri,
    required String betaIdentity,
    required DetectedSongSegment segment,
  }) async {
    await _player.setAudioSource(
      ClippingAudioSource(
        start: _durationFromSeconds(segment.startS),
        end: _durationFromSeconds(segment.endS),
        child: ProgressiveAudioSource(
          audioUri,
          headers: {'X-Konopro-Beta-User': betaIdentity},
        ),
      ),
    );
    await _player.play();
  }

  @override
  Future<void> pause() => _player.pause();

  @override
  Future<void> dispose() => _player.dispose();
}

class BackendHealth {
  const BackendHealth({required this.status, required this.environment});

  final String status;
  final String environment;
}

class BackendConnectionException implements Exception {
  const BackendConnectionException(this.message);

  final String message;

  @override
  String toString() => message;
}

class BackendSession {
  const BackendSession({
    required this.id,
    required this.originalFilename,
    required this.status,
    required this.sizeBytes,
    required this.createdAt,
    this.durationS,
    this.processingJobId,
  });

  final String id;
  final String originalFilename;
  final String status;
  final int sizeBytes;
  final DateTime createdAt;
  final double? durationS;
  final String? processingJobId;

  factory BackendSession.fromJson(Map<String, dynamic> json) {
    return BackendSession(
      id: json['id']?.toString() ?? '',
      originalFilename: json['original_filename']?.toString() ?? 'upload',
      status: json['status']?.toString() ?? 'unknown',
      sizeBytes: json['size_bytes'] is num
          ? (json['size_bytes'] as num).toInt()
          : 0,
      durationS: json['duration_s'] is num
          ? (json['duration_s'] as num).toDouble()
          : null,
      processingJobId: json['processing_job_id']?.toString(),
      createdAt:
          DateTime.tryParse(json['created_at']?.toString() ?? '') ??
          DateTime.fromMillisecondsSinceEpoch(0),
    );
  }
}

class BackendJob {
  const BackendJob({
    required this.id,
    required this.sessionId,
    required this.status,
    required this.jobType,
    this.errorMessage,
  });

  final String id;
  final String sessionId;
  final String status;
  final String jobType;
  final String? errorMessage;

  factory BackendJob.fromJson(Map<String, dynamic> json) {
    return BackendJob(
      id: json['id']?.toString() ?? '',
      sessionId: json['session_id']?.toString() ?? '',
      status: json['status']?.toString() ?? 'unknown',
      jobType: json['job_type']?.toString() ?? 'unknown',
      errorMessage: json['error_message']?.toString(),
    );
  }
}

class UploadSessionResult {
  const UploadSessionResult({required this.session, required this.job});

  final BackendSession session;
  final BackendJob job;

  factory UploadSessionResult.fromJson(Map<String, dynamic> json) {
    final sessionJson = json['session'];
    final jobJson = json['job'];
    if (sessionJson is! Map<String, dynamic> ||
        jobJson is! Map<String, dynamic>) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }
    return UploadSessionResult(
      session: BackendSession.fromJson(sessionJson),
      job: BackendJob.fromJson(jobJson),
    );
  }
}

class PickedAudioFile {
  const PickedAudioFile({
    required this.name,
    required this.bytes,
    required this.contentType,
  });

  final String name;
  final List<int> bytes;
  final String contentType;

  int get sizeBytes => bytes.length;
}

class MultipartUploadBody {
  const MultipartUploadBody({required this.contentType, required this.bytes});

  final String contentType;
  final List<int> bytes;
}

class AnalysisSummary {
  const AnalysisSummary({
    required this.status,
    required this.message,
    required this.confidenceLevel,
    required this.acceptedIntervalCount,
    required this.weakCandidateCount,
  });

  final String status;
  final String message;
  final String confidenceLevel;
  final int acceptedIntervalCount;
  final int weakCandidateCount;

  factory AnalysisSummary.fromJson(Map<String, dynamic> json) {
    return AnalysisSummary(
      status: json['status']?.toString() ?? 'unknown',
      message: json['message']?.toString() ?? '',
      confidenceLevel: json['confidence_level']?.toString() ?? 'unknown',
      acceptedIntervalCount: json['accepted_interval_count'] is num
          ? (json['accepted_interval_count'] as num).toInt()
          : 0,
      weakCandidateCount: json['weak_candidate_count'] is num
          ? (json['weak_candidate_count'] as num).toInt()
          : 0,
    );
  }
}

class DetectedSongSegment {
  const DetectedSongSegment({
    required this.title,
    required this.artist,
    required this.startS,
    required this.endS,
    required this.confidenceLabel,
    required this.confidenceScore,
    this.reason,
    this.isWeak = false,
  });

  final String title;
  final String artist;
  final double startS;
  final double endS;
  final String confidenceLabel;
  final double? confidenceScore;
  final String? reason;
  final bool isWeak;

  factory DetectedSongSegment.fromIntervalJson(Map<String, dynamic> json) {
    return DetectedSongSegment(
      title: json['song']?.toString() ?? 'Unknown song',
      artist: json['artist']?.toString() ?? '',
      startS: json['start_s'] is num ? (json['start_s'] as num).toDouble() : 0,
      endS: json['end_s'] is num ? (json['end_s'] as num).toDouble() : 0,
      confidenceLabel: json['confidence_level']?.toString() ?? 'unknown',
      confidenceScore: json['confidence_score'] is num
          ? (json['confidence_score'] as num).toDouble()
          : null,
    );
  }

  factory DetectedSongSegment.fromWeakCandidateJson(Map<String, dynamic> json) {
    return DetectedSongSegment(
      title: json['song']?.toString() ?? 'Unknown clue',
      artist: json['artist']?.toString() ?? '',
      startS: json['start_s'] is num ? (json['start_s'] as num).toDouble() : 0,
      endS: json['end_s'] is num ? (json['end_s'] as num).toDouble() : 0,
      confidenceLabel: 'weak',
      confidenceScore: json['provider_confidence'] is num
          ? (json['provider_confidence'] as num).toDouble()
          : null,
      reason: json['reason']?.toString(),
      isWeak: true,
    );
  }
}

class SessionAnalysis {
  const SessionAnalysis({
    required this.status,
    required this.provider,
    required this.summary,
    required this.intervals,
    required this.weakCandidates,
  });

  final String status;
  final String provider;
  final AnalysisSummary summary;
  final List<DetectedSongSegment> intervals;
  final List<DetectedSongSegment> weakCandidates;

  factory SessionAnalysis.fromJson(Map<String, dynamic> json) {
    final summaryJson = json['result_summary'];
    return SessionAnalysis(
      status: json['status']?.toString() ?? 'unknown',
      provider: json['provider']?.toString() ?? 'unknown',
      summary: summaryJson is Map<String, dynamic>
          ? AnalysisSummary.fromJson(summaryJson)
          : const AnalysisSummary(
              status: 'unknown',
              message: '',
              confidenceLevel: 'unknown',
              acceptedIntervalCount: 0,
              weakCandidateCount: 0,
            ),
      intervals: [
        for (final item
            in json['intervals'] is List ? json['intervals'] as List : const [])
          if (item is Map<String, dynamic>)
            DetectedSongSegment.fromIntervalJson(item),
      ],
      weakCandidates: [
        for (final item
            in json['weak_candidates'] is List
                ? json['weak_candidates'] as List
                : const [])
          if (item is Map<String, dynamic>)
            DetectedSongSegment.fromWeakCandidateJson(item),
      ],
    );
  }
}

Future<BackendHealth> defaultBackendHealthCheck(Uri baseUrl) async {
  final healthUri = baseUrl.replace(
    path: _joinUriPath(baseUrl.path, 'health'),
    queryParameters: null,
    fragment: null,
  );
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);

  try {
    final request = await client.getUrl(healthUri);
    final response = await request.close().timeout(const Duration(seconds: 8));
    final body = await response.transform(utf8.decoder).join();

    if (response.statusCode != HttpStatus.ok) {
      throw BackendConnectionException(
        'Backend returned HTTP ${response.statusCode}.',
      );
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }

    return BackendHealth(
      status: decoded['status']?.toString() ?? 'unknown',
      environment: decoded['environment']?.toString() ?? 'unknown',
    );
  } on BackendConnectionException {
    rethrow;
  } on TimeoutException {
    throw const BackendConnectionException('Connection timed out.');
  } on FormatException {
    throw const BackendConnectionException('Backend returned invalid JSON.');
  } on SocketException catch (error) {
    throw BackendConnectionException(error.message);
  } finally {
    client.close(force: true);
  }
}

Future<List<BackendSession>> defaultSessionListClient(
  Uri baseUrl,
  String betaIdentity,
) async {
  final sessionsUri = baseUrl.replace(
    path: _joinUriPath(baseUrl.path, '/v1/sessions'),
    queryParameters: null,
    fragment: null,
  );
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);

  try {
    final request = await client.getUrl(sessionsUri);
    request.headers.set('X-Konopro-Beta-User', betaIdentity);
    final response = await request.close().timeout(const Duration(seconds: 8));
    final body = await response.transform(utf8.decoder).join();

    if (response.statusCode != HttpStatus.ok) {
      throw BackendConnectionException(
        'Backend returned HTTP ${response.statusCode}.',
      );
    }

    final decoded = jsonDecode(body);
    if (decoded is! List) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }

    return [
      for (final item in decoded)
        if (item is Map<String, dynamic>) BackendSession.fromJson(item),
    ];
  } on BackendConnectionException {
    rethrow;
  } on TimeoutException {
    throw const BackendConnectionException('Connection timed out.');
  } on FormatException {
    throw const BackendConnectionException('Backend returned invalid JSON.');
  } on SocketException catch (error) {
    throw BackendConnectionException(error.message);
  } finally {
    client.close(force: true);
  }
}

Future<PickedAudioFile?> defaultAudioFilePicker() async {
  final result = await FilePicker.pickFiles(
    type: FileType.audio,
    allowMultiple: false,
    withData: true,
  );
  final file = result?.files.single;
  final bytes = file?.bytes;
  if (file == null || bytes == null) {
    return null;
  }
  return PickedAudioFile(
    name: file.name,
    bytes: bytes,
    contentType: _guessAudioContentType(file.name),
  );
}

Future<UploadSessionResult> defaultUploadSessionClient(
  Uri baseUrl,
  String betaIdentity,
  PickedAudioFile file,
) async {
  final uploadUri = baseUrl.replace(
    path: _joinUriPath(baseUrl.path, '/v1/sessions'),
    queryParameters: null,
    fragment: null,
  );
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);
  final boundary = 'konopro-${DateTime.now().microsecondsSinceEpoch}';
  final body = buildUploadSessionBody(file: file, boundary: boundary);

  try {
    final request = await client.postUrl(uploadUri);
    request.headers.set('X-Konopro-Beta-User', betaIdentity);
    request.headers.set(HttpHeaders.contentTypeHeader, body.contentType);
    request.contentLength = body.bytes.length;
    request.add(body.bytes);

    final response = await request.close().timeout(const Duration(seconds: 30));
    final responseBody = await response.transform(utf8.decoder).join();

    if (response.statusCode != HttpStatus.created) {
      throw BackendConnectionException(
        'Backend returned HTTP ${response.statusCode}.',
      );
    }

    final decoded = jsonDecode(responseBody);
    if (decoded is! Map<String, dynamic>) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }
    return UploadSessionResult.fromJson(decoded);
  } on BackendConnectionException {
    rethrow;
  } on TimeoutException {
    throw const BackendConnectionException('Upload timed out.');
  } on FormatException {
    throw const BackendConnectionException('Backend returned invalid JSON.');
  } on SocketException catch (error) {
    throw BackendConnectionException(error.message);
  } finally {
    client.close(force: true);
  }
}

Future<BackendJob> defaultJobStatusClient(
  Uri baseUrl,
  String betaIdentity,
  String jobId,
) async {
  final jobUri = baseUrl.replace(
    path: _joinUriPath(baseUrl.path, '/v1/jobs/$jobId'),
    queryParameters: null,
    fragment: null,
  );
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);

  try {
    final request = await client.getUrl(jobUri);
    request.headers.set('X-Konopro-Beta-User', betaIdentity);
    final response = await request.close().timeout(const Duration(seconds: 8));
    final body = await response.transform(utf8.decoder).join();

    if (response.statusCode != HttpStatus.ok) {
      throw BackendConnectionException(
        'Backend returned HTTP ${response.statusCode}.',
      );
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }
    return BackendJob.fromJson(decoded);
  } on BackendConnectionException {
    rethrow;
  } on TimeoutException {
    throw const BackendConnectionException('Connection timed out.');
  } on FormatException {
    throw const BackendConnectionException('Backend returned invalid JSON.');
  } on SocketException catch (error) {
    throw BackendConnectionException(error.message);
  } finally {
    client.close(force: true);
  }
}

Future<SessionAnalysis> defaultSessionAnalysisClient(
  Uri baseUrl,
  String betaIdentity,
  String sessionId,
) async {
  final analysisUri = baseUrl.replace(
    path: _joinUriPath(baseUrl.path, '/v1/sessions/$sessionId/analysis'),
    queryParameters: null,
    fragment: null,
  );
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);

  try {
    final request = await client.getUrl(analysisUri);
    request.headers.set('X-Konopro-Beta-User', betaIdentity);
    final response = await request.close().timeout(const Duration(seconds: 8));
    final body = await response.transform(utf8.decoder).join();

    if (response.statusCode != HttpStatus.ok) {
      throw BackendConnectionException(
        response.statusCode == HttpStatus.notFound
            ? 'Analysis not ready.'
            : 'Backend returned HTTP ${response.statusCode}.',
      );
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const BackendConnectionException('Backend returned invalid JSON.');
    }
    return SessionAnalysis.fromJson(decoded);
  } on BackendConnectionException {
    rethrow;
  } on TimeoutException {
    throw const BackendConnectionException('Connection timed out.');
  } on FormatException {
    throw const BackendConnectionException('Backend returned invalid JSON.');
  } on SocketException catch (error) {
    throw BackendConnectionException(error.message);
  } finally {
    client.close(force: true);
  }
}

MultipartUploadBody buildUploadSessionBody({
  required PickedAudioFile file,
  required String boundary,
}) {
  final builder = BytesBuilder(copy: false);
  void addText(String value) => builder.add(utf8.encode(value));

  addText('--$boundary\r\n');
  addText('Content-Disposition: form-data; name="source"\r\n\r\n');
  addText('flutter_app\r\n');
  addText('--$boundary\r\n');
  addText(
    'Content-Disposition: form-data; name="file"; filename="${_escapeMultipartHeader(file.name)}"\r\n',
  );
  addText('Content-Type: ${file.contentType}\r\n\r\n');
  builder.add(file.bytes);
  addText('\r\n--$boundary--\r\n');

  return MultipartUploadBody(
    contentType: 'multipart/form-data; boundary=$boundary',
    bytes: builder.takeBytes(),
  );
}

String _escapeMultipartHeader(String value) {
  return value.replaceAll('"', r'\"');
}

String _guessAudioContentType(String filename) {
  final lower = filename.toLowerCase();
  if (lower.endsWith('.wav')) return 'audio/wav';
  if (lower.endsWith('.m4a')) return 'audio/mp4';
  if (lower.endsWith('.aac')) return 'audio/aac';
  if (lower.endsWith('.ogg')) return 'audio/ogg';
  if (lower.endsWith('.flac')) return 'audio/flac';
  return 'audio/mpeg';
}

String _joinUriPath(String basePath, String childPath) {
  final cleanBase = basePath.endsWith('/')
      ? basePath.substring(0, basePath.length - 1)
      : basePath;
  final cleanChild = childPath.startsWith('/')
      ? childPath.substring(1)
      : childPath;
  if (cleanBase.isEmpty) {
    return '/$cleanChild';
  }
  return '$cleanBase/$cleanChild';
}

Uri? _sessionAudioUri(Uri? baseUrl, String sessionId) {
  if (baseUrl == null) {
    return null;
  }
  return baseUrl.replace(
    path: _joinUriPath(baseUrl.path, '/v1/sessions/$sessionId/audio'),
    queryParameters: null,
    fragment: null,
  );
}

class KonoProApp extends StatelessWidget {
  const KonoProApp({
    super.key,
    this.healthCheckClient,
    this.sessionListClient,
    this.uploadSessionClient,
    this.audioFilePicker,
    this.jobStatusClient,
    this.sessionAnalysisClient,
    this.segmentPlaybackControllerFactory,
    this.jobPollInterval = const Duration(seconds: 2),
  });

  final HealthCheckClient? healthCheckClient;
  final SessionListClient? sessionListClient;
  final UploadSessionClient? uploadSessionClient;
  final AudioFilePicker? audioFilePicker;
  final JobStatusClient? jobStatusClient;
  final SessionAnalysisClient? sessionAnalysisClient;
  final SegmentPlaybackControllerFactory? segmentPlaybackControllerFactory;
  final Duration jobPollInterval;

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
      home: KonoProShell(
        healthCheckClient: healthCheckClient ?? defaultBackendHealthCheck,
        sessionListClient: sessionListClient ?? defaultSessionListClient,
        uploadSessionClient: uploadSessionClient ?? defaultUploadSessionClient,
        audioFilePicker: audioFilePicker ?? defaultAudioFilePicker,
        jobStatusClient: jobStatusClient ?? defaultJobStatusClient,
        sessionAnalysisClient:
            sessionAnalysisClient ?? defaultSessionAnalysisClient,
        segmentPlaybackControllerFactory:
            segmentPlaybackControllerFactory ??
            () => JustAudioSegmentPlaybackController(),
        jobPollInterval: jobPollInterval,
      ),
    );
  }
}

class KonoProShell extends StatefulWidget {
  const KonoProShell({
    required this.healthCheckClient,
    required this.sessionListClient,
    required this.uploadSessionClient,
    required this.audioFilePicker,
    required this.jobStatusClient,
    required this.sessionAnalysisClient,
    required this.segmentPlaybackControllerFactory,
    required this.jobPollInterval,
    super.key,
  });

  final HealthCheckClient healthCheckClient;
  final SessionListClient sessionListClient;
  final UploadSessionClient uploadSessionClient;
  final AudioFilePicker audioFilePicker;
  final JobStatusClient jobStatusClient;
  final SessionAnalysisClient sessionAnalysisClient;
  final SegmentPlaybackControllerFactory segmentPlaybackControllerFactory;
  final Duration jobPollInterval;

  @override
  State<KonoProShell> createState() => _KonoProShellState();
}

class _KonoProShellState extends State<KonoProShell> {
  int _selectedIndex = 0;
  Uri? _backendUrl;
  String _betaIdentity = 'peter-demo';

  void _setBackendConnection(Uri backendUrl, String betaIdentity) {
    setState(() {
      _backendUrl = backendUrl;
      _betaIdentity = betaIdentity;
    });
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomeScreen(
        healthCheckClient: widget.healthCheckClient,
        sessionListClient: widget.sessionListClient,
        initialBackendUrl: _backendUrl,
        initialBetaIdentity: _betaIdentity,
        onBackendConnected: _setBackendConnection,
      ),
      RecordFlowScreen(
        backendUrl: _backendUrl,
        betaIdentity: _betaIdentity,
        uploadSessionClient: widget.uploadSessionClient,
        audioFilePicker: widget.audioFilePicker,
        jobStatusClient: widget.jobStatusClient,
        sessionAnalysisClient: widget.sessionAnalysisClient,
        segmentPlaybackControllerFactory:
            widget.segmentPlaybackControllerFactory,
        jobPollInterval: widget.jobPollInterval,
      ),
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

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    required this.healthCheckClient,
    required this.sessionListClient,
    required this.initialBackendUrl,
    required this.initialBetaIdentity,
    required this.onBackendConnected,
    super.key,
  });

  final HealthCheckClient healthCheckClient;
  final SessionListClient sessionListClient;
  final Uri? initialBackendUrl;
  final String initialBetaIdentity;
  final void Function(Uri backendUrl, String betaIdentity) onBackendConnected;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  Uri? _backendUrl;
  late String _betaIdentity;
  List<BackendSession>? _sessions;
  String? _sessionError;
  bool _isLoadingSessions = false;

  @override
  void initState() {
    super.initState();
    _backendUrl = widget.initialBackendUrl;
    _betaIdentity = widget.initialBetaIdentity;
  }

  Future<void> _loadSessions(Uri backendUrl, String betaIdentity) async {
    setState(() {
      _backendUrl = backendUrl;
      _betaIdentity = betaIdentity;
      _isLoadingSessions = true;
      _sessionError = null;
    });
    widget.onBackendConnected(backendUrl, betaIdentity);

    try {
      final sessions = await widget.sessionListClient(backendUrl, betaIdentity);
      if (!mounted) return;
      setState(() => _sessions = sessions);
    } on BackendConnectionException catch (error) {
      if (!mounted) return;
      setState(() => _sessionError = error.message);
    } catch (error) {
      if (!mounted) return;
      setState(() => _sessionError = error.toString());
    } finally {
      if (mounted) {
        setState(() => _isLoadingSessions = false);
      }
    }
  }

  Future<void> _refreshSessions() async {
    final backendUrl = _backendUrl;
    if (backendUrl == null) return;
    await _loadSessions(backendUrl, _betaIdentity);
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      children: [
        const HeaderBlock(),
        const SizedBox(height: 18),
        BackendConnectionCard(
          healthCheckClient: widget.healthCheckClient,
          onConnected: _loadSessions,
        ),
        const SizedBox(height: 16),
        BackendSessionListCard(
          sessions: _sessions,
          isLoading: _isLoadingSessions,
          errorMessage: _sessionError,
          onRefresh: _refreshSessions,
        ),
        const SizedBox(height: 16),
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

class BackendConnectionCard extends StatefulWidget {
  const BackendConnectionCard({
    required this.healthCheckClient,
    required this.onConnected,
    super.key,
  });

  final HealthCheckClient healthCheckClient;
  final Future<void> Function(Uri backendUrl, String betaIdentity) onConnected;

  @override
  State<BackendConnectionCard> createState() => _BackendConnectionCardState();
}

class _BackendConnectionCardState extends State<BackendConnectionCard> {
  final _backendUrlController = TextEditingController(
    text: 'http://127.0.0.1:8000',
  );
  final _identityController = TextEditingController(text: 'peter-demo');
  BackendHealth? _health;
  String? _errorMessage;
  bool _isChecking = false;

  @override
  void dispose() {
    _backendUrlController.dispose();
    _identityController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    final baseUrl = Uri.tryParse(_backendUrlController.text.trim());
    final betaIdentity = _identityController.text.trim();
    if (baseUrl == null || !baseUrl.hasScheme || baseUrl.host.isEmpty) {
      setState(() {
        _health = null;
        _errorMessage = 'Enter a full URL like http://127.0.0.1:8000.';
      });
      return;
    }
    if (betaIdentity.isEmpty) {
      setState(() {
        _health = null;
        _errorMessage = 'Enter a beta identity.';
      });
      return;
    }

    setState(() {
      _isChecking = true;
      _health = null;
      _errorMessage = null;
    });

    try {
      final health = await widget.healthCheckClient(baseUrl);
      if (!mounted) return;
      setState(() => _health = health);
      await widget.onConnected(baseUrl, betaIdentity);
    } on BackendConnectionException catch (error) {
      if (!mounted) return;
      setState(() => _errorMessage = error.message);
    } catch (error) {
      if (!mounted) return;
      setState(() => _errorMessage = error.toString());
    } finally {
      if (mounted) {
        setState(() => _isChecking = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final connected = _health != null;
    final statusText = connected
        ? 'Connected to ${_health!.environment}'
        : _errorMessage == null
        ? 'Not connected'
        : 'Connection failed';
    final detailText = connected
        ? 'Backend status: ${_health!.status}'
        : _errorMessage ?? 'Test your local backend before uploading audio.';

    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                connected
                    ? Icons.cloud_done_outlined
                    : Icons.cloud_off_outlined,
                color: connected
                    ? const Color(0xFF6DA7A1)
                    : Theme.of(context).colorScheme.primary,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      statusText,
                      style: const TextStyle(fontWeight: FontWeight.w900),
                    ),
                    Text(
                      detailText,
                      style: const TextStyle(color: Colors.white60),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _backendUrlController,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'Backend URL',
              prefixIcon: Icon(Icons.link),
            ),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _identityController,
            decoration: const InputDecoration(
              labelText: 'Beta identity',
              prefixIcon: Icon(Icons.person_outline),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _isChecking ? null : _testConnection,
            icon: _isChecking
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.sync),
            label: Text(_isChecking ? 'Checking' : 'Test connection'),
          ),
        ],
      ),
    );
  }
}

class BackendSessionListCard extends StatelessWidget {
  const BackendSessionListCard({
    required this.sessions,
    required this.isLoading,
    required this.errorMessage,
    required this.onRefresh,
    super.key,
  });

  final List<BackendSession>? sessions;
  final bool isLoading;
  final String? errorMessage;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    final sessions = this.sessions;
    return AppCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Recent sessions',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900),
                ),
              ),
              IconButton(
                onPressed: isLoading ? null : onRefresh,
                icon: const Icon(Icons.refresh),
                tooltip: 'Refresh sessions',
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (isLoading)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: Center(child: CircularProgressIndicator()),
            )
          else if (errorMessage != null)
            SessionMessage(
              icon: Icons.error_outline,
              title: 'Could not load sessions',
              detail: errorMessage!,
            )
          else if (sessions == null)
            const SessionMessage(
              icon: Icons.cloud_queue_outlined,
              title: 'Connect to your backend',
              detail: 'Your uploaded karaoke sessions will appear here.',
            )
          else if (sessions.isEmpty)
            const SessionMessage(
              icon: Icons.library_music_outlined,
              title: 'No sessions yet',
              detail:
                  'Upload or record karaoke audio to start your practice history.',
            )
          else
            for (final session in sessions.take(3))
              BackendSessionRow(session: session),
        ],
      ),
    );
  }
}

class SessionMessage extends StatelessWidget {
  const SessionMessage({
    required this.icon,
    required this.title,
    required this.detail,
    super.key,
  });

  final IconData icon;
  final String title;
  final String detail;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: Colors.white70),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 3),
                Text(detail, style: const TextStyle(color: Colors.white60)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class BackendSessionRow extends StatelessWidget {
  const BackendSessionRow({required this.session, super.key});

  final BackendSession session;

  @override
  Widget build(BuildContext context) {
    final duration = session.durationS == null
        ? 'duration unknown'
        : _formatDuration(session.durationS!);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          const Icon(Icons.audiotrack_outlined),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  session.originalFilename,
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
                Text(
                  '$duration • ${_formatBytes(session.sizeBytes)}',
                  style: const TextStyle(color: Colors.white60),
                ),
              ],
            ),
          ),
          Chip(label: Text(session.status)),
        ],
      ),
    );
  }
}

String _formatDuration(double seconds) {
  final totalSeconds = seconds.round();
  final minutes = totalSeconds ~/ 60;
  final remainingSeconds = totalSeconds % 60;
  return '$minutes:${remainingSeconds.toString().padLeft(2, '0')}';
}

String _formatBytes(int bytes) {
  if (bytes < 1024) {
    return '$bytes B';
  }
  final kib = bytes / 1024;
  if (kib < 1024) {
    return '${kib.toStringAsFixed(1)} KB';
  }
  return '${(kib / 1024).toStringAsFixed(1)} MB';
}

Duration _durationFromSeconds(double seconds) {
  return Duration(milliseconds: (seconds * 1000).round());
}

String _shortId(String id) {
  if (id.length <= 8) {
    return id;
  }
  return id.substring(0, 8);
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
  const RecordFlowScreen({
    required this.backendUrl,
    required this.betaIdentity,
    required this.uploadSessionClient,
    required this.audioFilePicker,
    required this.jobStatusClient,
    required this.sessionAnalysisClient,
    required this.segmentPlaybackControllerFactory,
    required this.jobPollInterval,
    super.key,
  });

  final Uri? backendUrl;
  final String betaIdentity;
  final UploadSessionClient uploadSessionClient;
  final AudioFilePicker audioFilePicker;
  final JobStatusClient jobStatusClient;
  final SessionAnalysisClient sessionAnalysisClient;
  final SegmentPlaybackControllerFactory segmentPlaybackControllerFactory;
  final Duration jobPollInterval;

  @override
  State<RecordFlowScreen> createState() => _RecordFlowScreenState();
}

class _RecordFlowScreenState extends State<RecordFlowScreen> {
  bool _isReady = false;
  PickedAudioFile? _selectedFile;
  UploadSessionResult? _uploadResult;
  BackendJob? _currentJob;
  SessionAnalysis? _analysis;
  String? _uploadError;
  String? _analysisError;
  bool _isPicking = false;
  bool _isUploading = false;
  bool _isPollingJob = false;

  Future<void> _pickAudioFile() async {
    setState(() {
      _isPicking = true;
      _uploadError = null;
    });

    try {
      final file = await widget.audioFilePicker();
      if (!mounted) return;
      if (file != null) {
        setState(() {
          _selectedFile = file;
          _uploadResult = null;
          _currentJob = null;
          _analysis = null;
          _analysisError = null;
        });
      }
    } catch (error) {
      if (!mounted) return;
      setState(() => _uploadError = error.toString());
    } finally {
      if (mounted) {
        setState(() => _isPicking = false);
      }
    }
  }

  Future<void> _uploadAudioFile() async {
    final backendUrl = widget.backendUrl;
    final file = _selectedFile;
    if (backendUrl == null) {
      setState(() => _uploadError = 'Connect to your backend from Home first.');
      return;
    }
    if (file == null) {
      setState(() => _uploadError = 'Choose an audio file first.');
      return;
    }

    setState(() {
      _isUploading = true;
      _uploadError = null;
      _analysisError = null;
      _uploadResult = null;
      _currentJob = null;
      _analysis = null;
    });

    try {
      final result = await widget.uploadSessionClient(
        backendUrl,
        widget.betaIdentity,
        file,
      );
      if (!mounted) return;
      setState(() {
        _uploadResult = result;
        _currentJob = result.job;
      });
      unawaited(_pollJobAndAnalysis(backendUrl, widget.betaIdentity, result));
    } on BackendConnectionException catch (error) {
      if (!mounted) return;
      setState(() => _uploadError = error.message);
    } catch (error) {
      if (!mounted) return;
      setState(() => _uploadError = error.toString());
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  Future<void> _pollJobAndAnalysis(
    Uri backendUrl,
    String betaIdentity,
    UploadSessionResult uploadResult,
  ) async {
    setState(() {
      _isPollingJob = true;
      _analysisError = null;
    });

    BackendJob latestJob = uploadResult.job;
    try {
      for (var attempt = 0; attempt < 12; attempt += 1) {
        latestJob = await widget.jobStatusClient(
          backendUrl,
          betaIdentity,
          uploadResult.job.id,
        );
        if (!mounted) return;
        setState(() => _currentJob = latestJob);

        if (_isTerminalJobStatus(latestJob.status)) {
          break;
        }
        await Future<void>.delayed(widget.jobPollInterval);
      }

      if (!mounted) return;
      if (latestJob.status == 'completed') {
        final analysis = await widget.sessionAnalysisClient(
          backendUrl,
          betaIdentity,
          uploadResult.session.id,
        );
        if (!mounted) return;
        setState(() => _analysis = analysis);
      } else if (latestJob.status == 'failed') {
        setState(() {
          final errorMessage = latestJob.errorMessage;
          _analysisError = errorMessage == null || errorMessage.isEmpty
              ? 'Processing failed on the backend.'
              : errorMessage;
        });
      } else if (latestJob.status == 'cancelled') {
        setState(() => _analysisError = 'Processing was cancelled.');
      }
    } on BackendConnectionException catch (error) {
      if (!mounted) return;
      setState(() => _analysisError = error.message);
    } catch (error) {
      if (!mounted) return;
      setState(() => _analysisError = error.toString());
    } finally {
      if (mounted) {
        setState(() => _isPollingJob = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isReady) {
      return RecordHelper(onStart: () => setState(() => _isReady = true));
    }

    return RecordingScreen(
      backendUrl: widget.backendUrl,
      betaIdentity: widget.betaIdentity,
      selectedFile: _selectedFile,
      uploadResult: _uploadResult,
      currentJob: _currentJob,
      analysis: _analysis,
      uploadError: _uploadError,
      analysisError: _analysisError,
      segmentPlaybackControllerFactory: widget.segmentPlaybackControllerFactory,
      isPicking: _isPicking,
      isUploading: _isUploading,
      isPollingJob: _isPollingJob,
      onPickFile: _pickAudioFile,
      onUpload: _uploadAudioFile,
    );
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
  const RecordingScreen({
    required this.backendUrl,
    required this.betaIdentity,
    required this.selectedFile,
    required this.uploadResult,
    required this.currentJob,
    required this.analysis,
    required this.uploadError,
    required this.analysisError,
    required this.segmentPlaybackControllerFactory,
    required this.isPicking,
    required this.isUploading,
    required this.isPollingJob,
    required this.onPickFile,
    required this.onUpload,
    super.key,
  });

  final Uri? backendUrl;
  final String betaIdentity;
  final PickedAudioFile? selectedFile;
  final UploadSessionResult? uploadResult;
  final BackendJob? currentJob;
  final SessionAnalysis? analysis;
  final String? uploadError;
  final String? analysisError;
  final SegmentPlaybackControllerFactory segmentPlaybackControllerFactory;
  final bool isPicking;
  final bool isUploading;
  final bool isPollingJob;
  final VoidCallback onPickFile;
  final VoidCallback onUpload;

  @override
  Widget build(BuildContext context) {
    final result = uploadResult;
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      children: [
        UploadSessionCard(
          backendUrl: backendUrl,
          betaIdentity: betaIdentity,
          selectedFile: selectedFile,
          uploadResult: result,
          currentJob: currentJob,
          uploadError: uploadError,
          isPicking: isPicking,
          isUploading: isUploading,
          onPickFile: onPickFile,
          onUpload: onUpload,
        ),
        const SizedBox(height: 16),
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
        PickedUpCard(
          uploadResult: result,
          currentJob: currentJob,
          analysis: analysis,
          analysisError: analysisError,
          backendUrl: backendUrl,
          betaIdentity: betaIdentity,
          segmentPlaybackControllerFactory: segmentPlaybackControllerFactory,
          isPollingJob: isPollingJob,
        ),
        const SizedBox(height: 16),
        const WaveformCard(),
      ],
    );
  }
}

class UploadSessionCard extends StatelessWidget {
  const UploadSessionCard({
    required this.backendUrl,
    required this.betaIdentity,
    required this.selectedFile,
    required this.uploadResult,
    required this.currentJob,
    required this.uploadError,
    required this.isPicking,
    required this.isUploading,
    required this.onPickFile,
    required this.onUpload,
    super.key,
  });

  final Uri? backendUrl;
  final String betaIdentity;
  final PickedAudioFile? selectedFile;
  final UploadSessionResult? uploadResult;
  final BackendJob? currentJob;
  final String? uploadError;
  final bool isPicking;
  final bool isUploading;
  final VoidCallback onPickFile;
  final VoidCallback onUpload;

  @override
  Widget build(BuildContext context) {
    final file = selectedFile;
    final result = uploadResult;
    final job = currentJob ?? result?.job;
    final connected = backendUrl != null;
    return AppCard(
      emphasized: true,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                connected ? Icons.cloud_upload_outlined : Icons.cloud_off,
                color: connected ? const Color(0xFF6DA7A1) : Colors.white60,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      connected ? 'Upload karaoke audio' : 'Backend required',
                      style: const TextStyle(fontWeight: FontWeight.w900),
                    ),
                    Text(
                      connected
                          ? '${backendUrl!} • $betaIdentity'
                          : 'Connect from Home before uploading.',
                      style: const TextStyle(color: Colors.white60),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          if (file == null)
            const SessionMessage(
              icon: Icons.audio_file_outlined,
              title: 'No audio selected',
              detail: 'Choose an MP3, M4A, WAV, AAC, OGG, or FLAC file.',
            )
          else
            SessionMessage(
              icon: Icons.audio_file_outlined,
              title: file.name,
              detail: '${_formatBytes(file.sizeBytes)} • ${file.contentType}',
            ),
          if (uploadError != null) ...[
            const SizedBox(height: 8),
            SessionMessage(
              icon: Icons.error_outline,
              title: 'Upload failed',
              detail: uploadError!,
            ),
          ],
          if (result != null) ...[
            const SizedBox(height: 8),
            SessionMessage(
              icon: Icons.check_circle_outline,
              title: 'Uploaded',
              detail:
                  'Session ${_shortId(result.session.id)} • Job ${_shortId(result.job.id)} is ${job?.status ?? result.job.status}.',
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: isPicking || isUploading ? null : onPickFile,
                  icon: isPicking
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.folder_open),
                  label: Text(isPicking ? 'Choosing' : 'Choose file'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: FilledButton.icon(
                  onPressed:
                      !connected || file == null || isPicking || isUploading
                      ? null
                      : onUpload,
                  icon: isUploading
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.upload),
                  label: Text(isUploading ? 'Uploading' : 'Upload'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class PickedUpCard extends StatelessWidget {
  const PickedUpCard({
    required this.uploadResult,
    required this.currentJob,
    required this.analysis,
    required this.analysisError,
    required this.backendUrl,
    required this.betaIdentity,
    required this.segmentPlaybackControllerFactory,
    required this.isPollingJob,
    super.key,
  });

  final UploadSessionResult? uploadResult;
  final BackendJob? currentJob;
  final SessionAnalysis? analysis;
  final String? analysisError;
  final Uri? backendUrl;
  final String betaIdentity;
  final SegmentPlaybackControllerFactory segmentPlaybackControllerFactory;
  final bool isPollingJob;

  @override
  Widget build(BuildContext context) {
    final job = currentJob ?? uploadResult?.job;
    final analysis = this.analysis;
    return AppCard(
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
          if (uploadResult == null)
            const SessionMessage(
              icon: Icons.hourglass_empty,
              title: 'Waiting for upload',
              detail: 'Detected songs will appear after processing starts.',
            )
          else if (analysisError != null)
            SessionMessage(
              icon: Icons.error_outline,
              title: job?.status == 'failed'
                  ? 'Processing failed'
                  : 'Analysis unavailable',
              detail: analysisError!,
            )
          else if (analysis != null)
            AnalysisResults(
              analysis: analysis,
              audioUri: _sessionAudioUri(backendUrl, uploadResult!.session.id),
              betaIdentity: betaIdentity,
              segmentPlaybackControllerFactory:
                  segmentPlaybackControllerFactory,
            )
          else
            ProcessingStatusMessage(job: job, isPollingJob: isPollingJob),
        ],
      ),
    );
  }
}

class ProcessingStatusMessage extends StatelessWidget {
  const ProcessingStatusMessage({
    required this.job,
    required this.isPollingJob,
    super.key,
  });

  final BackendJob? job;
  final bool isPollingJob;

  @override
  Widget build(BuildContext context) {
    final status = job?.status ?? 'queued';
    return SessionMessage(
      icon: isPollingJob ? Icons.sync : Icons.hourglass_empty,
      title: _isTerminalJobStatus(status)
          ? 'Processing finished'
          : 'Processing',
      detail:
          'Job ${job == null ? 'pending' : _shortId(job!.id)} is $status. Results will appear here.',
    );
  }
}

class AnalysisResults extends StatefulWidget {
  const AnalysisResults({
    required this.analysis,
    required this.audioUri,
    required this.betaIdentity,
    required this.segmentPlaybackControllerFactory,
    super.key,
  });

  final SessionAnalysis analysis;
  final Uri? audioUri;
  final String betaIdentity;
  final SegmentPlaybackControllerFactory segmentPlaybackControllerFactory;

  @override
  State<AnalysisResults> createState() => _AnalysisResultsState();
}

class _AnalysisResultsState extends State<AnalysisResults> {
  late final SegmentPlaybackController _playbackController;
  DetectedSongSegment? _playingSegment;
  DetectedSongSegment? _loadingSegment;
  String? _playbackError;

  @override
  void initState() {
    super.initState();
    _playbackController = widget.segmentPlaybackControllerFactory();
  }

  @override
  void dispose() {
    unawaited(_playbackController.dispose());
    super.dispose();
  }

  Future<void> _togglePlayback(DetectedSongSegment segment) async {
    if (_playingSegment == segment) {
      await _playbackController.pause();
      if (!mounted) return;
      setState(() {
        _playingSegment = null;
        _playbackError = null;
      });
      return;
    }

    final audioUri = widget.audioUri;
    if (audioUri == null) {
      setState(
        () => _playbackError = 'Connect to the backend before playback.',
      );
      return;
    }

    setState(() {
      _loadingSegment = segment;
      _playbackError = null;
    });

    try {
      await _playbackController.playSegment(
        audioUri: audioUri,
        betaIdentity: widget.betaIdentity,
        segment: segment,
      );
      if (!mounted) return;
      setState(() => _playingSegment = segment);
    } catch (error) {
      if (!mounted) return;
      setState(() => _playbackError = error.toString());
    } finally {
      if (mounted) {
        setState(() => _loadingSegment = null);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final analysis = widget.analysis;
    final summary = analysis.summary;
    final segments = analysis.intervals.isNotEmpty
        ? analysis.intervals
        : analysis.weakCandidates;
    final weak =
        analysis.intervals.isEmpty && analysis.weakCandidates.isNotEmpty;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SessionMessage(
          icon: weak
              ? Icons.warning_amber_outlined
              : Icons.check_circle_outline,
          title: _analysisTitle(summary),
          detail: summary.message.isEmpty
              ? 'Provider: ${analysis.provider}'
              : summary.message,
        ),
        if (_playbackError != null)
          SessionMessage(
            icon: Icons.error_outline,
            title: 'Playback failed',
            detail: _playbackError!,
          ),
        if (segments.isEmpty)
          const SessionMessage(
            icon: Icons.search_off,
            title: 'No reliable match',
            detail:
                'The worker finished, but no song segment passed detection.',
          )
        else
          for (final segment in segments)
            PlaybackTrackRow(
              segment: segment,
              isPlaying: _playingSegment == segment,
              isLoading: _loadingSegment == segment,
              onToggle: () => _togglePlayback(segment),
            ),
      ],
    );
  }
}

class PlaybackTrackRow extends StatelessWidget {
  const PlaybackTrackRow({
    required this.segment,
    required this.isPlaying,
    required this.isLoading,
    required this.onToggle,
    super.key,
  });

  final DetectedSongSegment segment;
  final bool isPlaying;
  final bool isLoading;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        IconButton(
          onPressed: isLoading ? null : onToggle,
          icon: isLoading
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : Icon(isPlaying ? Icons.pause : Icons.play_arrow),
          tooltip: isPlaying ? 'Pause segment' : 'Play segment',
        ),
        Expanded(
          child: DetectedTrackRow(
            title: segment.artist.isEmpty
                ? segment.title
                : '${segment.title} - ${segment.artist}',
            time:
                '${_formatDuration(segment.startS)} - ${_formatDuration(segment.endS)}',
            badge: segment.isWeak ? 'weak' : segment.confidenceLabel,
            subtitle: _segmentSubtitle(segment),
          ),
        ),
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
    this.subtitle,
    super.key,
  });

  final String title;
  final String time;
  final String badge;
  final String? subtitle;

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
                if (subtitle != null)
                  Text(
                    subtitle!,
                    style: const TextStyle(color: Colors.white54),
                  ),
              ],
            ),
          ),
          Chip(label: Text(badge)),
        ],
      ),
    );
  }
}

bool _isTerminalJobStatus(String status) {
  return status == 'completed' || status == 'failed' || status == 'cancelled';
}

String _analysisTitle(AnalysisSummary summary) {
  if (summary.acceptedIntervalCount > 0) {
    return '${summary.acceptedIntervalCount} detected song segment${summary.acceptedIntervalCount == 1 ? '' : 's'}';
  }
  if (summary.weakCandidateCount > 0) {
    return '${summary.weakCandidateCount} weak clue${summary.weakCandidateCount == 1 ? '' : 's'} found';
  }
  return 'No detected songs';
}

String _segmentSubtitle(DetectedSongSegment segment) {
  final details = <String>[];
  if (segment.confidenceScore != null) {
    details.add('confidence ${segment.confidenceScore!.toStringAsFixed(1)}');
  }
  if (segment.reason != null && segment.reason!.isNotEmpty) {
    details.add(segment.reason!);
  }
  return details.join(' • ');
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
