import 'package:flutter_test/flutter_test.dart';
import 'package:konopro/main.dart';

void main() {
  test(
    'default backend health client reaches configured live backend',
    () async {
      const backendUrl = String.fromEnvironment('KONOPRO_LIVE_BACKEND_URL');
      if (backendUrl.isEmpty) {
        return;
      }

      final health = await defaultBackendHealthCheck(Uri.parse(backendUrl));

      expect(health.status, 'ok');
    },
  );

  test('default session list client reaches configured live backend', () async {
    const backendUrl = String.fromEnvironment('KONOPRO_LIVE_BACKEND_URL');
    if (backendUrl.isEmpty) {
      return;
    }

    final sessions = await defaultSessionListClient(
      Uri.parse(backendUrl),
      'peter-demo',
    );

    expect(sessions, isA<List<BackendSession>>());
  });

  test('default upload client creates live backend session and job', () async {
    const backendUrl = String.fromEnvironment('KONOPRO_LIVE_BACKEND_URL');
    if (backendUrl.isEmpty) {
      return;
    }

    final result = await defaultUploadSessionClient(
      Uri.parse(backendUrl),
      'peter-demo',
      const PickedAudioFile(
        name: 'flutter-live-upload.wav',
        bytes: [1, 2, 3, 4, 5],
        contentType: 'audio/wav',
      ),
    );

    expect(result.session.originalFilename, 'flutter-live-upload.wav');
    expect(result.session.status, 'queued');
    expect(result.job.sessionId, result.session.id);
    expect(result.job.status, 'queued');
  });

  test('default job client reaches configured live backend', () async {
    const backendUrl = String.fromEnvironment('KONOPRO_LIVE_BACKEND_URL');
    if (backendUrl.isEmpty) {
      return;
    }

    final upload = await defaultUploadSessionClient(
      Uri.parse(backendUrl),
      'peter-demo',
      const PickedAudioFile(
        name: 'flutter-live-job.wav',
        bytes: [1, 2, 3, 4, 5],
        contentType: 'audio/wav',
      ),
    );
    final job = await defaultJobStatusClient(
      Uri.parse(backendUrl),
      'peter-demo',
      upload.job.id,
    );

    expect(job.id, upload.job.id);
    expect(job.sessionId, upload.session.id);
  });
}
