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
}
