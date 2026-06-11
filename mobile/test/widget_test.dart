import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:konopro/main.dart';

void main() {
  testWidgets('home shows the practice dashboard', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    expect(find.text('KonoPro'), findsOneWidget);
    expect(find.text('Backend URL'), findsOneWidget);
    expect(find.text('Test connection'), findsOneWidget);
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
      ),
    );

    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pump();

    expect(find.text('Connected to local'), findsOneWidget);
    expect(find.text('Backend status: ok'), findsOneWidget);
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
