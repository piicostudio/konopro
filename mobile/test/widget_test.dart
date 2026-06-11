import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:konopro/main.dart';

void main() {
  testWidgets('home shows the practice dashboard', (tester) async {
    await tester.pumpWidget(const KonoProApp());

    expect(find.text('KonoPro'), findsOneWidget);
    expect(find.text('7 songs found'), findsOneWidget);
    expect(find.text('Most practiced'), findsOneWidget);
    expect(find.text('Every Moment'), findsOneWidget);
  });

  testWidgets('song card opens playback history', (tester) async {
    await tester.pumpWidget(const KonoProApp());

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
