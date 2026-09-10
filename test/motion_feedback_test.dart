import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:aniscale/widgets/motion_feedback.dart';

void main() {
  testWidgets(
    'Press control activates once and disabled control stays disabled',
    (tester) async {
      var taps = 0;
      Widget app(VoidCallback? action) => MaterialApp(
        home: Scaffold(
          body: MotionTap(
            onTap: action,
            child: const SizedBox(width: 100, height: 60, child: Text('Tap')),
          ),
        ),
      );
      await tester.pumpWidget(app(() => taps++));
      await tester.tap(find.text('Tap'));
      await tester.pumpAndSettle();
      expect(taps, 1);
      await tester.pumpWidget(app(null));
      await tester.tap(find.text('Tap'));
      await tester.pumpAndSettle();
      expect(taps, 1);
    },
  );

  testWidgets('Tab transition and reduced motion retain child state', (
    tester,
  ) async {
    final fieldKey = GlobalKey();
    Widget app(int index, bool reduce) => MaterialApp(
      home: MediaQuery(
        data: MediaQueryData(disableAnimations: reduce),
        child: Scaffold(
          body: TabReveal(
            index: index,
            child: TextField(key: fieldKey),
          ),
        ),
      ),
    );
    await tester.pumpWidget(app(0, false));
    await tester.enterText(find.byType(TextField), 'keep my draft');
    await tester.pumpWidget(app(1, false));
    await tester.pumpAndSettle();
    await tester.pumpWidget(app(1, true));
    await tester.pumpAndSettle();
    expect(find.text('keep my draft'), findsOneWidget);
  });
}
