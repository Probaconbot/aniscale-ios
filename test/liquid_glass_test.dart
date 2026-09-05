import 'package:aniscale/widgets/liquid_glass_surface.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('glass follows touch without rebuilding content or moving it', (
    tester,
  ) async {
    var builds = 0;
    var taps = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Center(
            child: SizedBox(
              width: 300,
              height: 240,
              child: LiquidGlassSurface(
                child: Builder(
                  builder: (_) {
                    builds++;
                    return Center(
                      child: TextButton(
                        onPressed: () => taps++,
                        child: const Text('Upscale'),
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      ),
    );
    final button = find.text('Upscale');
    final original = tester.getRect(button);
    final before = builds;
    final gesture = await tester.startGesture(tester.getCenter(button));
    await tester.pump(const Duration(milliseconds: 100));
    await gesture.moveBy(const Offset(2, 2));
    await tester.pump();
    expect(builds, before);
    expect(tester.getRect(button), original);
    await gesture.up();
    await tester.pumpAndSettle();
    expect(taps, 1);
    expect(builds, before);
    expect(tester.binding.transientCallbackCount, 0);
    expect(tester.takeException(), isNull);
  });

  testWidgets('reduced motion skips blur and touch animation', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: MediaQuery(
          data: MediaQueryData(disableAnimations: true),
          child: Center(
            child: SizedBox(
              width: 280,
              height: 72,
              child: LiquidGlassSurface(child: Text('History')),
            ),
          ),
        ),
      ),
    );
    expect(find.byType(BackdropFilter), findsNothing);
    final gesture = await tester.startGesture(
      tester.getCenter(find.byType(LiquidGlassSurface)),
    );
    await gesture.moveBy(const Offset(20, 10));
    await tester.pump();
    expect(tester.binding.transientCallbackCount, 0);
    await gesture.up();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });

  testWidgets(
    'cancelled touches settle and a second finger cannot steal light',
    (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Center(
            child: SizedBox(
              width: 280,
              height: 100,
              child: LiquidGlassSurface(child: Text('Assistant')),
            ),
          ),
        ),
      );
      final center = tester.getCenter(find.byType(LiquidGlassSurface));
      final first = await tester.startGesture(center, pointer: 1);
      final second = await tester.startGesture(
        center + const Offset(50, 0),
        pointer: 2,
      );
      await tester.pump(const Duration(milliseconds: 100));
      await second.up();
      await first.cancel();
      await tester.pumpAndSettle();
      expect(tester.binding.transientCallbackCount, 0);
      expect(tester.takeException(), isNull);
    },
  );
}
