import 'package:aniscale/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('AniScale home screen renders', (tester) async {
    await tester.pumpWidget(const AniScaleApp());

    expect(find.text('AniScale'), findsOneWidget);
    expect(find.text('Create a cleaner frame.'), findsOneWidget);
    expect(find.text('@Search, enhance, upscale, or ask AI…'), findsOneWidget);
    expect(find.text('Video'), findsOneWidget);
    expect(find.text('Image'), findsOneWidget);
    expect(find.text('Upscale'), findsOneWidget);
    expect(find.text('Ask AI'), findsOneWidget);
    expect(find.text('Choose image'), findsOneWidget);
  });
}
