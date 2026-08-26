import 'package:aniscale/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('AniScale home screen renders', (tester) async {
    await tester.pumpWidget(const AniScaleApp());

    expect(find.text('AniScale'), findsOneWidget);
    expect(find.text('Enhance every detail.'), findsOneWidget);
    expect(find.text('Select Image'), findsOneWidget);
  });
}
