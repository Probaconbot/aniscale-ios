import 'package:aniscale/models/enhancement.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('image history entry survives JSON round trip', () {
    final createdAt = DateTime.utc(2026, 8, 27, 20, 30);
    final enhancement = Enhancement(
      originalPath: '/local/original.png',
      outputPath: '/local/output.png',
      scale: 2,
      createdAt: createdAt,
      originalWidth: 640,
      originalHeight: 480,
      engine: 'AniScale Fusion',
      actualOutputWidth: 1280,
      actualOutputHeight: 960,
    );

    final restored = Enhancement.fromJson(enhancement.toJson());

    expect(restored.outputPath, enhancement.outputPath);
    expect(restored.outputWidth, 1280);
    expect(restored.outputHeight, 960);
    expect(restored.createdAt, createdAt);
    expect(restored.isVideo, isFalse);
  });

  test('video history entry preserves duration and media type', () {
    final enhancement = Enhancement(
      originalPath: '/local/original.mp4',
      outputPath: '/local/output.mp4',
      scale: 4,
      createdAt: DateTime.utc(2026, 8, 27),
      originalWidth: 480,
      originalHeight: 270,
      engine: 'AniScale Turbo',
      actualOutputWidth: 1920,
      actualOutputHeight: 1080,
      isVideo: true,
      durationSeconds: 12.5,
    );

    final restored = Enhancement.fromJson(enhancement.toJson());

    expect(restored.isVideo, isTrue);
    expect(restored.durationSeconds, 12.5);
    expect(restored.outputWidth, 1920);
  });
}
