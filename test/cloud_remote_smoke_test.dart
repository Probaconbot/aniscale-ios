import 'dart:io';

import 'package:aniscale/services/cloud_video_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/foundation.dart';

/// Opt-in only: uses the same Dart transport as Android/iOS against the private
/// GPU Space. Credentials are runtime environment values, not dart-defines.
void main() {
  final token = Platform.environment['ANISCALE_HF_TOKEN'];
  final source = Platform.environment['ANISCALE_SMOKE_VIDEO'];
  test(
    'real private GPU upload → AnimeSR → download',
    () async {
      final service = CloudVideoService(
        endpoint: CloudVideoSession.endpoint,
        token: token!,
      );
      final result = await service.run(
        path: source!,
        scale: 2,
        detail: 'natural',
        codec: 'h264',
        outputDirectory: await Directory.systemTemp.createTemp(
          'aniscale-gpu-result-',
        ),
        onProgress: (p) {
          debugPrint('${(p.fraction * 100).round()}% ${p.stage}');
        },
      );
      expect(result.originalWidth, 64);
      expect(result.originalHeight, 48);
      expect(result.outputWidth, 128);
      expect(result.outputHeight, 96);
      expect(await File(result.path).length(), greaterThan(1000));
      debugPrint('Real GPU result: ${result.path}');
      debugPrint('Server benchmark: ${result.benchmark}');
    },
    skip: token == null || source == null,
    timeout: const Timeout(Duration(minutes: 8)),
  );
}
