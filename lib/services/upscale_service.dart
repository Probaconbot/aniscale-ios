import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

class UpscaleRequest {
  const UpscaleRequest({required this.path, required this.scale});

  final String path;
  final int scale;
}

class UpscaleResult {
  const UpscaleResult({
    required this.path,
    required this.originalWidth,
    required this.originalHeight,
  });

  final String path;
  final int originalWidth;
  final int originalHeight;
}

Future<UpscaleResult> upscaleLocally(UpscaleRequest request) {
  return compute(_resizeImage, request);
}

Future<UpscaleResult> _resizeImage(UpscaleRequest request) async {
  final Uint8List bytes = await File(request.path).readAsBytes();
  final img.Image? source = img.decodeImage(bytes);
  if (source == null) {
    throw const FormatException('This image format could not be decoded.');
  }

  final resized = img.copyResize(
    source,
    width: source.width * request.scale,
    height: source.height * request.scale,
    interpolation: img.Interpolation.cubic,
  );
  final stamp = DateTime.now().millisecondsSinceEpoch;
  final output = File('${Directory.systemTemp.path}/aniscale_$stamp.png');
  await output.writeAsBytes(img.encodePng(resized, level: 6), flush: true);

  return UpscaleResult(
    path: output.path,
    originalWidth: source.width,
    originalHeight: source.height,
  );
}
