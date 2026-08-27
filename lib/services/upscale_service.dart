import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

const _upscaleChannel = MethodChannel('app.aniscale/upscaler');
const _progressChannel = EventChannel('app.aniscale/upscaler_progress');

Stream<double> get upscaleProgress => _progressChannel
    .receiveBroadcastStream()
    .where((event) => event is num)
    .map((event) => (event as num).toDouble());

class UpscaleRequest {
  const UpscaleRequest({
    required this.path,
    required this.scale,
    this.preserveTransparency = true,
  });

  final String path;
  final int scale;
  final bool preserveTransparency;
}

class UpscaleResult {
  const UpscaleResult({
    required this.path,
    required this.originalWidth,
    required this.originalHeight,
    required this.engine,
  });

  final String path;
  final int originalWidth;
  final int originalHeight;
  final String engine;
}

class VideoUpscaleResult {
  const VideoUpscaleResult({
    required this.path,
    required this.outputWidth,
    required this.outputHeight,
    required this.durationSeconds,
    required this.engine,
  });

  final String path;
  final int outputWidth;
  final int outputHeight;
  final double durationSeconds;
  final String engine;
}

Future<UpscaleResult> upscaleLocally(UpscaleRequest request) {
  if (Platform.isIOS) return _upscaleWithCoreML(request);
  return compute(_resizeImage, request);
}

Future<void> cancelUpscale() async {
  if (Platform.isIOS) await _upscaleChannel.invokeMethod<void>('cancel');
}

Future<VideoUpscaleResult> upscaleVideoLocally({
  required String path,
  required int scale,
}) async {
  if (!Platform.isIOS) {
    throw UnsupportedError(
      'Local video upscaling is currently available on iOS.',
    );
  }
  final response = await _upscaleChannel.invokeMapMethod<String, dynamic>(
    'upscaleVideo',
    {'path': path, 'scale': scale},
  );
  if (response == null) {
    throw PlatformException(
      code: 'empty_result',
      message: 'The on-device video engine returned no video.',
    );
  }
  return VideoUpscaleResult(
    path: response['path'] as String,
    outputWidth: response['outputWidth'] as int,
    outputHeight: response['outputHeight'] as int,
    durationSeconds: (response['durationSeconds'] as num).toDouble(),
    engine: response['engine'] as String? ?? 'Core Image GPU',
  );
}

Future<UpscaleResult> _upscaleWithCoreML(UpscaleRequest request) async {
  final response = await _upscaleChannel.invokeMapMethod<String, dynamic>(
    'upscaleImage',
    {
      'path': request.path,
      'scale': request.scale,
      'preserveTransparency': request.preserveTransparency,
    },
  );
  if (response == null) {
    throw PlatformException(
      code: 'empty_result',
      message: 'The on-device AI engine returned no image.',
    );
  }
  return UpscaleResult(
    path: response['path'] as String,
    originalWidth: response['originalWidth'] as int,
    originalHeight: response['originalHeight'] as int,
    engine: response['engine'] as String? ?? 'Core ML',
  );
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
    engine: 'High-quality cubic fallback',
  );
}
