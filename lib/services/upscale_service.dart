import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

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
    this.denoise = .2,
    this.sharpness = .2,
    this.detail = .5,
    this.colorFidelity = .9,
    this.outputFormat = 'automatic',
    this.tileSize = 256,
    this.preserveMetadata = true,
    this.engine = 'fusion',
    this.performance = 0,
  });

  final String path;
  final int scale;
  final bool preserveTransparency;
  final double denoise;
  final double sharpness;
  final double detail;
  final double colorFidelity;
  final String outputFormat;
  final int tileSize;
  final bool preserveMetadata;
  final String engine;
  final int performance;
}

class UpscaleResult {
  const UpscaleResult({
    required this.path,
    required this.originalWidth,
    required this.originalHeight,
    required this.outputWidth,
    required this.outputHeight,
    required this.engine,
  });

  final String path;
  final int originalWidth;
  final int originalHeight;
  final int outputWidth;
  final int outputHeight;
  final String engine;
}

class VideoUpscaleResult {
  const VideoUpscaleResult({
    required this.path,
    required this.originalWidth,
    required this.originalHeight,
    required this.outputWidth,
    required this.outputHeight,
    required this.durationSeconds,
    required this.engine,
    this.benchmark = const {},
  });

  final String path;
  final int originalWidth;
  final int originalHeight;
  final int outputWidth;
  final int outputHeight;
  final double durationSeconds;
  final String engine;
  final Map<String, dynamic> benchmark;
}

Future<UpscaleResult> upscaleLocally(UpscaleRequest request) {
  if (Platform.isIOS || Platform.isAndroid) {
    return _upscaleWithNativeAI(request);
  }
  return compute(_resizeImage, request);
}

Future<void> cancelUpscale() async {
  if (Platform.isIOS || Platform.isAndroid) {
    await _upscaleChannel.invokeMethod<void>('cancel');
  }
}

Future<VideoUpscaleResult> upscaleVideoLocally({
  required String path,
  required int scale,
  double? targetScale,
  required bool efficient,
  required int tileSize,
  required String engine,
  String content = 'auto',
  String detailMode = 'natural',
  String codec = 'hevc',
}) async {
  if (!Platform.isIOS && !Platform.isAndroid) {
    throw UnsupportedError(
      'Local video upscaling is available on iOS and Android.',
    );
  }
  final response = await _upscaleChannel.invokeMapMethod<String, dynamic>(
    'upscaleVideo',
    {
      'path': path,
      'scale': scale,
      'targetScale': targetScale ?? scale.toDouble(),
      'efficient': efficient,
      'tileSize': tileSize,
      'engine': engine,
      'content': content,
      'detailMode': detailMode,
      'codec': codec,
    },
  );
  if (response == null) {
    throw PlatformException(
      code: 'empty_result',
      message: 'The on-device video engine returned no video.',
    );
  }
  return VideoUpscaleResult(
    path: response['path'] as String,
    originalWidth:
        response['originalWidth'] as int? ??
        ((response['outputWidth'] as int) / (targetScale ?? scale)).round(),
    originalHeight:
        response['originalHeight'] as int? ??
        ((response['outputHeight'] as int) / (targetScale ?? scale)).round(),
    outputWidth: response['outputWidth'] as int,
    outputHeight: response['outputHeight'] as int,
    durationSeconds: (response['durationSeconds'] as num).toDouble(),
    engine: response['engine'] as String? ?? 'AniScale Fusion',
    benchmark: Map<String, dynamic>.from(
      response['benchmark'] as Map? ?? const {},
    ),
  );
}

Future<UpscaleResult> _upscaleWithNativeAI(UpscaleRequest request) async {
  final response = await _upscaleChannel.invokeMapMethod<String, dynamic>(
    'upscaleImage',
    {
      'path': request.path,
      'scale': request.scale,
      'preserveTransparency': request.preserveTransparency,
      'denoise': request.denoise,
      'sharpness': request.sharpness,
      'detail': request.detail,
      'colorFidelity': request.colorFidelity,
      'outputFormat': request.outputFormat,
      'tileSize': request.tileSize,
      'preserveMetadata': request.preserveMetadata,
      'engine': request.engine,
      'performance': request.performance,
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
    outputWidth:
        response['outputWidth'] as int? ??
        (response['originalWidth'] as int) * request.scale,
    outputHeight:
        response['outputHeight'] as int? ??
        (response['originalHeight'] as int) * request.scale,
    engine: response['engine'] as String? ?? 'Core ML',
  );
}

/// Produces a small JPEG data URL for vision analysis without decoding the
/// full camera image on the UI isolate.
Future<String> prepareVisionImage(String path) =>
    compute(_prepareVisionImage, path);

String _prepareVisionImage(String path) {
  final bytes = File(path).readAsBytesSync();
  var image = img.decodeImage(bytes);
  if (image == null) {
    throw const FormatException('This image format could not be decoded.');
  }
  image = img.bakeOrientation(image);
  const maxEdge = 1024;
  if (image.width > maxEdge || image.height > maxEdge) {
    if (image.width >= image.height) {
      image = img.copyResize(image, width: maxEdge);
    } else {
      image = img.copyResize(image, height: maxEdge);
    }
  }
  final jpeg = img.encodeJpg(image, quality: 82);
  return 'data:image/jpeg;base64,${base64Encode(jpeg)}';
}

Future<UpscaleResult> _resizeImage(UpscaleRequest request) async {
  final Uint8List bytes = await File(request.path).readAsBytes();
  var source = img.decodeImage(bytes);
  if (source == null) {
    throw const FormatException('This image format could not be decoded.');
  }
  source = img.bakeOrientation(source);
  final originalWidth = source.width;
  final originalHeight = source.height;
  const safeOutputPixels = 18000000;
  final requestedPixels =
      source.width * request.scale * source.height * request.scale;
  if (requestedPixels > safeOutputPixels) {
    final ratio = (safeOutputPixels / requestedPixels).clamp(0, 1).toDouble();
    final linearRatio = ratio <= 0 ? 1.0 : math.sqrt(ratio);
    source = img.copyResize(
      source,
      width: (source.width * linearRatio).round().clamp(1, source.width),
      height: (source.height * linearRatio).round().clamp(1, source.height),
    );
  }

  final resized = img.copyResize(
    source,
    width: source.width * request.scale,
    height: source.height * request.scale,
    interpolation: img.Interpolation.cubic,
  );
  final stamp = DateTime.now().millisecondsSinceEpoch;
  final usePng =
      request.outputFormat == 'png' ||
      (request.outputFormat == 'automatic' && source.numChannels == 4);
  final output = File(
    '${Directory.systemTemp.path}/aniscale_$stamp.${usePng ? 'png' : 'jpg'}',
  );
  await output.writeAsBytes(
    usePng
        ? img.encodePng(resized, level: 6)
        : img.encodeJpg(resized, quality: 96),
    flush: true,
  );

  return UpscaleResult(
    path: output.path,
    originalWidth: originalWidth,
    originalHeight: originalHeight,
    outputWidth: resized.width,
    outputHeight: resized.height,
    engine: 'High-quality mobile resampler',
  );
}
