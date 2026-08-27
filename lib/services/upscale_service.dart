import 'dart:convert';
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
    this.denoise = .2,
    this.sharpness = .2,
    this.detail = .5,
    this.colorFidelity = .9,
  });

  final String path;
  final int scale;
  final bool preserveTransparency;
  final double denoise;
  final double sharpness;
  final double detail;
  final double colorFidelity;
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
  required bool efficient,
}) async {
  if (!Platform.isIOS) {
    throw UnsupportedError(
      'Local video upscaling is currently available on iOS.',
    );
  }
  final response = await _upscaleChannel.invokeMapMethod<String, dynamic>(
    'upscaleVideo',
    {'path': path, 'scale': scale, 'efficient': efficient},
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
    engine: response['engine'] as String? ?? 'Real-ESRGAN Core ML',
  );
}

Future<UpscaleResult> _upscaleWithCoreML(UpscaleRequest request) async {
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
    outputWidth: resized.width,
    outputHeight: resized.height,
    engine: 'High-quality cubic fallback',
  );
}
