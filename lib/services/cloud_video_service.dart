import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'upscale_service.dart';

/// Private Space credentials are session-only. Never store a token in the APK/IPA.
class CloudVideoSession {
  static String token = '';
  static final endpoint = Uri.parse(
    'https://lushus6pg-aniscale-video.hf.space',
  );
}

class CloudVideoException implements Exception {
  const CloudVideoException(this.message);
  final String message;
  @override
  String toString() => message;
}

class CloudProgress {
  const CloudProgress(this.fraction, this.stage);
  final double fraction;
  final String stage;
}

/// Streaming Gradio 5.49 call/SSE protocol. There is deliberately no native
/// inference fallback: selecting cloud must not start work on the phone GPU/CPU.
class CloudVideoService {
  CloudVideoService({required this.endpoint, required String token})
    : _token = token.trim() {
    if (endpoint.scheme != 'https' &&
        !(endpoint.scheme == 'http' && endpoint.host == '127.0.0.1')) {
      throw const CloudVideoException('Cloud processing requires HTTPS.');
    }
    if (endpoint.userInfo.isNotEmpty ||
        endpoint.hasQuery ||
        endpoint.hasFragment) {
      throw const CloudVideoException('Invalid cloud address.');
    }
  }

  final Uri endpoint;
  final String _token;
  final HttpClient _client = HttpClient()
    ..connectionTimeout = const Duration(seconds: 30);
  final String _session = List.generate(
    16,
    (_) => Random.secure().nextInt(256).toRadixString(16).padLeft(2, '0'),
  ).join();
  String? _event;
  int? _function;
  bool _cancelled = false;
  bool _finished = false;

  void _checkCancelled() {
    if (_cancelled) throw const CloudVideoException('Cloud job cancelled.');
  }

  Future<HttpClientRequest> _request(
    String method,
    Uri uri, {
    HttpClient? client,
  }) async {
    if (uri.origin != endpoint.origin || uri.userInfo.isNotEmpty) {
      throw const CloudVideoException(
        'The server returned an unsafe download address.',
      );
    }
    final request = await (client ?? _client).openUrl(method, uri);
    request.followRedirects = false; // Never forward the token to a redirect.
    request.headers.set(HttpHeaders.authorizationHeader, 'Bearer $_token');
    return request;
  }

  Future<dynamic> _json(HttpClientResponse response) async {
    final bytes = <int>[];
    await for (final chunk in response.timeout(const Duration(seconds: 60))) {
      if (bytes.length + chunk.length > 4 * 1024 * 1024) {
        throw const CloudVideoException('Unexpectedly large cloud response.');
      }
      bytes.addAll(chunk);
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final status = response.statusCode;
      throw CloudVideoException(switch (status) {
        401 || 403 || 404 => 'Private Space access failed. Enter a Hugging Face read token with access to Lushus6pg/aniscale-video.',
        429 => 'Hugging Face GPU quota or queue limit reached. Try after your quota resets.',
        502 || 503 || 504 => 'The Hugging Face Space is starting or unavailable. Try again shortly.',
        _ =>
          'Hugging Face request failed (HTTP $status). No local processing was started.',
      });
    }
    try {
      return jsonDecode(utf8.decode(bytes));
    } catch (_) {
      throw const CloudVideoException(
        'The private Space returned an invalid response.',
      );
    }
  }

  Future<dynamic> _post(String route, Object body) async {
    final request = await _request('POST', endpoint.resolve(route));
    request.headers.contentType = ContentType.json;
    request.write(jsonEncode(body));
    return _json(await request.close());
  }

  Future<VideoUpscaleResult> run({
    required String path,
    required int scale,
    required String detail,
    required String codec,
    required Directory outputDirectory,
    required void Function(CloudProgress) onProgress,
  }) async {
    File? pending;
    try {
      if (!_token.startsWith('hf_') || _token.contains(RegExp(r'\s'))) {
        throw const CloudVideoException(
          'Enter your Hugging Face read token first.',
        );
      }
      if (![2, 4].contains(scale) ||
          !['natural', 'detailed', 'sharp'].contains(detail) ||
          !['hevc', 'h264'].contains(codec)) {
        throw const CloudVideoException('Unsupported cloud video settings.');
      }
      final file = File(path);
      final length = await file.length();
      if (length == 0 || length > 100 * 1024 * 1024) {
        throw const CloudVideoException(
          'Free GPU mode accepts videos up to 100 MB.',
        );
      }
      onProgress(
        const CloudProgress(.01, 'Connecting to private Hugging Face Space…'),
      );
      final configRequest = await _request('GET', endpoint.resolve('/config'));
      final config = await _json(await configRequest.close()) as Map;
      final functions = (config['dependencies'] as List).cast<Map>();
      final function = functions
          .where((f) => f['api_name'] == 'upscale')
          .firstOrNull;
      if (function == null || function['id'] is! int) {
        throw const CloudVideoException(
          'The Space has not deployed the AniScale video API yet.',
        );
      }
      _function = function['id'] as int;
      _checkCancelled();
      final boundary = 'aniscale_$_session';
      final header = utf8.encode(
        '--$boundary\r\nContent-Disposition: form-data; name="files"; filename="input.mp4"\r\nContent-Type: application/octet-stream\r\n\r\n',
      );
      final footer = utf8.encode('\r\n--$boundary--\r\n');
      final upload = await _request(
        'POST',
        endpoint.resolve('/gradio_api/upload'),
      );
      upload.headers.set(
        HttpHeaders.contentTypeHeader,
        'multipart/form-data; boundary=$boundary',
      );
      upload.contentLength = header.length + length + footer.length;
      upload.add(header);
      var sent = 0;
      await upload.addStream(
        file.openRead().map((chunk) {
          _checkCancelled();
          sent += chunk.length;
          onProgress(
            CloudProgress(
              .02 + .12 * sent / length,
              'Uploading to private Hugging Face Space…',
            ),
          );
          return chunk;
        }),
      );
      upload.add(footer);
      final uploaded = await _json(await upload.close()) as List;
      if (uploaded.length != 1 || uploaded.first is! String) {
        throw const CloudVideoException('The upload did not return a video.');
      }
      _checkCancelled();
      final job = await _post('/gradio_api/call/upscale', {
        // Gradio's /call SSE reader indexes messages by event_id. Let the
        // server use that event ID as the session too (do not supply our own).
        'data': [
          {
            'path': uploaded.first,
            'orig_name': 'input.mp4',
            'meta': {'_type': 'gradio.FileData'},
          },
          scale,
          detail,
          codec,
        ],
      }) as Map;
      _event = job['event_id'] as String?;
      if (_event == null || !RegExp(r'^[a-zA-Z0-9_-]+$').hasMatch(_event!)) {
        throw const CloudVideoException(
          'The Space did not return a valid job ID.',
        );
      }
      _checkCancelled();
      onProgress(const CloudProgress(.15, 'Waiting for a Hugging Face GPU…'));
      final request = await _request(
        'GET',
        endpoint.resolve('/gradio_api/call/upscale/$_event'),
      );
      final response = await request.close();
      if (response.statusCode != 200) await _json(response);
      Map? resultFile;
      Map? metadata;
      var event = '';
      var data = '';
      var complete = false;
      await for (final line
          in response
              .timeout(const Duration(seconds: 90))
              .transform(utf8.decoder)
              .transform(const LineSplitter())) {
        _checkCancelled();
        if (line.startsWith('event:')) event = line.substring(6).trim();
        if (line.startsWith('data:')) data += line.substring(5).trim();
        if (line.isNotEmpty) continue;
        if (event == 'error') {
          throw const CloudVideoException(
            'Hugging Face could not run the GPU job. Check the Space logs or your daily GPU quota; nothing ran on your phone.',
          );
        }
        if ((event == 'generating' || event == 'complete') && data.isNotEmpty) {
          final values = jsonDecode(data);
          if (values is List && values.length >= 2 && values[1] is Map) {
            metadata = values[1] as Map;
            if (metadata['error'] is String) {
              final safeError = (metadata['error'] as String).replaceAll(
                _token,
                '[redacted]',
              );
              throw CloudVideoException(
                safeError.length > 400
                    ? safeError.substring(0, 400)
                    : safeError,
              );
            }
            final fraction = metadata['progress'];
            onProgress(
              CloudProgress(
                .15 +
                    .70 *
                        (fraction is num ? fraction.toDouble().clamp(0, 1) : 0),
                metadata['stage'] as String? ??
                    'Processing on Hugging Face GPU…',
              ),
            );
            if (values[0] is Map) resultFile = values[0] as Map;
          }
          if (event == 'complete') {
            complete = true;
            break;
          }
        }
        event = '';
        data = '';
      }
      if (!complete || resultFile == null || metadata == null) {
        throw const CloudVideoException(
          'The cloud job ended without a finished video.',
        );
      }
      final download = Uri.tryParse(resultFile['url'] as String? ?? '');
      if (download == null ||
          !download.hasAuthority ||
          download.origin != endpoint.origin ||
          !download.path.startsWith('/gradio_api/file=')) {
        throw const CloudVideoException(
          'The Space returned an invalid video download.',
        );
      }
      for (final key in [
        'originalWidth',
        'originalHeight',
        'outputWidth',
        'outputHeight',
      ]) {
        if (metadata[key] is! int || (metadata[key] as int) <= 0) {
          throw const CloudVideoException(
            'The result is missing video dimensions.',
          );
        }
      }
      if (metadata['durationSeconds'] is! num) {
        throw const CloudVideoException('The result is missing its duration.');
      }
      await outputDirectory.create(recursive: true);
      pending = File('${outputDirectory.path}/cloud_$_session.mp4.part');
      final downloadRequest = await _request('GET', download);
      final output = await downloadRequest.close();
      if (output.statusCode != 200) await _json(output);
      final sink = pending.openWrite();
      var received = 0;
      try {
        await for (final chunk in output.timeout(const Duration(seconds: 90))) {
          _checkCancelled();
          received += chunk.length;
          if (received > 1024 * 1024 * 1024) {
            throw const CloudVideoException(
              'The output exceeds the 1 GB download limit.',
            );
          }
          sink.add(chunk);
          // Flush periodically to bound the disk writer's queue on slow devices.
          if (received ~/ (1024 * 1024) !=
              (received - chunk.length) ~/ (1024 * 1024)) {
            await sink.flush();
          }
          onProgress(
            CloudProgress(
              .86 +
                  .13 *
                      (output.contentLength > 0
                          ? (received / output.contentLength).clamp(0, 1)
                          : 0),
              'Downloading restored video…',
            ),
          );
        }
        await sink.flush();
      } finally {
        await sink.close();
      }
      if (received < 12 ||
          (output.contentLength >= 0 && received != output.contentLength)) {
        throw const CloudVideoException('Video download was incomplete.');
      }
      final handle = await pending.open();
      final signature = await handle.read(12);
      await handle.close();
      if (ascii.decode(signature.sublist(4, 8), allowInvalid: true) != 'ftyp') {
        throw const CloudVideoException(
          'The server did not return an MP4 video.',
        );
      }
      _checkCancelled();
      final finished = await pending.rename(
        '${outputDirectory.path}/cloud_$_session.mp4',
      );
      pending = null;
      _finished = true;
      onProgress(const CloudProgress(1, 'Video ready'));
      return VideoUpscaleResult(
        path: finished.path,
        originalWidth: metadata['originalWidth'] as int,
        originalHeight: metadata['originalHeight'] as int,
        outputWidth: metadata['outputWidth'] as int,
        outputHeight: metadata['outputHeight'] as int,
        durationSeconds: (metadata['durationSeconds'] as num).toDouble(),
        engine: 'AniUltraAnime • Hugging Face GPU',
        benchmark: {
          'processingLocation': 'huggingface',
          'serverSeconds': metadata['seconds'],
          'gpu': metadata['gpu'],
        },
      );
    } on TimeoutException {
      throw const CloudVideoException(
        'Cloud connection timed out. No local processing was started.',
      );
    } on SocketException {
      throw const CloudVideoException(
        'Cloud connection lost. Check your internet connection.',
      );
    } finally {
      if (!_finished) await cancel();
      _client.close(force: true);
      if (pending != null && await pending.exists()) await pending.delete();
    }
  }

  Future<void> cancel() async {
    if (_finished) return;
    _cancelled = true;
    _client.close(force: true);
    if (_event == null || _function == null) return;
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 5);
    try {
      final request = await _request(
        'POST',
        endpoint.resolve('/gradio_api/cancel'),
        client: client,
      );
      request.headers.contentType = ContentType.json;
      request.write(
        jsonEncode({
          'session_hash': _event,
          'fn_index': _function,
          'event_id': _event,
        }),
      );
      await (await request.close()).drain<void>().timeout(
        const Duration(seconds: 5),
      );
    } catch (_) {
      // Disconnection also closes the SSE iterator. Server GPU lease is bounded.
    } finally {
      client.close(force: true);
    }
  }
}
