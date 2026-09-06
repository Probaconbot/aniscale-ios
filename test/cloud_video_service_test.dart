import 'dart:convert';
import 'dart:io';

import 'package:aniscale/services/cloud_video_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late HttpServer server;
  late Directory directory;
  late Uri endpoint;
  late List<String> requests;
  String mode = 'success';
  Map? cancelled;
  final mp4 = [
    0,
    0,
    0,
    20,
    ...ascii.encode('ftypisom'),
    0,
    0,
    0,
    0,
    ...ascii.encode('isom'),
  ];

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('aniscale-cloud-test-');
    await File('${directory.path}/input.mp4').writeAsBytes(mp4);
    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    endpoint = Uri.parse('http://127.0.0.1:${server.port}');
    requests = [];
    mode = 'success';
    cancelled = null;
    server.listen((request) async {
      requests.add(request.uri.path);
      expect(request.headers.value('Authorization'), 'Bearer hf_test');
      final response = request.response;
      response.headers.contentType = ContentType.json;
      if (mode == 'unauthorized') {
        response.statusCode = 403;
        response.write('{}');
      } else if (request.uri.path == '/config') {
        response.write(
          jsonEncode({
            'dependencies': [
              {'id': 0, 'api_name': 'upscale'},
            ],
          }),
        );
      } else if (request.uri.path == '/gradio_api/upload') {
        final body = await utf8.decoder.bind(request).join();
        expect(body, contains('filename="input.mp4"'));
        response.write(jsonEncode(['/tmp/input.mp4']));
      } else if (request.uri.path == '/gradio_api/queue/join') {
        final body = jsonDecode(await utf8.decoder.bind(request).join()) as Map;
        expect(body['session_hash'], isNotEmpty);
        expect(body['fn_index'], 0);
        expect(body['simple_format'], true);
        expect((body['data'] as List)[1], 2);
        response.write('{"event_id":"job123"}');
      } else if (request.uri.path == '/gradio_api/cancel') {
        cancelled = jsonDecode(await utf8.decoder.bind(request).join()) as Map;
        response.write('{"success":true}');
      } else if (request.uri.path == '/gradio_api/queue/data') {
        response.headers.contentType = ContentType('text', 'event-stream');
        final meta = {
          'progress': 1,
          'stage': 'Ready',
          'originalWidth': 16,
          'originalHeight': 12,
          'outputWidth': 32,
          'outputHeight': 24,
          'durationSeconds': 1,
        };
        final values = mode == 'gpu_error'
            ? [
                null,
                {'error': 'Daily GPU quota exceeded'},
              ]
            : [
                {
                  'url': mode == 'unsafe_url'
                      ? 'https://evil.example/gradio_api/file=result.mp4'
                      : '$endpoint/gradio_api/file=result.mp4',
                },
                meta,
              ];
        response.write(
          'data: {"msg":"heartbeat"}\n\ndata: {"msg":"process_generating","success":true,"output":{"data":[null,{"progress":0.5,"stage":"GPU restoring"}]}}\n\n',
        );
        final message = mode == 'allocation_error'
            ? {
                'msg': 'process_completed',
                'success': false,
                'output': {
                  'error': 'GPU quota: 180 seconds requested, 154 remaining',
                },
              }
            : {
                'msg': 'process_completed',
                'success': true,
                'output': {'data': values},
              };
        response.write('data: ${jsonEncode(message)}\n\n');
      } else if (request.uri.path == '/gradio_api/file=result.mp4') {
        response.headers.contentType = ContentType('video', 'mp4');
        response.add(mode == 'bad_mp4' ? List.filled(20, 0) : mp4);
      } else {
        response.statusCode = 404;
      }
      await response.close();
    });
  });

  tearDown(() async {
    await server.close(force: true);
    await directory.delete(recursive: true);
  });

  Future<void> runFailure(String expected) async {
    final client = CloudVideoService(endpoint: endpoint, token: 'hf_test');
    await expectLater(
      client.run(
        path: '${directory.path}/input.mp4',
        scale: 2,
        detail: 'natural',
        codec: 'h264',
        outputDirectory: directory,
        onProgress: (_) {},
      ),
      throwsA(
        isA<CloudVideoException>().having(
          (e) => e.message,
          'message',
          contains(expected),
        ),
      ),
    );
    expect(
      directory.listSync().whereType<File>().where(
        (f) => f.path.endsWith('.part'),
      ),
      isEmpty,
    );
  }

  test('uploads, receives GPU progress and atomically saves result', () async {
    final stages = <String>[];
    final client = CloudVideoService(endpoint: endpoint, token: 'hf_test');
    final result = await client.run(
      path: '${directory.path}/input.mp4',
      scale: 2,
      detail: 'natural',
      codec: 'h264',
      outputDirectory: directory,
      onProgress: (p) => stages.add(p.stage),
    );
    expect(await File(result.path).readAsBytes(), mp4);
    expect(result.outputWidth, 32);
    expect(result.engine, contains('Hugging Face'));
    expect(stages, contains('GPU restoring'));
    expect(cancelled, isNull);
  });

  test(
    'rejects unauthorized Space without uploading or native fallback',
    () async {
      mode = 'unauthorized';
      await runFailure('Private Space access failed');
      expect(requests, ['/config']);
    },
  );

  test('surfaces GPU quota errors and cancels the remote iterator', () async {
    mode = 'gpu_error';
    await runFailure('Daily GPU quota exceeded');
    expect(cancelled?['event_id'], 'job123');
    expect(cancelled?['session_hash'], isNotEmpty);
  });

  test(
    'preserves the actual allocation error instead of a generic failure',
    () async {
      mode = 'allocation_error';
      await runFailure('180 seconds requested, 154 remaining');
    },
  );

  test('never forwards private token to another download host', () async {
    mode = 'unsafe_url';
    await runFailure('invalid video download');
    expect(requests, isNot(contains('/gradio_api/file=result.mp4')));
  });

  test('invalid video is removed rather than added to history', () async {
    mode = 'bad_mp4';
    await runFailure('did not return an MP4');
  });
}
