import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/enhancement.dart';

abstract final class HistoryService {
  static const _key = 'enhancementHistoryV2';

  static Future<List<Enhancement>> load() async {
    final preferences = await SharedPreferences.getInstance();
    final encoded = preferences.getString(_key);
    if (encoded == null || encoded.isEmpty) return [];
    try {
      final values = jsonDecode(encoded) as List<dynamic>;
      final history = values
          .map((value) => Enhancement.fromJson(value as Map<String, dynamic>))
          .where((item) => File(item.outputPath).existsSync())
          .take(50)
          .toList();
      if (history.length != values.length) await save(history);
      return history;
    } catch (_) {
      return [];
    }
  }

  static Future<Enhancement> persist(Enhancement enhancement) async {
    final root = await _historyDirectory();
    final stamp = enhancement.createdAt.microsecondsSinceEpoch;
    final outputPath = await _durableCopy(
      enhancement.outputPath,
      root,
      '${stamp}_output',
    );
    final originalPath = enhancement.isVideo
        ? enhancement.originalPath
        : await _durableCopy(
            enhancement.originalPath,
            root,
            '${stamp}_original',
          );
    return Enhancement(
      originalPath: originalPath,
      outputPath: outputPath,
      scale: enhancement.scale,
      createdAt: enhancement.createdAt,
      originalWidth: enhancement.originalWidth,
      originalHeight: enhancement.originalHeight,
      engine: enhancement.engine,
      actualOutputWidth: enhancement.actualOutputWidth,
      actualOutputHeight: enhancement.actualOutputHeight,
      isVideo: enhancement.isVideo,
      durationSeconds: enhancement.durationSeconds,
    );
  }

  static Future<void> save(List<Enhancement> history) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(
      _key,
      jsonEncode(history.take(50).map((item) => item.toJson()).toList()),
    );
  }

  static Future<void> delete(Enhancement enhancement) async {
    final root = await _historyDirectory();
    final nativeRoot = await _nativeOutputDirectory();
    await _deleteOwnedFile(enhancement.outputPath, [root, nativeRoot]);
    if (!enhancement.isVideo) {
      await _deleteOwnedFile(enhancement.originalPath, [root]);
    }
  }

  static Future<void> clear() async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.remove(_key);
    final root = await _historyDirectory();
    if (await root.exists()) {
      await root.delete(recursive: true);
    }
    final nativeRoot = await _nativeOutputDirectory();
    if (await nativeRoot.exists()) {
      await nativeRoot.delete(recursive: true);
    }
  }

  static Future<Directory> _historyDirectory() async {
    final documents = await getApplicationDocumentsDirectory();
    final directory = Directory(
      '${documents.path}${Platform.pathSeparator}AniScale${Platform.pathSeparator}History',
    );
    if (!await directory.exists()) await directory.create(recursive: true);
    return directory;
  }

  static Future<String> _durableCopy(
    String sourcePath,
    Directory root,
    String basename,
  ) async {
    final source = File(sourcePath);
    if (!await source.exists()) return sourcePath;
    final normalized = source.absolute.path.toLowerCase().replaceAll('\\', '/');
    if (normalized.contains('/files/enhancements/')) {
      return source.absolute.path;
    }
    final suffixIndex = sourcePath.lastIndexOf('.');
    final suffix = suffixIndex >= 0 ? sourcePath.substring(suffixIndex) : '';
    final destination = File(
      '${root.path}${Platform.pathSeparator}$basename$suffix',
    );
    if (source.absolute.path == destination.absolute.path) return sourcePath;
    await source.copy(destination.path);
    return destination.path;
  }

  static Future<Directory> _nativeOutputDirectory() async {
    final support = await getApplicationSupportDirectory();
    return Directory('${support.path}${Platform.pathSeparator}enhancements');
  }

  static Future<void> _deleteOwnedFile(
    String path,
    List<Directory> roots,
  ) async {
    final file = File(path);
    if (!await file.exists()) return;
    final candidate = file.absolute.path.toLowerCase();
    final owned = roots.any((root) {
      final ownedRoot = '${root.absolute.path}${Platform.pathSeparator}'
          .toLowerCase();
      return candidate.startsWith(ownedRoot);
    });
    if (owned) await file.delete();
  }
}
