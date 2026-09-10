import 'dart:async';
import 'dart:io';

import 'package:audioplayers/audioplayers.dart';

/// Two short, cached cues. Audio never delays an action or requests focus.
class UiSounds {
  static bool enabled = true;
  static final Map<bool, AudioPlayer> _players = {};
  static Future<void>? _ready;
  static int _last = 0;

  static void configure(bool value) {
    enabled = value;
    if (!Platform.isAndroid && !Platform.isIOS) return;
    if (value) {
      unawaited(_prepare());
    } else {
      for (final player in _players.values) {
        unawaited(player.stop().catchError((_) {}));
      }
    }
  }

  static Future<void> _prepare() => _ready ??= _load();
  static Future<void> _load() async {
    try {
      for (final whoosh in [false, true]) {
        final player = AudioPlayer();
        _players[whoosh] = player;
        await player.setAudioContext(
          AudioContext(
            android: const AudioContextAndroid(
              audioFocus: AndroidAudioFocus.none,
            ),
            iOS: AudioContextIOS(category: AVAudioSessionCategory.ambient),
          ),
        );
        await player.setReleaseMode(ReleaseMode.stop);
        await player.setVolume(whoosh ? .22 : .30);
        await player.setSource(
          AssetSource(whoosh ? 'audio/ui_whoosh.wav' : 'audio/ui_tap.wav'),
        );
      }
    } catch (_) {
      // Unsupported/muted audio must never interfere with navigation.
    }
  }

  static void play({bool whoosh = false}) {
    if (!enabled || (!Platform.isAndroid && !Platform.isIOS)) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    if (now - _last < 120) return;
    _last = now;
    unawaited(_play(whoosh));
  }

  static Future<void> _play(bool whoosh) async {
    try {
      await _prepare();
      if (!enabled) return;
      final player = _players[whoosh];
      if (player == null) return;
      await player.seek(Duration.zero);
      if (enabled) await player.resume();
    } catch (_) {
      // Ignore transient audio failures.
    }
  }
}
