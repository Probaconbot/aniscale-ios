import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/app_shell.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      statusBarBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarIconBrightness: Brightness.light,
      systemNavigationBarDividerColor: Colors.transparent,
    ),
  );
  runApp(const AniScaleApp());
}

class AniScaleApp extends StatelessWidget {
  const AniScaleApp({
    super.key,
    this.showIntro = true,
    this.playIntroAudio = true,
  });

  final bool showIntro;
  final bool playIntroAudio;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AniScale',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.dark,
      home: showIntro
          ? LaunchExperience(playAudio: playIntroAudio)
          : const AppShell(),
    );
  }
}

class LaunchExperience extends StatefulWidget {
  const LaunchExperience({super.key, this.playAudio = true});

  final bool playAudio;

  @override
  State<LaunchExperience> createState() => _LaunchExperienceState();
}

class _LaunchExperienceState extends State<LaunchExperience>
    with SingleTickerProviderStateMixin {
  late final AnimationController _animation;
  final AudioPlayer _voice = AudioPlayer();
  final AudioPlayer _sfx = AudioPlayer();
  Timer? _sfxTimer;
  Timer? _openTimer;
  bool _opened = false;

  @override
  void initState() {
    super.initState();
    _animation = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    );
    unawaited(_startLaunch());
  }

  Future<void> _startLaunch() async {
    if (widget.playAudio) {
      try {
        await Future.wait([
          _voice.setSource(AssetSource('audio/voice.mp3')),
          _sfx.setSource(AssetSource('audio/sfx.mp3')),
        ]);
        await _voice.resume();
      } catch (_) {
        // The visual launch must never block the app if audio is unavailable.
      }
    }
    if (!mounted) return;
    _animation.forward();
    if (widget.playAudio) {
      _sfxTimer = Timer(
        const Duration(seconds: 1),
        () => unawaited(_playSfx()),
      );
    }
    _openTimer = Timer(const Duration(seconds: 3), () {
      if (mounted) setState(() => _opened = true);
    });
  }

  Future<void> _playSfx() async {
    try {
      await _sfx.setVolume(.82);
      await _sfx.resume();
    } catch (_) {
      // Continue the launch if a device has disabled or interrupted audio.
    }
  }

  @override
  void dispose() {
    _sfxTimer?.cancel();
    _openTimer?.cancel();
    _animation.dispose();
    unawaited(_voice.dispose());
    unawaited(_sfx.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 480),
      switchInCurve: Curves.easeOutCubic,
      switchOutCurve: Curves.easeInCubic,
      child: _opened
          ? const AppShell(key: ValueKey('app'))
          : Scaffold(
              key: const ValueKey('intro'),
              backgroundColor: Colors.black,
              body: AnimatedBuilder(
                animation: _animation,
                builder: (context, child) {
                  final t = _animation.value;
                  final opacity =
                      (t < .18
                              ? t / .18
                              : (t < .82 ? 1.0 : ((1 - t) / .18).clamp(0, 1)))
                          .toDouble();
                  final scale = .72 + Curves.easeOutExpo.transform(t) * .3;
                  return Opacity(
                    opacity: opacity,
                    child: Transform.scale(scale: scale, child: child),
                  );
                },
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 124,
                        height: 124,
                        padding: const EdgeInsets.all(4),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(32),
                          border: Border.all(color: const Color(0x66FFFFFF)),
                          boxShadow: const [
                            BoxShadow(
                              color: Color(0x32FFFFFF),
                              blurRadius: 42,
                              spreadRadius: -8,
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(28),
                          child: Image.asset(
                            'assets/aniscale_logo.jpg',
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                      const SizedBox(height: 26),
                      const Text(
                        'Welcome to AniScale',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -.5,
                        ),
                      ),
                      const SizedBox(height: 9),
                      const Text(
                        'Restore what matters.',
                        style: TextStyle(
                          color: Color(0xFF888A91),
                          fontSize: 13,
                          letterSpacing: 1.1,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
    );
  }
}
