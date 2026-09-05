import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:share_plus/share_plus.dart';
import 'package:video_player/video_player.dart';

import '../models/app_settings.dart';
import '../models/enhancement.dart';
import '../services/upscale_service.dart';
import '../services/history_service.dart';
import '../theme/app_theme.dart';
import '../widgets/liquid_glass_surface.dart';

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _tab = 0;
  final List<Enhancement> _history = [];
  AppSettings _settings = const AppSettings();

  @override
  void initState() {
    super.initState();
    _restoreState();
  }

  Future<void> _restoreState() async {
    final values = await Future.wait<dynamic>([
      AppSettings.load(),
      HistoryService.load(),
    ]);
    if (!mounted) return;
    setState(() {
      _settings = values[0] as AppSettings;
      _history
        ..clear()
        ..addAll(values[1] as List<Enhancement>);
    });
  }

  void _updateSettings(AppSettings settings) {
    setState(() => _settings = settings);
    unawaited(settings.save());
  }

  void _addEnhancement(Enhancement enhancement) async {
    if (!_settings.saveHistory) return;
    final durable = await HistoryService.persist(enhancement);
    if (!mounted) return;
    setState(() => _history.insert(0, durable));
    await HistoryService.save(_history);
  }

  void _deleteEnhancement(Enhancement enhancement) async {
    setState(() => _history.remove(enhancement));
    await HistoryService.delete(enhancement);
    await HistoryService.save(_history);
  }

  void _clearHistory() async {
    setState(_history.clear);
    await HistoryService.clear();
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      HomeScreen(
        history: _history,
        onOpenSettings: () => setState(() => _tab = 3),
        onOpenHistory: () => setState(() => _tab = 1),
        onOpenAssistant: () => setState(() => _tab = 2),
        onEnhanced: _addEnhancement,
        settings: _settings,
      ),
      HistoryScreen(history: _history, onDelete: _deleteEnhancement),
      GroqAssistantScreen(onEnhanced: _addEnhancement, settings: _settings),
      SettingsScreen(
        onClearHistory: _clearHistory,
        settings: _settings,
        onSettingsChanged: _updateSettings,
      ),
    ];

    final mediaQuery = MediaQuery.of(context);
    return MediaQuery(
      data: mediaQuery.copyWith(
        disableAnimations:
            mediaQuery.disableAnimations || _settings.reduceMotion,
      ),
      child: Scaffold(
        extendBody: true,
        body: AmbientBackground(
          child: SafeArea(
            bottom: false,
            child: IndexedStack(index: _tab, children: pages),
          ),
        ),
        bottomNavigationBar: FloatingNav(
          selectedIndex: _tab,
          onSelected: (index) => setState(() => _tab = index),
        ),
      ),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    super.key,
    required this.history,
    required this.onOpenSettings,
    required this.onOpenHistory,
    required this.onOpenAssistant,
    required this.onEnhanced,
    required this.settings,
  });

  final List<Enhancement> history;
  final VoidCallback onOpenSettings;
  final VoidCallback onOpenHistory;
  final VoidCallback onOpenAssistant;
  final ValueChanged<Enhancement> onEnhanced;
  final AppSettings settings;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  bool _videoMode = false;
  bool _picking = false;

  Future<void> _pickMedia() async {
    if (_videoMode) {
      setState(() => _picking = true);
      try {
        final file = await _picker.pickVideo(source: ImageSource.gallery);
        if (file == null || !mounted) return;
        await Navigator.of(context).push(
          MaterialPageRoute<void>(
            builder: (_) => VideoSelectedScreen(
              inputPath: file.path,
              settings: widget.settings,
              onEnhanced: widget.onEnhanced,
            ),
          ),
        );
      } catch (_) {
        _message('AniScale could not open that video. Try MP4 or MOV.');
      } finally {
        if (mounted) setState(() => _picking = false);
      }
      return;
    }
    setState(() => _picking = true);
    try {
      final file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null || !mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => EditorScreen(
            inputPath: file.path,
            onEnhanced: widget.onEnhanced,
            settings: widget.settings,
          ),
        ),
      );
    } catch (_) {
      _message('AniScale could not open that image. Try PNG, JPG, or WebP.');
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  void _message(String text) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      slivers: [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(22, 16, 22, 124),
          sliver: SliverList.list(
            children: [
              BrandHeader(onSettings: widget.onOpenSettings),
              const SizedBox(height: 30),
              Text(
                'Create a cleaner frame.',
                style: Theme.of(context).textTheme.displaySmall,
              ),
              const SizedBox(height: 12),
              const Text(
                'Choose media, select a restoration model, and let AniScale recover the detail.',
                style: TextStyle(
                  color: AniColors.secondaryText,
                  fontSize: 15,
                  height: 1.45,
                ),
              ),
              const SizedBox(height: 26),
              CommandDeck(
                videoMode: _videoMode,
                loading: _picking,
                onModeChanged: (video) => setState(() => _videoMode = video),
                onPick: _pickMedia,
                onAssistant: widget.onOpenAssistant,
                onSettings: widget.onOpenSettings,
              ),
              const SizedBox(height: 18),
              const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  FeaturePill(
                    icon: Icons.auto_awesome_rounded,
                    label: 'Free',
                    color: AniColors.purple,
                  ),
                  SizedBox(width: 8),
                  FeaturePill(
                    icon: Icons.phone_iphone_rounded,
                    label: 'Offline',
                    color: AniColors.blue,
                  ),
                  SizedBox(width: 8),
                  FeaturePill(
                    icon: Icons.shield_outlined,
                    label: 'Private',
                    color: AniColors.success,
                  ),
                ],
              ),
              if (widget.history.isNotEmpty) ...[
                const SizedBox(height: 30),
                SectionTitle(
                  title: 'Recent',
                  action: 'See all',
                  onAction: widget.onOpenHistory,
                ),
                const SizedBox(height: 12),
                ...widget.history
                    .take(2)
                    .map(
                      (item) => HistoryTile(
                        item,
                        onTap: () => openEnhancement(context, item),
                      ),
                    ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class VideoSelectedScreen extends StatefulWidget {
  const VideoSelectedScreen({
    super.key,
    required this.inputPath,
    required this.settings,
    required this.onEnhanced,
  });

  final String inputPath;
  final AppSettings settings;
  final ValueChanged<Enhancement> onEnhanced;

  @override
  State<VideoSelectedScreen> createState() => _VideoSelectedScreenState();
}

enum _VideoEngine { fusion, render, turbo, superUltra, animeUltra, realism }

class _VideoSelectedScreenState extends State<VideoSelectedScreen> {
  int _scale = 2;
  double _targetScale = 2;
  late int _performance;
  _VideoEngine _engine = _VideoEngine.fusion;
  int _content = 0;
  int _detailMode = 0;
  int _codec = 0;

  @override
  void initState() {
    super.initState();
    _performance = widget.settings.performance == 1 ? 1 : 0;
  }

  @override
  Widget build(BuildContext context) {
    final file = File(widget.inputPath);
    final filename = widget.inputPath.split(Platform.pathSeparator).last;
    final engines = Platform.isAndroid || Platform.isIOS
        ? _VideoEngine.values
        : _VideoEngine.values
              .where((engine) => engine != _VideoEngine.superUltra)
              .toList();
    final selectedEngineIndex = engines
        .indexOf(_engine)
        .clamp(0, engines.length - 1);
    return Scaffold(
      appBar: const GlassAppBar(title: 'Video'),
      body: AmbientBackground(
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 28, 20, 140),
            children: [
              GlassCard(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 42,
                ),
                child: Column(
                  children: [
                    const DecoratedBox(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient: aniGradient,
                      ),
                      child: Padding(
                        padding: EdgeInsets.all(18),
                        child: Icon(
                          Icons.video_file_rounded,
                          color: Colors.white,
                          size: 34,
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    const Text(
                      'Video selected',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      filename,
                      textAlign: TextAlign.center,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: AniColors.secondaryText),
                    ),
                    const SizedBox(height: 8),
                    FutureBuilder<int>(
                      future: file.length(),
                      builder: (context, snapshot) {
                        if (!snapshot.hasData) return const SizedBox.shrink();
                        final megabytes = snapshot.data! / (1024 * 1024);
                        return Text(
                          '${megabytes.toStringAsFixed(1)} MB • kept on this device',
                          style: const TextStyle(
                            color: AniColors.mutedText,
                            fontSize: 12,
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 18),
              const ControlLabel('VIDEO ENGINE'),
              const SizedBox(height: 9),
              SegmentedGlass(
                labels: engines
                    .map(
                      (engine) => switch (engine) {
                        _VideoEngine.fusion => 'Fusion',
                        _VideoEngine.render => 'Render',
                        _VideoEngine.turbo => 'Turbo',
                        _VideoEngine.superUltra => 'Super',
                        _VideoEngine.animeUltra => 'Anime VSR',
                        _VideoEngine.realism => 'Realism',
                      },
                    )
                    .toList(),
                selected: selectedEngineIndex,
                onSelected: (index) => setState(() {
                  _engine = engines[index];
                  if (_engine == _VideoEngine.superUltra) {
                    _scale = _targetScale <= 2 ? 2 : 4;
                  } else if (_engine == _VideoEngine.realism) {
                    _detailMode = 1;
                  } else if (_engine == _VideoEngine.animeUltra) {
                    _detailMode = 1;
                  }
                }),
              ),
              const SizedBox(height: 8),
              Text(
                switch (_engine) {
                  _VideoEngine.fusion => 'AniScale Fusion — tuned for anime and stylized 3D with strong, faithful detail.',
                  _VideoEngine.render => 'AniScale Render — a heavier 23-block model for clean 3D surfaces, sharper geometry, and restrained noise.',
                  _VideoEngine.turbo => 'AniScale Turbo — a compact video model for faster processing and lower heat.',
                  _VideoEngine.superUltra => 'SuperUltra — offline SPAN restoration with native mobile acceleration and no temporary frame files.',
                  _VideoEngine.animeUltra => 'AniUltraAnime — official AnimeSR_v2 with neighboring frames, persistent recurrent state, and automatic scene-cut reset.',
                  _VideoEngine.realism => 'AniRealism Test — CDA-VSR recurrent live-action restoration with persistent temporal states and decoded-frame priors.',
                },
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: AniColors.mutedText,
                  fontSize: 11,
                ),
              ),
              const SizedBox(height: 18),
              ControlLabel(
                _engine == _VideoEngine.superUltra
                    ? 'SUPERULTRA SCALE'
                    : _engine == _VideoEngine.animeUltra
                    ? 'ANIME UPSCALE'
                    : _engine == _VideoEngine.realism
                    ? 'LIVE-ACTION UPSCALE'
                    : 'VIDEO UPSCALE',
              ),
              const SizedBox(height: 9),
              SegmentedGlass(
                labels: _engine == _VideoEngine.superUltra
                    ? const ['1.5×', '2×', '3×', '4×']
                    : const ['2×', '4×'],
                selected: _engine == _VideoEngine.superUltra
                    ? const [1.5, 2.0, 3.0, 4.0].indexOf(_targetScale)
                    : (_scale == 2 ? 0 : 1),
                onSelected: (index) => setState(() {
                  if (_engine == _VideoEngine.superUltra) {
                    _targetScale = const [1.5, 2.0, 3.0, 4.0][index];
                    _scale = _targetScale <= 2 ? 2 : 4;
                  } else {
                    _scale = index == 0 ? 2 : 4;
                    _targetScale = _scale.toDouble();
                  }
                }),
              ),
              if (_engine == _VideoEngine.superUltra) ...[
                const SizedBox(height: 18),
                const ControlLabel('CONTENT'),
                const SizedBox(height: 9),
                SegmentedGlass(
                  labels: const ['Auto', 'Live Action', 'Anime'],
                  selected: _content,
                  onSelected: (index) => setState(() => _content = index),
                ),
              ],
              if (_engine == _VideoEngine.superUltra ||
                  _engine == _VideoEngine.realism ||
                  _engine == _VideoEngine.animeUltra) ...[
                const SizedBox(height: 18),
                const ControlLabel('DETAIL'),
                const SizedBox(height: 9),
                SegmentedGlass(
                  labels: const ['Natural', 'Detailed', 'Sharp'],
                  selected: _detailMode,
                  onSelected: (index) => setState(() => _detailMode = index),
                ),
                const SizedBox(height: 18),
                const ControlLabel('CODEC'),
                const SizedBox(height: 9),
                SegmentedGlass(
                  labels: const ['HEVC', 'H.264'],
                  selected: _codec,
                  onSelected: (index) => setState(() => _codec = index),
                ),
              ],
              const SizedBox(height: 18),
              const ControlLabel('AI PERFORMANCE'),
              const SizedBox(height: 9),
              SegmentedGlass(
                labels: const ['Efficient', 'Maximum'],
                selected: _performance,
                onSelected: (index) => setState(() => _performance = index),
              ),
              const SizedBox(height: 8),
              Text(
                _performance == 0
                    ? 'Recommended: lower heat and faster processing with a quality-preserving working resolution.'
                    : 'Higher neural working resolution. Slower, with higher memory and battery use.',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: AniColors.mutedText,
                  fontSize: 11,
                ),
              ),
              const SizedBox(height: 18),
              GlassCard(
                child: ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: const Icon(
                    Icons.bolt_rounded,
                    color: AniColors.blue,
                  ),
                  title: Text(switch (_engine) {
                    _VideoEngine.fusion => 'AniScale Fusion — Anime & 3D',
                    _VideoEngine.render => 'AniScale Render — 3D',
                    _VideoEngine.turbo => 'AniScale Turbo — Fast',
                    _VideoEngine.superUltra => 'SuperUltra — Offline SPAN',
                    _VideoEngine.animeUltra => 'AniUltraAnime — AnimeSR_v2',
                    _VideoEngine.realism => 'AniRealism Test — CDA-VSR',
                  }),
                  subtitle: Text(
                    '${switch (_engine) {
                      _VideoEngine.superUltra => Platform.isIOS ? 'Core ML/Metal FP16' : 'ncnn Vulkan FP16',
                      _VideoEngine.animeUltra => Platform.isIOS ? 'Core ML recurrent VSR' : 'ONNX Runtime recurrent VSR',
                      _VideoEngine.realism => 'ONNX Runtime recurrent CDA-VSR',
                      _ => Platform.isIOS ? 'Core ML' : 'ncnn Vulkan',
                    }} processes locally. Original audio is preserved and oversized results fit a safe 4K output.',
                  ),
                ),
              ),
              const SizedBox(height: 14),
              const Text(
                'Your video was selected locally and was never uploaded.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AniColors.mutedText, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomAction(
        label: 'Start Video Upscaling',
        note: 'Real AI processes every frame. Keep AniScale open; this can take a while.',
        onPressed: Platform.isIOS || Platform.isAndroid
            ? () => Navigator.of(context).push(
                MaterialPageRoute<void>(
                  builder: (_) => VideoProcessingScreen(
                    inputPath: widget.inputPath,
                    scale: _scale,
                    targetScale: _engine == _VideoEngine.superUltra
                        ? _targetScale
                        : _scale.toDouble(),
                    efficient: _performance == 0,
                    tileSize: widget.settings.engineTileSize,
                    engine: _engine.name,
                    content: const ['auto', 'live', 'anime'][_content],
                    detailMode: const [
                      'natural',
                      'detailed',
                      'sharp',
                    ][_detailMode],
                    codec: _codec == 0 ? 'hevc' : 'h264',
                    onEnhanced: widget.onEnhanced,
                  ),
                ),
              )
            : null,
      ),
    );
  }
}

class VideoProcessingScreen extends StatefulWidget {
  const VideoProcessingScreen({
    super.key,
    required this.inputPath,
    required this.scale,
    required this.targetScale,
    required this.efficient,
    required this.tileSize,
    required this.engine,
    required this.content,
    required this.detailMode,
    required this.codec,
    required this.onEnhanced,
  });

  final String inputPath;
  final int scale;
  final double targetScale;
  final bool efficient;
  final int tileSize;
  final String engine;
  final String content;
  final String detailMode;
  final String codec;
  final ValueChanged<Enhancement> onEnhanced;

  @override
  State<VideoProcessingScreen> createState() => _VideoProcessingScreenState();
}

class _VideoProcessingScreenState extends State<VideoProcessingScreen> {
  double _progress = .01;
  bool _cancelled = false;
  StreamSubscription<double>? _subscription;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    _subscription = upscaleProgress.listen((value) {
      if (mounted && !_cancelled) {
        setState(() => _progress = value.clamp(.01, 1));
      }
    });
    try {
      final result = await upscaleVideoLocally(
        path: widget.inputPath,
        scale: widget.scale,
        targetScale: widget.targetScale,
        efficient: widget.efficient,
        tileSize: widget.tileSize,
        engine: widget.engine,
        content: widget.content,
        detailMode: widget.detailMode,
        codec: widget.codec,
      );
      if (!mounted || _cancelled) return;
      widget.onEnhanced(
        Enhancement(
          originalPath: widget.inputPath,
          outputPath: result.path,
          scale: widget.targetScale.round(),
          createdAt: DateTime.now(),
          originalWidth: result.originalWidth,
          originalHeight: result.originalHeight,
          engine: result.engine,
          actualOutputWidth: result.outputWidth,
          actualOutputHeight: result.outputHeight,
          isVideo: true,
          durationSeconds: result.durationSeconds,
        ),
      );
      await Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => VideoResultScreen(result: result),
        ),
      );
    } catch (error) {
      if (!mounted || _cancelled) return;
      final detail = error is PlatformException && error.message != null
          ? error.message!
          : 'Try a shorter or lower-resolution video.';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Video upscaling failed. $detail')),
      );
      Navigator.of(context).pop();
    }
  }

  @override
  void dispose() {
    _subscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AmbientBackground(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(28),
            child: Column(
              children: [
                const Spacer(),
                LiquidGlassSurface(
                  borderRadius: 34,
                  padding: const EdgeInsets.all(34),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.movie_filter_rounded,
                        color: AniColors.blue,
                        size: 52,
                      ),
                      const SizedBox(height: 26),
                      SizedBox(
                        width: 96,
                        height: 96,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            CircularProgressIndicator(
                              value: _progress,
                              strokeWidth: 7,
                              backgroundColor: const Color(0x332D3350),
                              color: AniColors.blue,
                            ),
                            Center(
                              child: Text(
                                '${(_progress * 100).round()}%',
                                style: const TextStyle(
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        'AI-cleaning video frames…',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 21,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 9),
                      const Text(
                        'Audio will be added back during the finishing stage.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: AniColors.secondaryText),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 22),
                TextButton(
                  onPressed: () async {
                    _cancelled = true;
                    await cancelUpscale();
                    if (context.mounted) Navigator.of(context).pop();
                  },
                  child: const Text('Cancel'),
                ),
                const Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class VideoResultScreen extends StatefulWidget {
  const VideoResultScreen({super.key, required this.result});

  final VideoUpscaleResult result;

  @override
  State<VideoResultScreen> createState() => _VideoResultScreenState();
}

class _VideoResultScreenState extends State<VideoResultScreen> {
  bool _saving = false;
  late final VideoPlayerController _controller;
  late final Future<void> _initializing;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.file(File(widget.result.path));
    _initializing = _controller.initialize().then((_) {
      _controller.setLooping(true);
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      await Gal.putVideo(widget.result.path, album: 'AniScale');
      if (mounted) _message('Enhanced video saved to Photos.');
    } catch (_) {
      if (mounted) {
        _message(
          'Could not save the video. Check Photos permission and storage.',
        );
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _share() async {
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(widget.result.path)],
        text: 'Enhanced locally with AniScale',
      ),
    );
  }

  void _message(String text) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  @override
  Widget build(BuildContext context) {
    final result = widget.result;
    return Scaffold(
      appBar: const GlassAppBar(title: 'Video Result'),
      body: AmbientBackground(
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 28, 20, 150),
            children: [
              const Center(
                child: Icon(
                  Icons.check_circle_rounded,
                  color: AniColors.success,
                  size: 72,
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'Your video is ready.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 25, fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 8),
              const Text(
                'Enhanced locally with the original audio preserved.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AniColors.secondaryText),
              ),
              const SizedBox(height: 24),
              LiquidGlassSurface(
                borderRadius: 24,
                padding: const EdgeInsets.all(8),
                child: FutureBuilder<void>(
                  future: _initializing,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState != ConnectionState.done ||
                        !_controller.value.isInitialized) {
                      return const AspectRatio(
                        aspectRatio: 16 / 9,
                        child: Center(child: CircularProgressIndicator()),
                      );
                    }
                    return Column(
                      children: [
                        AspectRatio(
                          aspectRatio: _controller.value.aspectRatio,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(18),
                                child: VideoPlayer(_controller),
                              ),
                              Center(
                                child: IconButton.filled(
                                  tooltip: _controller.value.isPlaying
                                      ? 'Pause'
                                      : 'Play',
                                  onPressed: () {
                                    setState(() {
                                      _controller.value.isPlaying
                                          ? _controller.pause()
                                          : _controller.play();
                                    });
                                  },
                                  icon: Icon(
                                    _controller.value.isPlaying
                                        ? Icons.pause_rounded
                                        : Icons.play_arrow_rounded,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        VideoProgressIndicator(
                          _controller,
                          allowScrubbing: true,
                          padding: const EdgeInsets.only(top: 8),
                          colors: const VideoProgressColors(
                            playedColor: AniColors.purple,
                            bufferedColor: Color(0x665B65A8),
                            backgroundColor: Color(0x332D3350),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
              if (result.benchmark.isNotEmpty) ...[
                const SizedBox(height: 16),
                LiquidGlassSurface(
                  borderRadius: 24,
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'On-device performance',
                        style: TextStyle(fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 12),
                      _BenchmarkRow(
                        label: 'Processing speed',
                        value:
                            '${_number(result.benchmark['processingFps'], 2)} FPS',
                      ),
                      _BenchmarkRow(
                        label: 'Per video minute',
                        value:
                            '${_number(result.benchmark['secondsPerVideoMinute'], 1)} sec',
                      ),
                      _BenchmarkRow(
                        label: 'Core ML inference',
                        value:
                            '${_number(result.benchmark['modelInferenceMeanMs'], 1)} ms mean',
                      ),
                      _BenchmarkRow(
                        label: 'Thermal state',
                        value:
                            '${result.benchmark['thermalStart'] ?? 'unknown'} → '
                            '${result.benchmark['thermalEnd'] ?? 'unknown'}',
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'GPU, Neural Engine, CPU, and peak-memory percentages require an Instruments trace on a physical iPhone.',
                        style: TextStyle(
                          color: AniColors.mutedText,
                          fontSize: 11,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 16),
              LiquidGlassSurface(
                borderRadius: 26,
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    const Icon(
                      Icons.video_file_rounded,
                      color: AniColors.blue,
                      size: 56,
                    ),
                    const SizedBox(height: 14),
                    Text(
                      '${result.originalWidth} × ${result.originalHeight}  →  '
                      '${result.outputWidth} × ${result.outputHeight}',
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '${result.durationSeconds.toStringAsFixed(1)} seconds',
                      style: const TextStyle(color: AniColors.secondaryText),
                    ),
                    const SizedBox(height: 12),
                    FeaturePill(
                      icon: Icons.memory_rounded,
                      label: result.engine,
                      color: AniColors.blue,
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: _share,
                icon: const Icon(Icons.ios_share_rounded),
                label: const Text('Share Video'),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomAction(
        label: _saving ? 'Saving…' : 'Save Video to Photos',
        note: 'No cloud upload. No watermark.',
        onPressed: _saving ? null : _save,
      ),
    );
  }

  String _number(Object? value, int decimals) =>
      value is num ? value.toStringAsFixed(decimals) : '—';
}

class _BenchmarkRow extends StatelessWidget {
  const _BenchmarkRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 4),
    child: Row(
      children: [
        Expanded(
          child: Text(
            label,
            style: const TextStyle(color: AniColors.secondaryText),
          ),
        ),
        Text(value, style: const TextStyle(fontWeight: FontWeight.w600)),
      ],
    ),
  );
}

class EditorScreen extends StatefulWidget {
  const EditorScreen({
    super.key,
    required this.inputPath,
    required this.onEnhanced,
    required this.settings,
  });

  final String inputPath;
  final ValueChanged<Enhancement> onEnhanced;
  final AppSettings settings;

  @override
  State<EditorScreen> createState() => _EditorScreenState();
}

class _EditorScreenState extends State<EditorScreen> {
  int _scale = 2;
  bool _transparency = true;
  _VideoEngine _engine = _VideoEngine.fusion;

  void _start() {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ProcessingScreen(
          inputPath: widget.inputPath,
          scale: _scale,
          preserveTransparency: _transparency,
          onEnhanced: widget.onEnhanced,
          outputFormat: widget.settings.outputFormat.engineValue,
          tileSize: widget.settings.engineTileSize,
          performance: widget.settings.performance,
          preserveMetadata: widget.settings.preserveMetadata,
          engine: _engine.name,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: GlassAppBar(
        title: 'Upscale',
        trailing: TextButton(
          onPressed: () => setState(() {
            _scale = 2;
            _transparency = true;
            _engine = _VideoEngine.fusion;
          }),
          child: const Text('Reset'),
        ),
      ),
      body: AmbientBackground(
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 150),
            children: [
              GlassCard(
                padding: const EdgeInsets.all(8),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: SizedBox(
                    height: 310,
                    child: InteractiveViewer(
                      minScale: .7,
                      maxScale: 5,
                      child: Image.file(
                        File(widget.inputPath),
                        fit: BoxFit.contain,
                        cacheWidth: 1200,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              GlassCard(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 14,
                ),
                child: Row(
                  children: [
                    const Expanded(
                      child: Metadata(
                        label: 'ORIGINAL',
                        value: 'Auto detected',
                      ),
                    ),
                    Container(width: 1, height: 28, color: AniColors.border),
                    Expanded(
                      child: Metadata(
                        label: 'OUTPUT',
                        value: '$_scale× resolution',
                        alignEnd: true,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 22),
              const ControlLabel('UPSCALE'),
              const SizedBox(height: 9),
              SegmentedGlass(
                labels: const ['2×', '4×'],
                selected: _scale == 2 ? 0 : 1,
                onSelected: (index) =>
                    setState(() => _scale = index == 0 ? 2 : 4),
              ),
              const SizedBox(height: 22),
              const ControlLabel('AI ENGINE'),
              const SizedBox(height: 9),
              SegmentedGlass(
                labels: const ['Fusion', 'Render', 'Turbo'],
                selected: _engine.index,
                onSelected: (index) =>
                    setState(() => _engine = _VideoEngine.values[index]),
              ),
              const SizedBox(height: 9),
              Text(
                switch (_engine) {
                  _VideoEngine.fusion =>
                    'Anime and illustrations with faithful edge recovery.',
                  _VideoEngine.render => 'Photos, CGI, and rendered textures with restrained cleanup.',
                  _VideoEngine.turbo =>
                    'Compact 2×/4× model for the fastest, coolest result.',
                  _VideoEngine.superUltra => 'Video-only offline SPAN engine; choose another engine for images.',
                  _VideoEngine.animeUltra => 'Video-only recurrent AnimeSR_v2 engine; choose another engine for images.',
                  _VideoEngine.realism => 'Private-test live-action CDA-VSR engine; choose another engine for images.',
                },
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: AniColors.mutedText,
                  fontSize: 11,
                ),
              ),
              const SizedBox(height: 18),
              GlassCard(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                child: SwitchListTile.adaptive(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Preserve transparency'),
                  subtitle: Text(
                    widget.settings.outputFormat == OutputFormat.jpeg
                        ? 'Disabled because JPEG has no transparency.'
                        : 'PNG output is used when needed.',
                  ),
                  value:
                      _transparency &&
                      widget.settings.outputFormat != OutputFormat.jpeg,
                  activeTrackColor: AniColors.purple,
                  onChanged: widget.settings.outputFormat == OutputFormat.jpeg
                      ? null
                      : (value) => setState(() => _transparency = value),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomAction(
        label: 'Start Upscaling',
        note: 'Processing happens entirely on your device.',
        onPressed: _start,
      ),
    );
  }
}

class ProcessingScreen extends StatefulWidget {
  const ProcessingScreen({
    super.key,
    required this.inputPath,
    required this.scale,
    required this.preserveTransparency,
    required this.onEnhanced,
    this.denoise = .2,
    this.sharpness = .2,
    this.detail = .5,
    this.colorFidelity = .9,
    this.outputFormat = 'automatic',
    this.tileSize = 256,
    this.performance = 0,
    this.preserveMetadata = true,
    this.engine = 'fusion',
  });

  final String inputPath;
  final int scale;
  final bool preserveTransparency;
  final ValueChanged<Enhancement> onEnhanced;
  final double denoise;
  final double sharpness;
  final double detail;
  final double colorFidelity;
  final String outputFormat;
  final int tileSize;
  final int performance;
  final bool preserveMetadata;
  final String engine;

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  double _progress = .04;
  int _status = 0;
  bool _cancelled = false;
  Timer? _timer;
  StreamSubscription<double>? _progressSubscription;
  static const _statuses = [
    'Preparing media',
    'Processing pixels',
    'Preserving details',
    'Finishing',
  ];

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    _progressSubscription = upscaleProgress.listen((progress) {
      if (!mounted || _cancelled) return;
      setState(() {
        _progress = (.06 + progress * .88).clamp(.06, .94);
        _status = ((_progress * _statuses.length).floor()).clamp(
          0,
          _statuses.length - 1,
        );
      });
    });

    _timer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      if (!mounted || _cancelled || _progress >= .12) return;
      setState(() => _progress = (_progress + .01).clamp(.04, .12));
    });

    try {
      final performanceDenoise = switch (widget.performance) {
        1 => .28,
        2 => .10,
        _ => .20,
      };
      final performanceDetail = switch (widget.performance) {
        1 => .72,
        2 => .42,
        _ => .55,
      };
      final result = await upscaleLocally(
        UpscaleRequest(
          path: widget.inputPath,
          scale: widget.scale,
          preserveTransparency: widget.preserveTransparency,
          denoise: widget.denoise == .2 ? performanceDenoise : widget.denoise,
          sharpness: widget.sharpness,
          detail: widget.detail == .5 ? performanceDetail : widget.detail,
          colorFidelity: widget.colorFidelity,
          outputFormat: widget.outputFormat,
          tileSize: widget.tileSize,
          preserveMetadata: widget.preserveMetadata,
          engine: widget.engine,
          performance: widget.performance,
        ),
      );
      if (!mounted || _cancelled) return;
      _timer?.cancel();
      setState(() => _progress = 1);
      final enhancement = Enhancement(
        originalPath: widget.inputPath,
        outputPath: result.path,
        scale: widget.scale,
        createdAt: DateTime.now(),
        originalWidth: result.originalWidth,
        originalHeight: result.originalHeight,
        engine: result.engine,
        actualOutputWidth: result.outputWidth,
        actualOutputHeight: result.outputHeight,
      );
      widget.onEnhanced(enhancement);
      await Future<void>.delayed(const Duration(milliseconds: 80));
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => ResultScreen(enhancement: enhancement),
        ),
      );
    } catch (error) {
      _timer?.cancel();
      if (!mounted || _cancelled) return;
      final detail = error is PlatformException && error.message != null
          ? error.message!
          : 'Try a smaller image or free some device memory.';
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Upscaling failed. $detail')));
      Navigator.of(context).pop();
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _progressSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AmbientBackground(
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const Spacer(),
                GlassCard(
                  padding: const EdgeInsets.all(8),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Stack(
                      children: [
                        SizedBox(
                          height: 320,
                          width: double.infinity,
                          child: Image.file(
                            File(widget.inputPath),
                            fit: BoxFit.cover,
                            cacheWidth: 1000,
                          ),
                        ),
                        Positioned.fill(child: ScanLine(progress: _progress)),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 34),
                SizedBox(
                  width: 82,
                  height: 82,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CircularProgressIndicator(
                        value: _progress,
                        strokeWidth: 6,
                        backgroundColor: const Color(0xFF24283B),
                        color: AniColors.blue,
                      ),
                      Center(
                        child: Text(
                          '${(_progress * 100).round()}%',
                          style: const TextStyle(fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 22),
                Text(
                  'Enhancing your image…',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 8),
                Text(_statuses[_status]),
                const SizedBox(height: 8),
                const Text(
                  'You can keep the app open while processing.',
                  style: TextStyle(color: AniColors.mutedText, fontSize: 12),
                ),
                const SizedBox(height: 24),
                TextButton(
                  onPressed: () async {
                    _cancelled = true;
                    _timer?.cancel();
                    await cancelUpscale();
                    if (!context.mounted) return;
                    Navigator.of(context).pop();
                  },
                  child: const Text(
                    'Cancel',
                    style: TextStyle(color: AniColors.secondaryText),
                  ),
                ),
                const Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.enhancement});

  final Enhancement enhancement;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  double _split = .5;
  bool _saving = false;

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      await Gal.putImage(widget.enhancement.outputPath, album: 'AniScale');
      if (mounted) _message('Saved to Photos.');
    } catch (_) {
      if (mounted) {
        _message(
          'Could not save the image. Check photo permissions and storage.',
        );
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _share() async {
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(widget.enhancement.outputPath)],
        text: 'Enhanced privately with AniScale',
      ),
    );
  }

  void _message(String text) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  @override
  Widget build(BuildContext context) {
    final e = widget.enhancement;
    return Scaffold(
      appBar: const GlassAppBar(title: 'Result'),
      body: AmbientBackground(
        child: SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 150),
            children: [
              const Center(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: aniGradient,
                  ),
                  child: Padding(
                    padding: EdgeInsets.all(12),
                    child: Icon(Icons.check_rounded, color: Colors.black),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Text(
                'Your image is ready.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 6),
              const Text(
                'Enhanced privately on your device.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              Center(
                child: FeaturePill(
                  icon: Icons.memory_rounded,
                  label: e.engine,
                  color: AniColors.blue,
                ),
              ),
              const SizedBox(height: 16),
              GlassCard(
                padding: const EdgeInsets.all(8),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    return GestureDetector(
                      onHorizontalDragUpdate: (details) {
                        setState(
                          () => _split =
                              (details.localPosition.dx / constraints.maxWidth)
                                  .clamp(.05, .95),
                        );
                      },
                      onTapDown: (details) {
                        setState(
                          () => _split =
                              (details.localPosition.dx / constraints.maxWidth)
                                  .clamp(.05, .95),
                        );
                      },
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(18),
                        child: SizedBox(
                          height: 370,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              Image.file(
                                File(e.outputPath),
                                fit: BoxFit.cover,
                                cacheWidth: 1200,
                              ),
                              Align(
                                alignment: Alignment.centerLeft,
                                widthFactor: _split,
                                child: ClipRect(
                                  child: Align(
                                    alignment: Alignment.centerLeft,
                                    widthFactor: _split,
                                    child: SizedBox(
                                      width: constraints.maxWidth,
                                      height: 370,
                                      child: Image.file(
                                        File(e.originalPath),
                                        fit: BoxFit.cover,
                                        cacheWidth: 1200,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              Positioned(
                                left: 12,
                                top: 12,
                                child: CompareBadge('Before'),
                              ),
                              Positioned(
                                right: 12,
                                top: 12,
                                child: CompareBadge('After'),
                              ),
                              Positioned(
                                left: constraints.maxWidth * _split - 1,
                                top: 0,
                                bottom: 0,
                                child: Container(width: 2, color: Colors.white),
                              ),
                              Positioned(
                                left: constraints.maxWidth * _split - 18,
                                top: 168,
                                child: const CircleAvatar(
                                  radius: 18,
                                  backgroundColor: Colors.white,
                                  child: Icon(
                                    Icons.drag_handle_rounded,
                                    color: AniColors.deepNavy,
                                    size: 20,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 14),
              GlassCard(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 14,
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Metadata(
                        label: 'BEFORE',
                        value: '${e.originalWidth} × ${e.originalHeight}',
                      ),
                    ),
                    const Icon(
                      Icons.arrow_forward_rounded,
                      color: AniColors.purple,
                    ),
                    Expanded(
                      child: Metadata(
                        label: 'AFTER',
                        value: '${e.outputWidth} × ${e.outputHeight}',
                        alignEnd: true,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: _share,
                icon: const Icon(Icons.ios_share_rounded),
                label: const Text('Share'),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size.fromHeight(52),
                  side: const BorderSide(color: AniColors.border),
                ),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomAction(
        label: _saving ? 'Saving…' : 'Save to Photos',
        note: 'No watermark. Original quality preserved.',
        onPressed: _saving ? null : _save,
      ),
    );
  }
}

void openEnhancement(BuildContext context, Enhancement enhancement) {
  if (!File(enhancement.outputPath).existsSync()) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('This local result is no longer available.'),
      ),
    );
    return;
  }
  if (enhancement.isVideo) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => VideoResultScreen(
          result: VideoUpscaleResult(
            path: enhancement.outputPath,
            originalWidth: enhancement.originalWidth,
            originalHeight: enhancement.originalHeight,
            outputWidth: enhancement.outputWidth,
            outputHeight: enhancement.outputHeight,
            durationSeconds: enhancement.durationSeconds ?? 0,
            engine: enhancement.engine,
          ),
        ),
      ),
    );
    return;
  }
  Navigator.of(context).push(
    MaterialPageRoute<void>(
      builder: (_) => ResultScreen(enhancement: enhancement),
    ),
  );
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({
    super.key,
    required this.history,
    required this.onDelete,
  });

  final List<Enhancement> history;
  final ValueChanged<Enhancement> onDelete;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  int _filter = 0;

  @override
  Widget build(BuildContext context) {
    final visible = widget.history.where((item) {
      if (_filter == 1) return !item.isVideo;
      if (_filter == 2) return item.isVideo;
      return true;
    }).toList();
    return Padding(
      padding: const EdgeInsets.fromLTRB(22, 22, 22, 120),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('History', style: Theme.of(context).textTheme.displaySmall),
          const SizedBox(height: 8),
          const Text('Your enhancements stay on this device.'),
          const SizedBox(height: 18),
          SegmentedGlass(
            labels: const ['All', 'Images', 'Videos'],
            selected: _filter,
            onSelected: (value) => setState(() => _filter = value),
          ),
          const SizedBox(height: 18),
          if (visible.isEmpty)
            const Expanded(child: EmptyHistory())
          else
            Expanded(
              child: ListView.separated(
                itemCount: visible.length,
                separatorBuilder: (_, _) => const SizedBox(height: 10),
                itemBuilder: (_, index) {
                  final item = visible[index];
                  return HistoryTile(
                    item,
                    onTap: () => openEnhancement(context, item),
                    onDelete: () => widget.onDelete(item),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
}

class _AssistantPlan {
  const _AssistantPlan({
    required this.message,
    required this.scale,
    required this.denoise,
    required this.sharpness,
    required this.detail,
    required this.colorFidelity,
  });

  factory _AssistantPlan.fromJson(Map<String, dynamic> json) {
    double setting(String key, double fallback) =>
        ((json[key] as num?)?.toDouble() ?? fallback).clamp(0, 1).toDouble();
    return _AssistantPlan(
      message:
          json['message'] as String? ??
          'I prepared a faithful restoration recipe for this image.',
      scale: json['scale'] == 4 ? 4 : 2,
      denoise: setting('denoise', .2),
      sharpness: setting('sharpness', .2),
      detail: setting('detail', .5),
      colorFidelity: setting('colorFidelity', .9),
    );
  }

  final String message;
  final int scale;
  final double denoise;
  final double sharpness;
  final double detail;
  final double colorFidelity;
}

class GroqAssistantScreen extends StatefulWidget {
  const GroqAssistantScreen({
    super.key,
    required this.onEnhanced,
    required this.settings,
  });

  final ValueChanged<Enhancement> onEnhanced;
  final AppSettings settings;

  @override
  State<GroqAssistantScreen> createState() => _GroqAssistantScreenState();
}

class _GroqAssistantScreenState extends State<GroqAssistantScreen> {
  final _keyController = TextEditingController();
  final _promptController = TextEditingController();
  final _scrollController = ScrollController();
  final _picker = ImagePicker();
  final List<({bool user, String text})> _messages = [];
  bool _sending = false;
  bool _showKey = false;
  String? _selectedPath;
  _AssistantPlan? _pendingPlan;

  @override
  void dispose() {
    _keyController.dispose();
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null || !mounted) return;
    setState(() {
      _selectedPath = file.path;
      _pendingPlan = null;
    });
  }

  Map<String, dynamic> _decodeJsonObject(String content) {
    var clean = content.trim();
    if (clean.startsWith('```')) {
      clean = clean.replaceFirst(RegExp(r'^```(?:json)?\s*'), '');
      clean = clean.replaceFirst(RegExp(r'\s*```$'), '');
    }
    try {
      return jsonDecode(clean) as Map<String, dynamic>;
    } on FormatException {
      // Vision models occasionally wrap an otherwise valid object in a short
      // explanation. Recover that object instead of failing the request.
      final start = clean.indexOf('{');
      final end = clean.lastIndexOf('}');
      if (start < 0 || end <= start) rethrow;
      return jsonDecode(clean.substring(start, end + 1))
          as Map<String, dynamic>;
    }
  }

  _AssistantPlan _localFallbackPlan(String prompt) {
    final request = prompt.toLowerCase();
    final wantsFour =
        request.contains('4x') ||
        request.contains('4×') ||
        request.contains('maximum') ||
        request.contains('large');
    final wantsCleanup =
        request.contains('noise') ||
        request.contains('grain') ||
        request.contains('artifact') ||
        request.contains('blurry') ||
        request.contains('blur');
    final wantsSharp =
        request.contains('sharp') ||
        request.contains('detail') ||
        request.contains('clear');
    return _AssistantPlan(
      message: 'Cloud analysis was unavailable, so I prepared a conservative local recipe from your request. It keeps the image faithful and avoids heavy sharpening.',
      scale: wantsFour ? 4 : 2,
      denoise: wantsCleanup ? .38 : .2,
      sharpness: wantsSharp ? .32 : .18,
      detail: wantsSharp ? .62 : .5,
      colorFidelity: .94,
    );
  }

  void _showPlan(_AssistantPlan plan) {
    _pendingPlan = plan;
    _messages.add((
      user: false,
      text:
          '${plan.message}\n\nRecipe: ${plan.scale}× · cleanup ${(plan.denoise * 100).round()}% · detail ${(plan.detail * 100).round()}% · sharpness ${(plan.sharpness * 100).round()}%',
    ));
  }

  Future<void> _send() async {
    final key = _keyController.text.trim();
    final prompt = _promptController.text.trim();
    if (key.isEmpty || prompt.isEmpty || _sending) return;
    final selectedPath = _selectedPath;
    setState(() {
      _messages.add((user: true, text: prompt));
      _sending = true;
      _pendingPlan = null;
      _promptController.clear();
    });

    final client = HttpClient();
    try {
      final request = await client.postUrl(
        Uri.parse('https://api.groq.com/openai/v1/chat/completions'),
      );
      request.headers.contentType = ContentType.json;
      request.headers.set(HttpHeaders.authorizationHeader, 'Bearer $key');
      final history = _messages
          .skip(_messages.length > 10 ? _messages.length - 10 : 0)
          .map(
            (message) => {
              'role': message.user ? 'user' : 'assistant',
              'content': message.text,
            },
          )
          .toList();
      final hasImage = selectedPath != null;
      final visionImage = hasImage
          ? await prepareVisionImage(selectedPath)
          : null;
      final requestBody = <String, dynamic>{
        'model': hasImage ? 'qwen/qwen3.8-27b' : 'openai/gpt-oss-20b',
        'temperature': 0.25,
        'max_completion_tokens': 700,
        if (hasImage) 'reasoning_effort': 'none',
        'messages': [
          {
            'role': 'system',
            'content': hasImage
                ? 'You are the planning layer for AniScale, not the image engine. Inspect the image and translate the user request into conservative settings for AniScale Fusion. Preserve identity, composition, natural texture, and color. Avoid hallucinated detail, halos, and plastic smoothing.'
                : 'You are AniScale Assistant. Help users choose faithful image and video restoration settings. Prioritize artifact removal, natural texture, temporal stability, privacy, and honest limitations. Never claim that text AI performs the upscaling itself.',
          },
          if (!hasImage) ...history,
          if (hasImage)
            {
              'role': 'user',
              'content': [
                {'type': 'text', 'text': prompt},
                {
                  'type': 'image_url',
                  'image_url': {'url': visionImage},
                },
              ],
            },
        ],
        if (hasImage)
          'response_format': {
            'type': 'json_schema',
            'json_schema': {
              'name': 'aniscale_recipe',
              'strict': true,
              'schema': {
                'type': 'object',
                'properties': {
                  'message': {'type': 'string'},
                  'scale': {
                    'type': 'integer',
                    'enum': [2, 4],
                  },
                  'denoise': {'type': 'number', 'minimum': 0, 'maximum': 1},
                  'sharpness': {'type': 'number', 'minimum': 0, 'maximum': 1},
                  'detail': {'type': 'number', 'minimum': 0, 'maximum': 1},
                  'colorFidelity': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                  },
                },
                'required': [
                  'message',
                  'scale',
                  'denoise',
                  'sharpness',
                  'detail',
                  'colorFidelity',
                ],
                'additionalProperties': false,
              },
            },
          },
      };
      request.add(utf8.encode(jsonEncode(requestBody)));
      final response = await request.close();
      final body = await utf8.decoder.bind(response).join();
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      if (response.statusCode < 200 || response.statusCode >= 300) {
        final error = decoded['error'] as Map<String, dynamic>?;
        throw HttpException(
          error?['message'] as String? ??
              'Groq returned ${response.statusCode}.',
        );
      }
      final choices = decoded['choices'] as List<dynamic>?;
      final message = choices?.firstOrNull as Map<String, dynamic>?;
      final content =
          (message?['message'] as Map<String, dynamic>?)?['content'] as String?;
      if (content == null || content.trim().isEmpty) {
        throw const FormatException('Groq returned an empty response.');
      }
      if (mounted) {
        if (hasImage) {
          final plan = _AssistantPlan.fromJson(_decodeJsonObject(content));
          setState(() {
            _showPlan(plan);
          });
        } else {
          setState(() => _messages.add((user: false, text: content)));
        }
      }
    } catch (error) {
      if (mounted) {
        if (selectedPath != null) {
          final plan = _localFallbackPlan(prompt);
          setState(() => _showPlan(plan));
        } else {
          setState(
            () => _messages.add((
              user: false,
              text:
                  'Assistant request failed. Check the key and connection, then try again. ${error is HttpException ? error.message : ''}',
            )),
          );
        }
      }
    } finally {
      client.close(force: true);
      if (mounted) {
        setState(() => _sending = false);
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_scrollController.hasClients) {
            _scrollController.animateTo(
              _scrollController.position.maxScrollExtent,
              duration: const Duration(milliseconds: 140),
              curve: Curves.easeOut,
            );
          }
        });
      }
    }
  }

  Future<void> _applyPlan() async {
    final path = _selectedPath;
    final plan = _pendingPlan;
    if (path == null || plan == null) return;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => ProcessingScreen(
          inputPath: path,
          scale: plan.scale,
          preserveTransparency: true,
          denoise: plan.denoise,
          sharpness: plan.sharpness,
          detail: plan.detail,
          colorFidelity: plan.colorFidelity,
          outputFormat: widget.settings.outputFormat.engineValue,
          tileSize: widget.settings.engineTileSize,
          performance: widget.settings.performance,
          preserveMetadata: widget.settings.preserveMetadata,
          onEnhanced: widget.onEnhanced,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 22, 20, 116),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Icon(
                  Icons.auto_awesome_rounded,
                  color: Colors.black,
                ),
              ),
              const SizedBox(width: 13),
              Expanded(
                child: Text(
                  'AI Assistant',
                  style: Theme.of(context).textTheme.displaySmall,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Text(
            'Attach an image and describe the restoration. AniScale turns it into a precise local-engine recipe.',
          ),
          const SizedBox(height: 16),
          LiquidGlassSurface(
            borderRadius: 22,
            padding: const EdgeInsets.fromLTRB(14, 4, 8, 4),
            child: TextField(
              controller: _keyController,
              obscureText: !_showKey,
              autocorrect: false,
              enableSuggestions: false,
              decoration: InputDecoration(
                labelText: 'Groq API key · memory only',
                hintText: 'Paste your key',
                filled: false,
                border: InputBorder.none,
                enabledBorder: InputBorder.none,
                focusedBorder: InputBorder.none,
                suffixIcon: IconButton(
                  onPressed: () => setState(() => _showKey = !_showKey),
                  icon: Icon(
                    _showKey
                        ? Icons.visibility_off_outlined
                        : Icons.visibility_outlined,
                  ),
                ),
              ),
            ),
          ),
          if (_selectedPath != null) ...[
            const SizedBox(height: 12),
            SizedBox(
              height: 92,
              child: GlassCard(
                padding: const EdgeInsets.all(8),
                child: Row(
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(
                        File(_selectedPath!),
                        width: 76,
                        height: 76,
                        fit: BoxFit.cover,
                        cacheWidth: 220,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'Attached for analysis\nSent to Groq only when you tap send.',
                        style: TextStyle(fontSize: 12),
                      ),
                    ),
                    IconButton(
                      onPressed: _sending
                          ? null
                          : () => setState(() {
                              _selectedPath = null;
                              _pendingPlan = null;
                            }),
                      icon: const Icon(Icons.close_rounded),
                    ),
                  ],
                ),
              ),
            ),
          ],
          const SizedBox(height: 16),
          Expanded(
            child: _messages.isEmpty
                ? const GlassCard(
                    child: Center(
                      child: Text(
                        'Ask which mode to use, how to avoid artifacts, or why a clip is taking longer.',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                : ListView.separated(
                    controller: _scrollController,
                    itemCount: _messages.length,
                    separatorBuilder: (_, _) => const SizedBox(height: 10),
                    itemBuilder: (_, index) {
                      final message = _messages[index];
                      final bubble = Padding(
                        padding: const EdgeInsets.all(14),
                        child: Text(
                          message.text,
                          style: TextStyle(
                            color: message.user ? Colors.black : Colors.white,
                          ),
                        ),
                      );
                      return Align(
                        alignment: message.user
                            ? Alignment.centerRight
                            : Alignment.centerLeft,
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 330),
                          child: message.user
                              ? DecoratedBox(
                                  decoration: BoxDecoration(
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(22),
                                  ),
                                  child: bubble,
                                )
                              : LiquidGlassSurface(
                                  borderRadius: 22,
                                  child: bubble,
                                ),
                        ),
                      );
                    },
                  ),
          ),
          if (_pendingPlan != null) ...[
            const SizedBox(height: 10),
            GradientButton(
              label: 'Apply Recipe with AniScale',
              icon: Icons.auto_fix_high_rounded,
              onPressed: _applyPlan,
            ),
          ],
          const SizedBox(height: 12),
          LiquidGlassSurface(
            borderRadius: 30,
            padding: const EdgeInsets.all(6),
            tint: const Color(0xF0141518),
            child: Row(
              children: [
                IconButton.filledTonal(
                  onPressed: _sending ? null : _pickImage,
                  tooltip: 'Attach image',
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xFF242529),
                    foregroundColor: Colors.white,
                  ),
                  icon: const Icon(Icons.add_rounded),
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: TextField(
                    controller: _promptController,
                    minLines: 1,
                    maxLines: 3,
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _send(),
                    decoration: const InputDecoration(
                      hintText: '@Describe, restore, or ask AI…',
                      filled: false,
                      border: InputBorder.none,
                      enabledBorder: InputBorder.none,
                      focusedBorder: InputBorder.none,
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                IconButton.filled(
                  onPressed: _sending ? null : _send,
                  style: IconButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.black,
                    minimumSize: const Size(50, 50),
                  ),
                  icon: _sending
                      ? const SizedBox.square(
                          dimension: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.arrow_upward_rounded),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    super.key,
    required this.onClearHistory,
    required this.settings,
    required this.onSettingsChanged,
  });

  final VoidCallback onClearHistory;
  final AppSettings settings;
  final ValueChanged<AppSettings> onSettingsChanged;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _version = '1.9.0';

  @override
  void initState() {
    super.initState();
    PackageInfo.fromPlatform().then((info) {
      if (mounted) setState(() => _version = info.version);
    });
  }

  void _change(AppSettings settings) => widget.onSettingsChanged(settings);

  Future<void> _chooseFormat() async {
    final selected = await showModalBottomSheet<OutputFormat>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const ListTile(title: Text('Default output format')),
            for (final format in OutputFormat.values)
              ListTile(
                title: Text(format.label),
                subtitle: Text(switch (format) {
                  OutputFormat.automatic =>
                    'JPEG for opaque images, PNG for transparency.',
                  OutputFormat.jpeg =>
                    'Smaller files with high-quality compression.',
                  OutputFormat.png => 'Lossless output with larger files.',
                }),
                trailing: format == widget.settings.outputFormat
                    ? const Icon(Icons.check_rounded)
                    : null,
                onTap: () => Navigator.pop(context, format),
              ),
          ],
        ),
      ),
    );
    if (selected != null) {
      _change(widget.settings.copyWith(outputFormat: selected));
    }
  }

  Future<void> _chooseTileSize() async {
    final selected = await showModalBottomSheet<int>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const ListTile(title: Text('AI tile size')),
            for (final tile in const [0, 128, 192, 256])
              ListTile(
                title: Text(tile == 0 ? 'Automatic' : '$tile px'),
                subtitle: Text(
                  tile == 128
                      ? 'Lowest memory; slower due to more tiles.'
                      : tile == 256
                      ? 'Fastest; uses the most working memory.'
                      : tile == 192
                      ? 'Balanced manual tile size.'
                      : 'AniScale chooses the fastest safe tile for the selected engine and device.',
                ),
                trailing: tile == widget.settings.tileSize
                    ? const Icon(Icons.check_rounded)
                    : null,
                onTap: () => Navigator.pop(context, tile),
              ),
          ],
        ),
      ),
    );
    if (selected != null) {
      _change(widget.settings.copyWith(tileSize: selected));
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 22, 22, 124),
      children: [
        Text('Settings', style: Theme.of(context).textTheme.displaySmall),
        const SizedBox(height: 24),
        const ControlLabel('OUTPUT'),
        const SizedBox(height: 9),
        GlassCard(
          child: Column(
            children: [
              SettingRow(
                icon: Icons.image_outlined,
                title: 'Default format',
                value: widget.settings.outputFormat.label,
                onTap: _chooseFormat,
              ),
              const Divider(color: AniColors.border, height: 1),
              SwitchListTile.adaptive(
                title: const Text('Preserve metadata'),
                value: widget.settings.preserveMetadata,
                activeTrackColor: AniColors.purple,
                onChanged: (value) =>
                    _change(widget.settings.copyWith(preserveMetadata: value)),
              ),
            ],
          ),
        ),
        const SizedBox(height: 20),
        const ControlLabel('PERFORMANCE'),
        const SizedBox(height: 9),
        GlassCard(
          child: Column(
            children: [
              SettingRow(
                icon: Icons.grid_view_rounded,
                title: 'Tile size',
                value: widget.settings.tileSizeLabel,
                onTap: _chooseTileSize,
              ),
              const Divider(color: AniColors.border, height: 1),
              Padding(
                padding: const EdgeInsets.all(14),
                child: SegmentedGlass(
                  labels: const ['Balanced', 'Quality', 'Faster'],
                  selected: widget.settings.performance,
                  onSelected: (value) =>
                      _change(widget.settings.copyWith(performance: value)),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 20),
        const ControlLabel('PRIVACY & APPEARANCE'),
        const SizedBox(height: 9),
        GlassCard(
          child: Column(
            children: [
              SwitchListTile.adaptive(
                title: const Text('Save processing history'),
                value: widget.settings.saveHistory,
                activeTrackColor: AniColors.purple,
                onChanged: (value) =>
                    _change(widget.settings.copyWith(saveHistory: value)),
              ),
              const Divider(color: AniColors.border, height: 1),
              SwitchListTile.adaptive(
                title: const Text('Reduce animations'),
                value: widget.settings.reduceMotion,
                activeTrackColor: AniColors.purple,
                onChanged: (value) =>
                    _change(widget.settings.copyWith(reduceMotion: value)),
              ),
              const Divider(color: AniColors.border, height: 1),
              ListTile(
                leading: const Icon(
                  Icons.delete_outline_rounded,
                  color: AniColors.error,
                ),
                title: const Text('Clear history'),
                onTap: () async {
                  final confirmed = await showDialog<bool>(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('Clear local history?'),
                      content: const Text(
                        'Saved AniScale previews and history entries will be removed from this device.',
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context, false),
                          child: const Text('Cancel'),
                        ),
                        FilledButton(
                          onPressed: () => Navigator.pop(context, true),
                          child: const Text('Clear'),
                        ),
                      ],
                    ),
                  );
                  if (confirmed != true || !context.mounted) return;
                  widget.onClearHistory();
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Local history cleared.')),
                  );
                },
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        const GlassCard(
          child: ListTile(
            leading: Icon(Icons.shield_outlined, color: AniColors.success),
            title: Text('Private by design'),
            subtitle: Text(
              'Upscaling stays on-device. Assistant prompts and an attached preview are sent to Groq only when you tap send.',
            ),
          ),
        ),
        const SizedBox(height: 20),
        const ControlLabel('ABOUT'),
        const SizedBox(height: 9),
        GlassCard(
          child: Column(
            children: [
              SettingRow(
                icon: Icons.info_outline_rounded,
                title: 'About AniScale',
                value: _version,
                onTap: () => showAboutDialog(
                  context: context,
                  applicationName: 'AniScale',
                  applicationVersion: _version,
                  applicationLegalese:
                      'Free, private image and video enhancement.',
                ),
              ),
              const Divider(color: AniColors.border, height: 1),
              SettingRow(
                icon: Icons.memory_rounded,
                title: 'Upscale engine',
                value: Platform.isIOS
                    ? 'Fusion + Render + Turbo + SuperUltra + AniUltraAnime'
                    : Platform.isAndroid
                    ? 'Fusion + Render + Turbo + SuperUltra + AniUltraAnime'
                    : 'Mobile resampler',
                onTap: () => showModalBottomSheet<void>(
                  context: context,
                  builder: (context) => const SafeArea(
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'AniScale engines',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          SizedBox(height: 14),
                          Text('Fusion — anime and stylized 3D quality.'),
                          SizedBox(height: 8),
                          Text('Render — photos, CGI, and rendered textures.'),
                          SizedBox(height: 8),
                          Text('Turbo — compact native 2×/4× processing.'),
                          SizedBox(height: 8),
                          Text(
                            'SuperUltra — offline SPAN 1.5×/2×/3×/4× video restoration with Live Action and Anime modes.',
                          ),
                          SizedBox(height: 8),
                          Text(
                            'AniUltraAnime — official AnimeSR_v2 recurrent anime-video restoration with neighboring-frame context and scene-cut state reset.',
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              const Divider(color: AniColors.border, height: 1),
              SettingRow(
                icon: Icons.description_outlined,
                title: 'Open-source licences',
                value: '',
                onTap: () => showLicensePage(
                  context: context,
                  applicationName: 'AniScale',
                  applicationVersion: _version,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class AmbientBackground extends StatelessWidget {
  const AmbientBackground({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        color: AniColors.background,
        gradient: RadialGradient(
          center: Alignment(-.7, -1),
          radius: 1.35,
          colors: [Color(0x24FFFFFF), Color(0xFF101012), AniColors.background],
          stops: [0, .46, 1],
        ),
      ),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Positioned(
            right: -100,
            bottom: 20,
            child: Container(
              width: 320,
              height: 320,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [Color(0x1FFFFFFF), Colors.transparent],
                ),
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

class GlassCard extends StatelessWidget {
  const GlassCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(16),
  });

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    return LiquidGlassSurface(
      borderRadius: 22,
      padding: padding,
      child: Material(type: MaterialType.transparency, child: child),
    );
  }
}

class LiquidGlassIconButton extends StatelessWidget {
  const LiquidGlassIconButton({
    super.key,
    required this.icon,
    required this.onPressed,
    this.tooltip,
  });

  final IconData icon;
  final VoidCallback onPressed;
  final String? tooltip;

  @override
  Widget build(BuildContext context) {
    return LiquidGlassSurface(
      borderRadius: 15,
      tint: const Color(0xE0121316),
      blur: 8,
      child: Tooltip(
        message: tooltip ?? '',
        child: Material(
          type: MaterialType.transparency,
          child: InkWell(
            borderRadius: BorderRadius.circular(15),
            onTap: onPressed,
            child: SizedBox(
              width: 44,
              height: 44,
              child: Icon(icon, size: 20, color: Colors.white),
            ),
          ),
        ),
      ),
    );
  }
}

class BrandHeader extends StatelessWidget {
  const BrandHeader({super.key, required this.onSettings});

  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: Image.asset(
            'assets/aniscale_logo.jpg',
            width: 34,
            height: 34,
            fit: BoxFit.cover,
            cacheWidth: 102,
          ),
        ),
        const SizedBox(width: 10),
        const Text(
          'AniScale',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
        ),
        const Spacer(),
        LiquidGlassIconButton(
          onPressed: onSettings,
          icon: Icons.tune_rounded,
          tooltip: 'Settings',
        ),
      ],
    );
  }
}

class CommandDeck extends StatelessWidget {
  const CommandDeck({
    super.key,
    required this.videoMode,
    required this.loading,
    required this.onModeChanged,
    required this.onPick,
    required this.onAssistant,
    required this.onSettings,
  });

  final bool videoMode;
  final bool loading;
  final ValueChanged<bool> onModeChanged;
  final VoidCallback onPick;
  final VoidCallback onAssistant;
  final VoidCallback onSettings;

  @override
  Widget build(BuildContext context) {
    Widget action(
      IconData icon,
      String label,
      VoidCallback onTap, {
      bool active = false,
    }) {
      return Material(
        color: active ? Colors.white : const Color(0xFF111215),
        borderRadius: BorderRadius.circular(24),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(24),
          child: Container(
            height: 48,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: const Color(0x32FFFFFF)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  icon,
                  size: 20,
                  color: active ? Colors.black : Colors.white,
                ),
                const SizedBox(width: 9),
                Text(
                  label,
                  style: TextStyle(
                    color: active ? Colors.black : Colors.white,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Column(
      children: [
        LiquidGlassSurface(
          borderRadius: 34,
          padding: const EdgeInsets.all(7),
          tint: const Color(0xF00D0E10),
          child: Row(
            children: [
              IconButton.filledTonal(
                onPressed: loading ? null : onPick,
                icon: const Icon(Icons.add_rounded),
                style: IconButton.styleFrom(
                  backgroundColor: const Color(0xFF1C1D21),
                  foregroundColor: Colors.white,
                ),
              ),
              const SizedBox(width: 6),
              const Expanded(
                child: Text(
                  '@Search, enhance, upscale, or ask AI…',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: AniColors.secondaryText,
                    fontSize: 14,
                  ),
                ),
              ),
              IconButton(
                onPressed: loading ? null : onPick,
                icon: const Icon(Icons.attach_file_rounded, size: 21),
                color: AniColors.secondaryText,
              ),
              IconButton(
                onPressed: onAssistant,
                icon: const Icon(Icons.graphic_eq_rounded, size: 21),
                color: AniColors.secondaryText,
              ),
              IconButton.filled(
                onPressed: loading ? null : onPick,
                icon: loading
                    ? const SizedBox.square(
                        dimension: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.arrow_upward_rounded),
                style: IconButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 14),
        Wrap(
          spacing: 9,
          runSpacing: 9,
          alignment: WrapAlignment.center,
          children: [
            action(
              Icons.video_call_outlined,
              'Video',
              () => onModeChanged(true),
              active: videoMode,
            ),
            action(
              Icons.image_outlined,
              'Image',
              () => onModeChanged(false),
              active: !videoMode,
            ),
            action(Icons.auto_awesome_rounded, 'Upscale', onPick),
            action(Icons.chat_bubble_outline_rounded, 'Ask AI', onAssistant),
          ],
        ),
        const SizedBox(height: 14),
        Row(
          children: [
            Expanded(
              child: action(
                videoMode ? Icons.movie_filter_outlined : Icons.photo_outlined,
                loading
                    ? 'Opening…'
                    : 'Choose ${videoMode ? 'video' : 'image'}',
                onPick,
              ),
            ),
            const SizedBox(width: 9),
            action(Icons.tune_rounded, 'Controls', onSettings),
          ],
        ),
      ],
    );
  }
}

class UploadCard extends StatelessWidget {
  const UploadCard({
    super.key,
    required this.videoMode,
    required this.loading,
    required this.onPressed,
  });

  final bool videoMode;
  final bool loading;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: const EdgeInsets.fromLTRB(24, 38, 24, 26),
      child: Column(
        children: [
          ShaderMask(
            shaderCallback: aniGradient.createShader,
            child: Icon(
              videoMode ? Icons.video_library_outlined : Icons.image_outlined,
              color: Colors.white,
              size: 44,
            ),
          ),
          const SizedBox(height: 22),
          Text(
            videoMode ? 'Choose a video.' : 'Choose an image.',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 6),
          Text(
            videoMode ? 'MP4 or MOV. Processed offline.' : 'PNG, JPG, or WebP.',
            style: const TextStyle(color: AniColors.mutedText),
          ),
          const SizedBox(height: 24),
          GradientButton(
            label: loading
                ? 'Opening…'
                : (videoMode ? 'Select Video' : 'Select Image'),
            icon: loading ? null : Icons.add_photo_alternate_outlined,
            onPressed: loading ? null : onPressed,
          ),
          const SizedBox(height: 18),
          Container(
            height: 46,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0x2BFFFFFF),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AniColors.border),
            ),
            child: const Text(
              'or drag & drop here',
              style: TextStyle(color: AniColors.mutedText, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

class GradientButton extends StatelessWidget {
  const GradientButton({
    super.key,
    required this.label,
    this.icon,
    this.onPressed,
  });

  final String label;
  final IconData? icon;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: onPressed == null
            ? const LinearGradient(colors: [Colors.grey, Colors.blueGrey])
            : aniGradient,
        borderRadius: BorderRadius.circular(18),
        boxShadow: const [
          BoxShadow(color: Color(0x24FFFFFF), blurRadius: 24, spreadRadius: -6),
        ],
      ),
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: icon == null ? const SizedBox.shrink() : Icon(icon, size: 19),
        label: Text(label),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          foregroundColor: Colors.black,
          minimumSize: const Size.fromHeight(54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
        ),
      ),
    );
  }
}

class SegmentedGlass extends StatelessWidget {
  const SegmentedGlass({
    super.key,
    required this.labels,
    required this.selected,
    required this.onSelected,
  });

  final List<String> labels;
  final int selected;
  final ValueChanged<int> onSelected;

  @override
  Widget build(BuildContext context) {
    return LiquidGlassSurface(
      borderRadius: 16,
      padding: const EdgeInsets.all(4),
      tint: const Color(0xE0101114),
      child: SizedBox(
        height: 44,
        child: Row(
          children: List.generate(labels.length, (index) {
            final active = index == selected;
            return Expanded(
              child: GestureDetector(
                onTap: () => onSelected(index),
                child: AnimatedContainer(
                  duration: Duration(
                    milliseconds: MediaQuery.disableAnimationsOf(context)
                        ? 0
                        : 140,
                  ),
                  curve: Curves.easeOutCubic,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: active ? Colors.white : null,
                    borderRadius: BorderRadius.circular(12),
                    border: active
                        ? Border.all(color: const Color(0x55FFFFFF))
                        : null,
                    boxShadow: active
                        ? const [
                            BoxShadow(color: Color(0x24FFFFFF), blurRadius: 12),
                          ]
                        : null,
                  ),
                  child: Text(
                    labels[index],
                    style: TextStyle(
                      color: active ? Colors.black : AniColors.mutedText,
                      fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                      fontSize: 13,
                    ),
                  ),
                ),
              ),
            );
          }),
        ),
      ),
    );
  }
}

class FeaturePill extends StatelessWidget {
  const FeaturePill({
    super.key,
    required this.icon,
    required this.label,
    required this.color,
  });

  final IconData icon;
  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: .1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: .28)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 13),
          const SizedBox(width: 5),
          Text(label, style: const TextStyle(fontSize: 11)),
        ],
      ),
    );
  }
}

class FloatingNav extends StatelessWidget {
  const FloatingNav({
    super.key,
    required this.selectedIndex,
    required this.onSelected,
  });

  final int selectedIndex;
  final ValueChanged<int> onSelected;

  @override
  Widget build(BuildContext context) {
    const items = [
      (Icons.home_outlined, 'Home'),
      (Icons.history_rounded, 'History'),
      (Icons.auto_awesome_outlined, 'Assistant'),
      (Icons.tune_rounded, 'Settings'),
    ];
    return SafeArea(
      minimum: const EdgeInsets.fromLTRB(18, 0, 18, 12),
      child: LiquidGlassSurface(
        borderRadius: 24,
        tint: const Color(0xD8101114),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 7),
        child: SizedBox(
          height: 58,
          child: Row(
            children: List.generate(items.length, (index) {
              final active = index == selectedIndex;
              return Expanded(
                child: InkWell(
                  borderRadius: BorderRadius.circular(17),
                  onTap: () => onSelected(index),
                  child: AnimatedContainer(
                    duration: Duration(
                      milliseconds: MediaQuery.disableAnimationsOf(context)
                          ? 0
                          : 140,
                    ),
                    curve: Curves.easeOutCubic,
                    decoration: BoxDecoration(
                      color: active ? Colors.white : null,
                      borderRadius: BorderRadius.circular(17),
                      border: active
                          ? Border.all(color: const Color(0x55FFFFFF))
                          : null,
                      boxShadow: active
                          ? const [
                              BoxShadow(
                                color: Color(0x22FFFFFF),
                                blurRadius: 14,
                              ),
                            ]
                          : null,
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          items[index].$1,
                          color: active ? Colors.black : AniColors.mutedText,
                          size: 21,
                        ),
                        const SizedBox(height: 3),
                        Text(
                          items[index].$2,
                          style: TextStyle(
                            color: active ? Colors.black : AniColors.mutedText,
                            fontSize: 10,
                            fontWeight: active
                                ? FontWeight.w700
                                : FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }),
          ),
        ),
      ),
    );
  }
}

class GlassAppBar extends StatelessWidget implements PreferredSizeWidget {
  const GlassAppBar({super.key, required this.title, this.trailing});

  final String title;
  final Widget? trailing;

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);

  @override
  Widget build(BuildContext context) {
    final canGoBack = Navigator.of(context).canPop();
    return AppBar(
      backgroundColor: AniColors.background.withValues(alpha: .72),
      surfaceTintColor: Colors.transparent,
      centerTitle: true,
      automaticallyImplyLeading: false,
      leadingWidth: canGoBack ? 60 : null,
      leading: canGoBack
          ? Padding(
              padding: const EdgeInsets.only(left: 10, top: 6, bottom: 6),
              child: LiquidGlassIconButton(
                icon: Icons.arrow_back_ios_new_rounded,
                tooltip: 'Back',
                onPressed: () => Navigator.of(context).maybePop(),
              ),
            )
          : null,
      title: Text(
        title,
        style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
      ),
      actions: trailing == null ? null : [trailing!, const SizedBox(width: 8)],
    );
  }
}

class BottomAction extends StatelessWidget {
  const BottomAction({
    super.key,
    required this.label,
    required this.note,
    required this.onPressed,
  });

  final String label;
  final String note;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(
          sigmaX: MediaQuery.disableAnimationsOf(context) ? 0 : 8,
          sigmaY: MediaQuery.disableAnimationsOf(context) ? 0 : 8,
        ),
        child: SafeArea(
          top: false,
          child: Container(
            padding: const EdgeInsets.fromLTRB(20, 12, 20, 10),
            color: AniColors.background.withValues(alpha: .82),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                GradientButton(
                  label: label,
                  icon: Icons.auto_awesome_rounded,
                  onPressed: onPressed,
                ),
                const SizedBox(height: 7),
                Text(
                  note,
                  style: const TextStyle(
                    color: AniColors.mutedText,
                    fontSize: 11,
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

class TuningSlider extends StatelessWidget {
  const TuningSlider({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        children: [
          Row(
            children: [
              Text(label),
              const Spacer(),
              Text(
                '${(value * 100).round()}',
                style: const TextStyle(
                  color: AniColors.lavender,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          Slider(value: value, onChanged: onChanged),
        ],
      ),
    );
  }
}

class Metadata extends StatelessWidget {
  const Metadata({
    super.key,
    required this.label,
    required this.value,
    this.alignEnd = false,
  });

  final String label;
  final String value;
  final bool alignEnd;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: alignEnd
          ? CrossAxisAlignment.end
          : CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: AniColors.mutedText,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: .8,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
        ),
      ],
    );
  }
}

class ControlLabel extends StatelessWidget {
  const ControlLabel(this.text, {super.key});
  final String text;

  @override
  Widget build(BuildContext context) => Text(
    text,
    style: const TextStyle(
      color: AniColors.mutedText,
      fontSize: 11,
      fontWeight: FontWeight.w700,
      letterSpacing: 1.2,
    ),
  );
}

class SectionTitle extends StatelessWidget {
  const SectionTitle({
    super.key,
    required this.title,
    this.action,
    this.onAction,
  });
  final String title;
  final String? action;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) => Row(
    children: [
      Text(title, style: Theme.of(context).textTheme.titleMedium),
      const Spacer(),
      if (action != null)
        TextButton(
          onPressed: onAction,
          child: Text(
            action!,
            style: const TextStyle(color: AniColors.lavender, fontSize: 12),
          ),
        ),
    ],
  );
}

class HistoryTile extends StatelessWidget {
  const HistoryTile(this.enhancement, {super.key, this.onTap, this.onDelete});
  final Enhancement enhancement;
  final VoidCallback? onTap;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: EdgeInsets.zero,
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: SizedBox(
                  width: 62,
                  height: 62,
                  child: enhancement.isVideo
                      ? const ColoredBox(
                          color: Color(0xFF171A28),
                          child: Icon(
                            Icons.play_circle_fill_rounded,
                            color: AniColors.blue,
                          ),
                        )
                      : Image.file(
                          File(enhancement.outputPath),
                          fit: BoxFit.cover,
                          cacheWidth: 180,
                          errorBuilder: (_, _, _) => const ColoredBox(
                            color: Color(0xFF171A28),
                            child: Icon(Icons.broken_image_outlined),
                          ),
                        ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      enhancement.isVideo ? 'Enhanced video' : 'Enhanced image',
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${enhancement.outputWidth} × ${enhancement.outputHeight} • ${enhancement.scale}×',
                      style: const TextStyle(
                        color: AniColors.mutedText,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              if (onDelete != null)
                IconButton(
                  tooltip: 'Delete local result',
                  onPressed: onDelete,
                  icon: const Icon(
                    Icons.delete_outline_rounded,
                    color: AniColors.mutedText,
                  ),
                )
              else
                const Icon(
                  Icons.chevron_right_rounded,
                  color: AniColors.mutedText,
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class EmptyHistory extends StatelessWidget {
  const EmptyHistory({super.key});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(
            Icons.auto_awesome_motion_outlined,
            color: AniColors.purple,
            size: 48,
          ),
          const SizedBox(height: 18),
          Text(
            'Your enhanced media will appear here.',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 7),
          const Text(
            'Everything stays on your device.',
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class SettingRow extends StatelessWidget {
  const SettingRow({
    super.key,
    required this.icon,
    required this.title,
    required this.value,
    this.onTap,
  });
  final IconData icon;
  final String title;
  final String value;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      onTap: onTap,
      leading: Icon(icon, color: AniColors.lavender),
      title: Text(title),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (value.isNotEmpty)
            Text(value, style: const TextStyle(color: AniColors.mutedText)),
          if (onTap != null) ...[
            const SizedBox(width: 4),
            const Icon(
              Icons.chevron_right_rounded,
              color: AniColors.mutedText,
              size: 19,
            ),
          ],
        ],
      ),
    );
  }
}

class ScanLine extends StatelessWidget {
  const ScanLine({super.key, required this.progress});
  final double progress;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (_, constraints) => Stack(
        children: [
          Positioned(
            top: constraints.maxHeight * progress,
            left: 0,
            right: 0,
            child: Container(
              height: 3,
              decoration: const BoxDecoration(
                gradient: aniGradient,
                boxShadow: [
                  BoxShadow(
                    color: AniColors.blue,
                    blurRadius: 16,
                    spreadRadius: 3,
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CompareBadge extends StatelessWidget {
  const CompareBadge(this.label, {super.key});
  final String label;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          color: const Color(0x99070A12),
          child: Text(
            label,
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700),
          ),
        ),
      ),
    );
  }
}
