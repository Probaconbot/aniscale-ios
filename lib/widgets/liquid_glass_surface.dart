import 'dart:ui' as ui;

import 'package:flutter/material.dart';

/// Shared glass material. Pointer motion only repaints the optical overlay;
/// it never rebuilds the controls or re-lays out the content underneath it.
class LiquidGlassSurface extends StatefulWidget {
  const LiquidGlassSurface({
    super.key,
    required this.child,
    this.borderRadius = 18,
    this.padding = EdgeInsets.zero,
    this.tint = const Color(0xD0101114),
    this.blur = 8,
  });

  final Widget child;
  final double borderRadius;
  final EdgeInsetsGeometry padding;
  final Color tint;
  final double blur;

  @override
  State<LiquidGlassSurface> createState() => _LiquidGlassSurfaceState();
}

class _LiquidGlassSurfaceState extends State<LiquidGlassSurface>
    with SingleTickerProviderStateMixin {
  final _touch = ValueNotifier<Offset>(Offset.zero);
  late final AnimationController _light = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 90),
    reverseDuration: const Duration(milliseconds: 160),
  );
  late final Listenable _optics = Listenable.merge([_touch, _light]);
  int? _pointer;
  bool _reduced = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _reduced = MediaQuery.disableAnimationsOf(context);
    if (_reduced) {
      _pointer = null;
      _light.value = 0;
    }
  }

  void _release(int pointer) {
    if (_pointer != pointer) return;
    _pointer = null;
    _light.reverse();
  }

  @override
  void dispose() {
    _light.dispose();
    _touch.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final radius = BorderRadius.circular(widget.borderRadius);
    final highContrast = MediaQuery.highContrastOf(context);
    final tint = highContrast ? const Color(0xFF101114) : widget.tint;
    final backdrop = ColoredBox(color: tint);
    return Listener(
      behavior: HitTestBehavior.translucent,
      onPointerDown: (event) {
        if (_reduced || _pointer != null) return;
        _pointer = event.pointer;
        _touch.value = event.localPosition;
        _light.forward();
      },
      onPointerMove: (event) {
        if (_pointer == event.pointer) _touch.value = event.localPosition;
      },
      onPointerUp: (event) => _release(event.pointer),
      onPointerCancel: (event) => _release(event.pointer),
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: radius,
          boxShadow: const [
            BoxShadow(
              color: Color(0x4A000000),
              blurRadius: 30,
              offset: Offset(0, 14),
            ),
            BoxShadow(
              color: Color(0x14FFFFFF),
              blurRadius: 20,
              spreadRadius: -8,
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: radius,
          child: Stack(
            children: [
              Positioned.fill(
                child: _reduced || highContrast || widget.blur <= 0
                    ? backdrop
                    : BackdropFilter(
                        filter: ui.ImageFilter.blur(
                          sigmaX: widget.blur,
                          sigmaY: widget.blur,
                        ),
                        child: backdrop,
                      ),
              ),
              DecoratedBox(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Color(0x24FFFFFF),
                      Color(0x0BFFFFFF),
                      Color(0x08000000),
                    ],
                  ),
                ),
                // Keep the original one-pixel border inset and all spacing.
                child: Padding(
                  padding: const EdgeInsets.all(1),
                  child: Padding(padding: widget.padding, child: widget.child),
                ),
              ),
              Positioned.fill(
                child: IgnorePointer(
                  child: RepaintBoundary(
                    child: CustomPaint(
                      painter: _GlassOpticsPainter(
                        touch: _touch,
                        light: _light,
                        repaint: _optics,
                        radius: widget.borderRadius,
                        highContrast: highContrast,
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _GlassOpticsPainter extends CustomPainter {
  _GlassOpticsPainter({
    required this.touch,
    required this.light,
    required Listenable repaint,
    required this.radius,
    required this.highContrast,
  }) : super(repaint: repaint);

  final ValueNotifier<Offset> touch;
  final Animation<double> light;
  final double radius;
  final bool highContrast;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;
    final rect = Offset.zero & size;
    final rim = RRect.fromRectAndRadius(
      rect.deflate(.65),
      Radius.circular(radius),
    );
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    paint.shader = LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: highContrast
          ? const [Color(0xA0FFFFFF), Color(0xA0FFFFFF)]
          : const [
              Color(0x88FFFFFF),
              Color(0x25FFFFFF),
              Color(0x12FFFFFF),
              Color(0x48FFFFFF),
            ],
    ).createShader(rect);
    canvas.drawRRect(rim, paint);
    // Fine inner reflection gives the curved perimeter depth without shifting it.
    paint.shader = const LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [Color(0x22FFFFFF), Colors.transparent, Color(0x24000000)],
    ).createShader(rect);
    canvas.drawRRect(rim.deflate(1.2), paint);

    final intensity = light.value;
    if (intensity <= 0) return;
    final position = Offset(
      touch.value.dx.clamp(0.0, size.width),
      touch.value.dy.clamp(0.0, size.height),
    );
    // Coordinates come from the actual painted surface, including tall cards.
    final glow = ui.Gradient.radial(
      position,
      (size.shortestSide * 1.3).clamp(70.0, 200.0),
      [
        Colors.white.withValues(alpha: .105 * intensity),
        Colors.white.withValues(alpha: .025 * intensity),
        Colors.transparent,
      ],
      const [0, .45, 1],
    );
    canvas.drawRRect(rim, Paint()..shader = glow);
    paint.shader = ui.Gradient.radial(position, 130, [
      Colors.white.withValues(alpha: .65 * intensity),
      Colors.transparent,
    ]);
    canvas.drawRRect(rim, paint);
  }

  @override
  bool shouldRepaint(_GlassOpticsPainter oldDelegate) =>
      oldDelegate.radius != radius ||
      oldDelegate.highContrast != highContrast ||
      oldDelegate.touch != touch ||
      oldDelegate.light != light;
}
