import 'package:flutter/material.dart';

import '../services/ui_sounds.dart';

class MotionTap extends StatefulWidget {
  const MotionTap({
    super.key,
    required this.child,
    this.onTap,
    this.borderRadius,
    this.whoosh = false,
  });
  final Widget child;
  final VoidCallback? onTap;
  final BorderRadius? borderRadius;
  final bool whoosh;
  @override
  State<MotionTap> createState() => _MotionTapState();
}

class _MotionTapState extends State<MotionTap> {
  bool _pressed = false;
  @override
  Widget build(BuildContext context) {
    final reduce = MediaQuery.disableAnimationsOf(context);
    return AnimatedScale(
      scale: _pressed && widget.onTap != null && !reduce ? .965 : 1,
      duration: Duration(milliseconds: reduce ? 0 : (_pressed ? 80 : 160)),
      curve: Curves.easeOutCubic,
      child: InkWell(
        borderRadius: widget.borderRadius,
        onHighlightChanged: (pressed) {
          if (mounted) setState(() => _pressed = pressed);
        },
        onTap: widget.onTap == null
            ? null
            : () {
                UiSounds.play(whoosh: widget.whoosh);
                widget.onTap!();
              },
        child: widget.child,
      ),
    );
  }
}

/// Retains the IndexedStack and its page state while animating tab changes.
class TabReveal extends StatefulWidget {
  const TabReveal({super.key, required this.index, required this.child});
  final int index;
  final Widget child;
  @override
  State<TabReveal> createState() => _TabRevealState();
}

class _TabRevealState extends State<TabReveal>
    with SingleTickerProviderStateMixin {
  late final AnimationController _animation = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 180),
    value: 1,
  );
  late final Animation<double> _curve = CurvedAnimation(
    parent: _animation,
    curve: Curves.easeOutCubic,
  );
  @override
  void didUpdateWidget(TabReveal oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.index != widget.index) {
      if (MediaQuery.disableAnimationsOf(context)) {
        _animation.value = 1;
      } else {
        _animation.forward(from: 0);
      }
    }
  }

  @override
  void dispose() {
    _animation.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final reduce = MediaQuery.disableAnimationsOf(context);
    return FadeTransition(
      opacity: reduce
          ? const AlwaysStoppedAnimation(1.0)
          : Tween<double>(begin: .88, end: 1).animate(_curve),
      child: SlideTransition(
        position: reduce
            ? const AlwaysStoppedAnimation(Offset.zero)
            : Tween<Offset>(
                begin: const Offset(0, .015),
                end: Offset.zero,
              ).animate(_curve),
        child: widget.child,
      ),
    );
  }
}
