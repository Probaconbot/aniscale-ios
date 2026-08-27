import 'package:flutter/material.dart';

abstract final class AniColors {
  static const background = Color(0xFF05070C);
  static const deepNavy = Color(0xFF0A0D16);
  static const glass = Color(0xB8141725);
  static const elevatedGlass = Color(0xC71E2237);
  static const purple = Color(0xFF8B7CFF);
  static const blue = Color(0xFF50D4FF);
  static const lavender = Color(0xFFD2CCFF);
  static const secondaryText = Color(0xFFB8BECC);
  static const mutedText = Color(0xFF7A8295);
  static const success = Color(0xFF42E6A4);
  static const error = Color(0xFFFF6B81);
  static const border = Color(0x1FFFFFFF);
}

abstract final class AppTheme {
  static ThemeData get dark {
    const scheme = ColorScheme.dark(
      primary: AniColors.purple,
      secondary: AniColors.blue,
      surface: AniColors.deepNavy,
      error: AniColors.error,
    );

    return ThemeData(
      brightness: Brightness.dark,
      colorScheme: scheme,
      scaffoldBackgroundColor: AniColors.background,
      fontFamily: 'SF Pro Display',
      useMaterial3: true,
      splashFactory: InkSparkle.splashFactory,
      textTheme: const TextTheme(
        displaySmall: TextStyle(
          color: Colors.white,
          fontSize: 34,
          height: 1.08,
          fontWeight: FontWeight.w700,
          letterSpacing: -1.2,
        ),
        headlineSmall: TextStyle(
          color: Colors.white,
          fontSize: 22,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.4,
        ),
        titleMedium: TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
        bodyLarge: TextStyle(
          color: AniColors.secondaryText,
          fontSize: 16,
          height: 1.45,
        ),
        bodyMedium: TextStyle(
          color: AniColors.secondaryText,
          fontSize: 14,
          height: 1.45,
        ),
        labelLarge: TextStyle(
          color: Colors.white,
          fontSize: 15,
          fontWeight: FontWeight.w700,
        ),
      ),
      sliderTheme: const SliderThemeData(
        activeTrackColor: AniColors.purple,
        inactiveTrackColor: Color(0xFF282D42),
        thumbColor: Colors.white,
        overlayColor: Color(0x339B6CFF),
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AniColors.elevatedGlass,
        contentTextStyle: const TextStyle(color: Colors.white),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    );
  }
}

const aniGradient = LinearGradient(
  colors: [Color(0xFF806CFF), Color(0xFF4AA9FF), Color(0xFF50D4FF)],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);
