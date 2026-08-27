import 'package:flutter/material.dart';

abstract final class AniColors {
  static const background = Color(0xFF05070C);
  static const deepNavy = Color(0xFF101114);
  static const glass = Color(0xD0121316);
  static const elevatedGlass = Color(0xF01A1B1F);
  static const purple = Color(0xFFF7F7F5);
  static const blue = Color(0xFFD5D7DC);
  static const lavender = Color(0xFFF7F7F5);
  static const secondaryText = Color(0xFFB7B9C0);
  static const mutedText = Color(0xFF747780);
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
      splashFactory: InkRipple.splashFactory,
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: const Color(0xFF111216),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AniColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AniColors.border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Colors.white, width: 1.2),
        ),
      ),
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
  colors: [Color(0xFFFFFFFF), Color(0xFFE7E8EB), Color(0xFFC9CCD2)],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);
