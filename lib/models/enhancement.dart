class Enhancement {
  const Enhancement({
    required this.originalPath,
    required this.outputPath,
    required this.scale,
    required this.createdAt,
    required this.originalWidth,
    required this.originalHeight,
  });

  final String originalPath;
  final String outputPath;
  final int scale;
  final DateTime createdAt;
  final int originalWidth;
  final int originalHeight;

  int get outputWidth => originalWidth * scale;
  int get outputHeight => originalHeight * scale;
}
