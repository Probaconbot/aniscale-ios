class Enhancement {
  const Enhancement({
    required this.originalPath,
    required this.outputPath,
    required this.scale,
    required this.createdAt,
    required this.originalWidth,
    required this.originalHeight,
    required this.engine,
    this.actualOutputWidth,
    this.actualOutputHeight,
  });

  final String originalPath;
  final String outputPath;
  final int scale;
  final DateTime createdAt;
  final int originalWidth;
  final int originalHeight;
  final String engine;
  final int? actualOutputWidth;
  final int? actualOutputHeight;

  int get outputWidth => actualOutputWidth ?? originalWidth * scale;
  int get outputHeight => actualOutputHeight ?? originalHeight * scale;
}
