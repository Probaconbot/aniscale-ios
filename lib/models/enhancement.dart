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
    this.isVideo = false,
    this.durationSeconds,
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
  final bool isVideo;
  final double? durationSeconds;

  int get outputWidth => actualOutputWidth ?? originalWidth * scale;
  int get outputHeight => actualOutputHeight ?? originalHeight * scale;

  Map<String, dynamic> toJson() => {
    'originalPath': originalPath,
    'outputPath': outputPath,
    'scale': scale,
    'createdAt': createdAt.toIso8601String(),
    'originalWidth': originalWidth,
    'originalHeight': originalHeight,
    'engine': engine,
    'actualOutputWidth': actualOutputWidth,
    'actualOutputHeight': actualOutputHeight,
    'isVideo': isVideo,
    'durationSeconds': durationSeconds,
  };

  factory Enhancement.fromJson(Map<String, dynamic> json) => Enhancement(
    originalPath: json['originalPath'] as String,
    outputPath: json['outputPath'] as String,
    scale: json['scale'] as int,
    createdAt: DateTime.parse(json['createdAt'] as String),
    originalWidth: json['originalWidth'] as int,
    originalHeight: json['originalHeight'] as int,
    engine: json['engine'] as String,
    actualOutputWidth: json['actualOutputWidth'] as int?,
    actualOutputHeight: json['actualOutputHeight'] as int?,
    isVideo: json['isVideo'] as bool? ?? false,
    durationSeconds: (json['durationSeconds'] as num?)?.toDouble(),
  );
}
