import 'package:shared_preferences/shared_preferences.dart';

enum OutputFormat { automatic, jpeg, png }

extension OutputFormatLabel on OutputFormat {
  String get label => switch (this) {
    OutputFormat.automatic => 'Automatic',
    OutputFormat.jpeg => 'JPEG',
    OutputFormat.png => 'PNG',
  };

  String get engineValue => name;
}

class AppSettings {
  const AppSettings({
    this.outputFormat = OutputFormat.automatic,
    this.tileSize = 0,
    this.performance = 0,
    this.preserveMetadata = true,
    this.saveHistory = true,
    this.reduceMotion = false,
  });

  final OutputFormat outputFormat;
  final int tileSize;
  final int performance;
  final bool preserveMetadata;
  final bool saveHistory;
  final bool reduceMotion;

  String get tileSizeLabel => tileSize == 0 ? 'Automatic' : '$tileSize px';
  // Zero lets each native engine choose the fastest safe tile for the device.
  int get engineTileSize => tileSize;

  AppSettings copyWith({
    OutputFormat? outputFormat,
    int? tileSize,
    int? performance,
    bool? preserveMetadata,
    bool? saveHistory,
    bool? reduceMotion,
  }) => AppSettings(
    outputFormat: outputFormat ?? this.outputFormat,
    tileSize: tileSize ?? this.tileSize,
    performance: performance ?? this.performance,
    preserveMetadata: preserveMetadata ?? this.preserveMetadata,
    saveHistory: saveHistory ?? this.saveHistory,
    reduceMotion: reduceMotion ?? this.reduceMotion,
  );

  static Future<AppSettings> load() async {
    final preferences = await SharedPreferences.getInstance();
    final storedFormat = preferences.getString('outputFormat');
    return AppSettings(
      outputFormat: OutputFormat.values.firstWhere(
        (value) => value.name == storedFormat,
        orElse: () => OutputFormat.automatic,
      ),
      tileSize: preferences.getInt('tileSize') ?? 0,
      performance: preferences.getInt('performance') ?? 0,
      preserveMetadata: preferences.getBool('preserveMetadata') ?? true,
      saveHistory: preferences.getBool('saveHistory') ?? true,
      reduceMotion: preferences.getBool('reduceMotion') ?? false,
    );
  }

  Future<void> save() async {
    final preferences = await SharedPreferences.getInstance();
    await Future.wait([
      preferences.setString('outputFormat', outputFormat.name),
      preferences.setInt('tileSize', tileSize),
      preferences.setInt('performance', performance),
      preferences.setBool('preserveMetadata', preserveMetadata),
      preferences.setBool('saveHistory', saveHistory),
      preferences.setBool('reduceMotion', reduceMotion),
    ]);
  }
}
