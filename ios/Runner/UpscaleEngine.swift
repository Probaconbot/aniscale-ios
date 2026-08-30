import CoreML
import AVFoundation
import CoreImage
import Flutter
import ImageIO
import UIKit
import UniformTypeIdentifiers

final class UpscaleEngine: NSObject, FlutterStreamHandler {
  static let methodChannelName = "app.aniscale/upscaler"
  static let progressChannelName = "app.aniscale/upscaler_progress"

  private let modelSize = 266
  private let nativeScale = 4
  private let overlap = 16
  private let lock = NSLock()
  private var cancelled = false
  private var progressSink: FlutterEventSink?
  private func loadModel(named resource: String) -> MLModel {
    let configuration = MLModelConfiguration()
    // Let Core ML specialize each graph for the fastest supported mix of CPU,
    // GPU/Metal, and Neural Engine on the current iPhone. Excluding the GPU
    // made some convolution-heavy graphs substantially slower.
    configuration.computeUnits = .all
    guard let url = Bundle.main.url(
      forResource: resource,
      withExtension: "mlmodelc"
    ) else {
      fatalError("AniScale AI model \(resource) is missing from the app bundle")
    }
    return try! MLModel(contentsOf: url, configuration: configuration)
  }

  private lazy var fusionModel = loadModel(named: "RealESRGAN_anime_6B_266_fp16")
  private lazy var renderModel = loadModel(named: "RealESRGAN_render_x4plus_266_fp16")
  private lazy var turboModel = loadModel(named: "AniScale_turbo_animevideo_266_fp16")
  private lazy var animeUltraModel = loadModel(named: "AniUltraAnime_v2_recurrent")
  private lazy var superUltra2xModel = loadModel(named: "SuperUltra_span_x2_fp16")
  private lazy var superUltra4xModel = loadModel(named: "SuperUltra_span_x4_fp16")
  private lazy var superUltraAnime2xModel = loadModel(named: "SuperUltra_span_anime_x2_fp16")

  func register(with messenger: FlutterBinaryMessenger) {
    let methods = FlutterMethodChannel(name: Self.methodChannelName, binaryMessenger: messenger)
    methods.setMethodCallHandler { [weak self] call, result in
      guard let self else { return }
      switch call.method {
      case "upscaleImage":
        guard
          let arguments = call.arguments as? [String: Any],
          let path = arguments["path"] as? String,
          let scale = arguments["scale"] as? Int,
          scale == 2 || scale == 4
        else {
          result(FlutterError(code: "bad_arguments", message: "Invalid image path or scale.", details: nil))
          return
        }
        let preserveTransparency = arguments["preserveTransparency"] as? Bool ?? true
        let denoise = (arguments["denoise"] as? NSNumber)?.doubleValue ?? 0.2
        let sharpness = (arguments["sharpness"] as? NSNumber)?.doubleValue ?? 0.2
        let detail = (arguments["detail"] as? NSNumber)?.doubleValue ?? 0.5
        let colorFidelity = (arguments["colorFidelity"] as? NSNumber)?.doubleValue ?? 0.9
        let outputFormat = arguments["outputFormat"] as? String ?? "automatic"
        let engine = arguments["engine"] as? String ?? "fusion"
        guard engine == "fusion" || engine == "render" || engine == "turbo" else {
          result(FlutterError(code: "bad_arguments", message: "Unknown image engine.", details: nil))
          return
        }
        let requestedTileSize = arguments["tileSize"] as? Int ?? 256
        let tileSize = [128, 192, 256].contains(requestedTileSize) ? requestedTileSize : 256
        let preserveMetadata = arguments["preserveMetadata"] as? Bool ?? true
        setCancelled(false)
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            let response = try self.upscale(
              path: path,
              requestedScale: scale,
              preserveTransparency: preserveTransparency,
              denoise: denoise,
              sharpness: sharpness,
              detail: detail,
              colorFidelity: colorFidelity,
              outputFormat: outputFormat,
              tileSize: tileSize,
              preserveMetadata: preserveMetadata,
              engine: engine
            )
            DispatchQueue.main.async { result(response) }
          } catch let error as EngineError {
            DispatchQueue.main.async {
              result(FlutterError(code: error.code, message: error.message, details: nil))
            }
          } catch {
            DispatchQueue.main.async {
              result(FlutterError(code: "engine_failed", message: error.localizedDescription, details: nil))
            }
          }
        }
      case "upscaleVideo":
        guard
          let arguments = call.arguments as? [String: Any],
          let path = arguments["path"] as? String,
          let scale = arguments["scale"] as? Int,
          scale == 2 || scale == 4
        else {
          result(FlutterError(code: "bad_arguments", message: "Invalid video path or scale.", details: nil))
          return
        }
        let efficient = arguments["efficient"] as? Bool ?? true
        let engine = arguments["engine"] as? String ?? "fusion"
        let targetScale = (arguments["targetScale"] as? NSNumber)?.doubleValue ?? Double(scale)
        let content = arguments["content"] as? String ?? "auto"
        let detailMode = arguments["detailMode"] as? String ?? "natural"
        let codec = arguments["codec"] as? String ?? "hevc"
        guard engine == "fusion" || engine == "render" || engine == "turbo" || engine == "superUltra" || engine == "animeUltra" || engine == "realism",
              [1.5, 2.0, 3.0, 4.0].contains(targetScale),
              ["auto", "live", "anime"].contains(content),
              ["natural", "detailed", "sharp"].contains(detailMode),
              ["hevc", "h264"].contains(codec) else {
          result(FlutterError(code: "bad_arguments", message: "Unknown video engine.", details: nil))
          return
        }
        let requestedTileSize = arguments["tileSize"] as? Int ?? 256
        let tileSize = [128, 192, 256].contains(requestedTileSize) ? requestedTileSize : 256
        setCancelled(false)
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            let response = try self.upscaleVideo(
              path: path,
              scale: scale,
              targetScale: targetScale,
              efficient: efficient,
              tileSize: tileSize,
              engine: engine,
              content: content,
              detailMode: detailMode,
              codec: codec
            )
            DispatchQueue.main.async { result(response) }
          } catch let error as EngineError {
            DispatchQueue.main.async {
              result(FlutterError(code: error.code, message: error.message, details: nil))
            }
          } catch {
            DispatchQueue.main.async {
              result(FlutterError(code: "video_engine_failed", message: error.localizedDescription, details: nil))
            }
          }
        }
      case "cancel":
        setCancelled(true)
        result(nil)
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    let events = FlutterEventChannel(name: Self.progressChannelName, binaryMessenger: messenger)
    events.setStreamHandler(self)
  }

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    progressSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    progressSink = nil
    return nil
  }

  private func upscale(
    path: String,
    requestedScale: Int,
    preserveTransparency: Bool,
    denoise: Double,
    sharpness: Double,
    detail: Double,
    colorFidelity: Double,
    outputFormat: String,
    tileSize: Int,
    preserveMetadata: Bool,
    engine: String
  ) throws -> [String: Any] {
    guard let image = UIImage(contentsOfFile: path) else {
      throw EngineError("decode_failed", "This image could not be decoded.")
    }
    var metadata: CFDictionary?
    if preserveMetadata,
       let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) {
      metadata = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
    }
    return try upscale(
      image: image,
      requestedScale: requestedScale,
      preserveTransparency: preserveTransparency,
      denoise: denoise,
      sharpness: sharpness,
      detail: detail,
      colorFidelity: colorFidelity,
      outputFormat: outputFormat,
      tileSize: tileSize,
      metadata: metadata,
      inferenceModel: engine == "render" ? renderModel : (engine == "turbo" ? turboModel : fusionModel),
      engineLabel: engine == "render"
        ? "AniScale Render"
        : (engine == "turbo" ? "AniScale Turbo" : "AniScale Fusion"),
      tileProgress: { [weak self] value in self?.emitProgress(value) }
    )
  }

  private func upscale(
    image: UIImage,
    requestedScale: Int,
    preserveTransparency: Bool,
    denoise: Double = 0.2,
    sharpness: Double = 0.2,
    detail: Double = 0.5,
    colorFidelity: Double = 0.9,
    outputFormat: String = "automatic",
    tileSize: Int = 256,
    metadata: CFDictionary? = nil,
    inferenceModel: MLModel? = nil,
    engineLabel: String = "AniScale Fusion",
    modelInputSize: Int = 266,
    modelOutputScale: Int = 4,
    encodeOutput: Bool = true,
    tileProgress: ((Double) -> Void)?
  ) throws -> [String: Any] {
    let originalWidth = Int(image.size.width * image.scale)
    let originalHeight = Int(image.size.height * image.scale)
    let safeOutputPixels = 18_000_000.0
    let requestedOutputPixels = Double(originalWidth * requestedScale * originalHeight * requestedScale)
    var engineImage = image
    var memoryFitted = false
    if requestedOutputPixels > safeOutputPixels {
      let ratio = sqrt(safeOutputPixels / requestedOutputPixels)
      let fittedWidth = max(1, Int(Double(originalWidth) * ratio))
      let fittedHeight = max(1, Int(Double(originalHeight) * ratio))
      let format = UIGraphicsImageRendererFormat.default()
      format.scale = 1
      format.opaque = false
      engineImage = UIGraphicsImageRenderer(
        size: CGSize(width: CGFloat(fittedWidth), height: CGFloat(fittedHeight)),
        format: format
      ).image { _ in
        image.draw(
          in: CGRect(x: 0, y: 0, width: CGFloat(fittedWidth), height: CGFloat(fittedHeight))
        )
      }
      memoryFitted = true
    }
    guard let source = normalizedRGBA(engineImage) else {
      throw EngineError("decode_failed", "This image could not be decoded.")
    }
    let width = source.width
    let height = source.height
    let outputWidth = width * requestedScale
    let outputHeight = height * requestedScale
    guard outputWidth * outputHeight <= 18_500_000 else {
      throw EngineError(
        "image_too_large",
        "That output would exceed the safe memory limit. Use a smaller source image."
      )
    }

    let xStarts = tileStarts(total: width, tileSize: tileSize)
    let yStarts = tileStarts(total: height, tileSize: tileSize)
    let totalTiles = xStarts.count * yStarts.count
    var output = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
    let inputArray = try MLMultiArray(
      shape: [1, 3, NSNumber(value: modelInputSize), NSNumber(value: modelInputSize)],
      dataType: .float32
    )
    var completed = 0
    var tileInferenceMilliseconds: [Double] = []

    for (row, y0) in yStarts.enumerated() {
      for (column, x0) in xStarts.enumerated() {
        if isCancelled() { throw EngineError("cancelled", "Upscaling was cancelled.") }
        let tileWidth = min(tileSize, width - x0)
        let tileHeight = min(tileSize, height - y0)
        fillInput(
          inputArray,
          pixels: source.pixels,
          sourceWidth: width,
          sourceHeight: height,
          x0: x0,
          y0: y0,
          tileWidth: tileWidth,
          tileHeight: tileHeight,
          denoise: denoise,
          modelInputSize: modelInputSize
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
        let inferenceStarted = ProcessInfo.processInfo.systemUptime
        let prediction = try (inferenceModel ?? fusionModel).prediction(from: provider)
        tileInferenceMilliseconds.append(
          (ProcessInfo.processInfo.systemUptime - inferenceStarted) * 1_000
        )
        guard
          let values = prediction.featureValue(for: "output")?.multiArrayValue
        else {
          throw EngineError("invalid_model", "The AI model returned an invalid result.")
        }

        guard
          values.dataType == .float32,
          values.shape.count == 4,
          values.shape[1].intValue == 3,
          values.shape[2].intValue == modelInputSize * modelOutputScale,
          values.shape[3].intValue == modelInputSize * modelOutputScale
        else {
          throw EngineError("invalid_model", "The AI model returned an unexpected tensor layout.")
        }

        let coreX0 = column == 0 ? x0 : (x0 + xStarts[column - 1] + tileSize) / 2
        let coreY0 = row == 0 ? y0 : (y0 + yStarts[row - 1] + tileSize) / 2
        let coreX1 = column == xStarts.count - 1
          ? x0 + tileWidth
          : (x0 + tileWidth + xStarts[column + 1]) / 2
        let coreY1 = row == yStarts.count - 1
          ? y0 + tileHeight
          : (y0 + tileHeight + yStarts[row + 1]) / 2
        copyOutput(
          values,
          into: &output,
          outputWidth: outputWidth,
          requestedScale: requestedScale,
          modelOutputScale: modelOutputScale,
          tileX: x0,
          tileY: y0,
          coreX0: coreX0,
          coreY0: coreY0,
          coreX1: coreX1,
          coreY1: coreY1,
          sourcePixels: source.pixels,
          sourceWidth: width,
          sourceHeight: height,
          preserveTransparency: preserveTransparency && outputFormat != "jpeg",
          sharpness: sharpness,
          detail: detail,
          colorFidelity: colorFidelity
        )
        completed += 1
        tileProgress?(Double(completed) / Double(totalTiles))
      }
    }

    guard let context = CGContext(
      data: &output,
      width: outputWidth,
      height: outputHeight,
      bitsPerComponent: 8,
      bytesPerRow: outputWidth * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let cgImage = context.makeImage() else {
      throw EngineError("encode_failed", "The enhanced image could not be assembled.")
    }
    if !encodeOutput {
      return [
        "cgImage": cgImage,
        "originalWidth": originalWidth,
        "originalHeight": originalHeight,
        "outputWidth": outputWidth,
        "outputHeight": outputHeight,
        "tileInferenceMilliseconds": tileInferenceMilliseconds,
        "engine": memoryFitted
          ? "\(engineLabel) (Memory-safe Core ML)"
          : "\(engineLabel) (Core ML)"
      ]
    }
    let usePNG = outputFormat == "png" || (
      outputFormat == "automatic" && preserveTransparency && hasTransparency(source.pixels)
    )
    let outputType = usePNG ? UTType.png : UTType.jpeg
    let outputURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("aniscale_\(UUID().uuidString).\(usePNG ? "png" : "jpg")")
    guard
      let destination = CGImageDestinationCreateWithURL(
        outputURL as CFURL,
        outputType.identifier as CFString,
        1,
        nil
      )
    else {
      throw EngineError("encode_failed", "The enhanced image could not be saved.")
    }
    var encodedProperties = metadata as? [String: Any] ?? [:]
    encodedProperties[kCGImagePropertyOrientation as String] = 1
    encodedProperties[kCGImagePropertyPixelWidth as String] = outputWidth
    encodedProperties[kCGImagePropertyPixelHeight as String] = outputHeight
    if !usePNG {
      encodedProperties[kCGImageDestinationLossyCompressionQuality as String] = 0.96
    }
    let properties = encodedProperties as CFDictionary
    CGImageDestinationAddImage(destination, cgImage, properties)
    guard CGImageDestinationFinalize(destination) else {
      throw EngineError("encode_failed", "The enhanced image could not be saved.")
    }
    return [
      "path": outputURL.path,
      "originalWidth": originalWidth,
      "originalHeight": originalHeight,
      "outputWidth": outputWidth,
      "outputHeight": outputHeight,
      "tileInferenceMilliseconds": tileInferenceMilliseconds,
      "engine": memoryFitted
        ? "\(engineLabel) (Memory-safe Core ML)"
        : "\(engineLabel) (Core ML)"
    ]
  }

  private func upscaleVideo(
    path: String,
    scale: Int,
    targetScale: Double,
    efficient: Bool,
    tileSize: Int,
    engine: String,
    content: String,
    detailMode: String,
    codec: String
  ) throws -> [String: Any] {
    let benchmarkStarted = ProcessInfo.processInfo.systemUptime
    let initialThermalState = thermalStateLabel()
    var tileInferenceMilliseconds: [Double] = []
    var processedFrames = 0
    let superUltraNativeScale = targetScale <= 2 ? 2 : 4
    let selectedModel: MLModel?
    if engine == "superUltra" {
      selectedModel = superUltraNativeScale == 2
        ? (content == "anime" ? superUltraAnime2xModel : superUltra2xModel)
        : superUltra4xModel
    } else if engine == "render" {
      selectedModel = renderModel
    } else if engine == "turbo" {
      selectedModel = turboModel
    } else if engine == "fusion" {
      selectedModel = fusionModel
    } else {
      // Temporal engines own their runtimes and must not also allocate a large
      // still-image Core ML graph while processing video.
      selectedModel = nil
    }
    let engineLabel = engine == "render"
      ? "AniScale Render"
      : (engine == "turbo" ? "AniScale Turbo" : (engine == "animeUltra" ? "AniUltraAnime" : (engine == "realism" ? "AniRealism Test" : (engine == "superUltra" ? "SuperUltra" : "AniScale Fusion"))))
    let videoDenoise = engine == "superUltra"
      ? (content == "anime" ? 0.08 : 0.04)
      : (engine == "render" ? 0.34 : (engine == "turbo" ? 0.16 : 0.2))
    let videoSharpness = engine == "superUltra"
      ? (detailMode == "sharp" ? 0.55 : (detailMode == "detailed" ? 0.28 : 0.0))
      : (engine == "render" ? 0.34 : (engine == "turbo" ? 0.2 : 0.26))
    let videoDetail = engine == "superUltra"
      ? (detailMode == "sharp" ? 0.82 : (detailMode == "detailed" ? 0.68 : 0.52))
      : (engine == "render" ? 0.68 : (engine == "turbo" ? 0.48 : 0.6))
    let sourceURL = URL(fileURLWithPath: path)
    let asset = AVAsset(url: sourceURL)
    guard let videoTrack = asset.tracks(withMediaType: .video).first else {
      throw EngineError("video_decode_failed", "That file does not contain a readable video track.")
    }

    let transformedRect = CGRect(origin: .zero, size: videoTrack.naturalSize)
      .applying(videoTrack.preferredTransform)
    let sourceWidth = Int(abs(transformedRect.width).rounded())
    let sourceHeight = Int(abs(transformedRect.height).rounded())
    let requestedWidth = Int((Double(sourceWidth) * targetScale).rounded())
    let requestedHeight = Int((Double(sourceHeight) * targetScale).rounded())
    // iPhone hardware encoders generally top out around 4K. Keep 4× as a
    // usable enhancement mode by automatically fitting oversized results into
    // the largest safe 4K canvas instead of rejecting the video outright.
    let longestEdgeLimit = 3_840.0
    let pixelLimit = 8_847_360.0
    let edgeRatio = longestEdgeLimit / Double(max(requestedWidth, requestedHeight))
    let pixelRatio = sqrt(pixelLimit / Double(requestedWidth * requestedHeight))
    let outputRatio = min(1, edgeRatio, pixelRatio)
    let outputWidth = max(2, (Int(Double(requestedWidth) * outputRatio) / 2) * 2)
    let outputHeight = max(2, (Int(Double(requestedHeight) * outputRatio) / 2) * 2)
    let durationSeconds = max(CMTimeGetSeconds(asset.duration), 0.01)

    let reader = try AVAssetReader(asset: asset)
    let readerOutput = AVAssetReaderTrackOutput(
      track: videoTrack,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ]
    )
    readerOutput.alwaysCopiesSampleData = false
    guard reader.canAdd(readerOutput) else {
      throw EngineError("video_decode_failed", "The video decoder could not be configured.")
    }
    reader.add(readerOutput)

    let silentURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("aniscale_video_frames_\(UUID().uuidString).mp4")
    let finalURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("aniscale_video_\(UUID().uuidString).mp4")
    let writer = try AVAssetWriter(outputURL: silentURL, fileType: .mp4)
    let outputCodec: AVVideoCodecType = codec == "hevc" ? .hevc : .h264
    let frameRate = max(1, Int(videoTrack.nominalFrameRate.rounded()))
    let bitsPerPixel = outputCodec == .hevc ? 0.11 : 0.18
    let bitrate = min(
      60_000_000,
      max(4_000_000, Int(Double(outputWidth * outputHeight * frameRate) * bitsPerPixel))
    )
    let writerInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: outputCodec,
        AVVideoWidthKey: outputWidth,
        AVVideoHeightKey: outputHeight,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: bitrate,
          AVVideoExpectedSourceFrameRateKey: frameRate
        ]
      ]
    )
    writerInput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: writerInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: outputWidth,
        kCVPixelBufferHeightKey as String: outputHeight,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:]
      ]
    )
    guard writer.canAdd(writerInput) else {
      throw EngineError("video_encode_failed", "The video encoder could not be configured.")
    }
    writer.add(writerInput)
    guard reader.startReading(), writer.startWriting() else {
      throw EngineError("video_start_failed", reader.error?.localizedDescription ?? writer.error?.localizedDescription ?? "Video processing could not start.")
    }
    writer.startSession(atSourceTime: .zero)

    let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    let animeState = AnimeCoreMLState()
    let cdaState = CDAEngine.State()
    let cdaRuntime: CDAEngine?
    if engine == "realism" {
      do {
        cdaRuntime = try CDAEngine()
      } catch {
        throw EngineError(
          "cda_load_failed",
          "AniRealism could not load on this iPhone: \(error.localizedDescription)"
        )
      }
    } else {
      cdaRuntime = nil
    }
    var previousAnimeFrame: CGImage?
    var previousCdaFrame: CGImage?
    var currentSample = readerOutput.copyNextSampleBuffer()
    var nextAnimeSample = engine == "animeUltra" ? readerOutput.copyNextSampleBuffer() : nil
    while let sample = currentSample {
      if isCancelled() {
        reader.cancelReading()
        writer.cancelWriting()
        try? FileManager.default.removeItem(at: silentURL)
        throw EngineError("cancelled", "Video upscaling was cancelled.")
      }
      while !writerInput.isReadyForMoreMediaData {
        if isCancelled() { break }
        Thread.sleep(forTimeInterval: 0.004)
      }
      let timestamp = CMSampleBufferGetPresentationTimeStamp(sample)
      let frameStart = min(0.92, max(0.01, CMTimeGetSeconds(timestamp) / durationSeconds * 0.92))
      let frameStep = 0.92 / max(
        1,
        durationSeconds * Double(max(videoTrack.nominalFrameRate, 1))
      )
      let enhancedImage: CGImage = try autoreleasepool {
        let frameLimit: CGFloat? = engine == "animeUltra"
          ? CGFloat(efficient ? 480 : 640)
          : (engine == "realism"
            ? CGFloat(efficient ? 256 : 320)
            : (efficient ? 960 : nil))
        let frameImage = try preparedVideoFrame(
          sample,
          track: videoTrack,
          context: ciContext,
          longestEdgeLimit: frameLimit
        )
        let frameWidth = frameImage.width
        let aiScale = engine == "superUltra"
          ? superUltraNativeScale
          : (outputWidth >= frameWidth * 3 ? 4 : 2)
        let enhancedImage: CGImage
        if engine == "animeUltra" {
          let nextFrame = try nextAnimeSample.map {
            try preparedVideoFrame(
              $0,
              track: videoTrack,
              context: ciContext,
              longestEdgeLimit: frameLimit
            )
          } ?? frameImage
          let previousFrame = previousAnimeFrame ?? frameImage
          if previousAnimeFrame != nil && isSceneCut(previousFrame, frameImage) {
            animeState.reset()
          }
          let inferenceStarted = ProcessInfo.processInfo.systemUptime
          enhancedImage = try upscaleAnime(
            previous: previousFrame,
            current: frameImage,
            next: nextFrame,
            state: animeState
          )
          previousAnimeFrame = frameImage
          tileInferenceMilliseconds.append(
            (ProcessInfo.processInfo.systemUptime - inferenceStarted) * 1_000
          )
          emitProgress(min(0.92, frameStart + frameStep))
        } else if engine == "realism" {
          if let previousCdaFrame, isSceneCut(previousCdaFrame, frameImage) {
            cdaState.reset()
          }
          if cdaState.framesSinceReset >= 12 { cdaState.reset() }
          let inferenceStarted = ProcessInfo.processInfo.systemUptime
          guard let cdaRuntime else {
            throw EngineError("cda_load_failed", "AniRealism is unavailable on this iPhone.")
          }
          enhancedImage = try cdaRuntime.process(
            previous: previousCdaFrame,
            current: frameImage,
            state: cdaState
          )
          previousCdaFrame = frameImage
          tileInferenceMilliseconds.append(
            (ProcessInfo.processInfo.systemUptime - inferenceStarted) * 1_000
          )
          emitProgress(min(0.92, frameStart + frameStep))
        } else {
          guard let selectedModel else {
            throw EngineError("invalid_model", "The selected video engine is unavailable.")
          }
          let enhanced = try upscale(
            image: UIImage(cgImage: frameImage),
            requestedScale: aiScale,
            preserveTransparency: false,
            denoise: videoDenoise,
            sharpness: videoSharpness,
            detail: videoDetail,
            tileSize: tileSize,
            inferenceModel: selectedModel,
            engineLabel: engineLabel,
            modelInputSize: engine == "superUltra" ? 256 : modelSize,
            modelOutputScale: engine == "superUltra" ? superUltraNativeScale : nativeScale,
            encodeOutput: false,
            tileProgress: { [weak self] tile in
              self?.emitProgress(min(0.92, frameStart + tile * frameStep))
            }
          )
          if let frameInference = enhanced["tileInferenceMilliseconds"] as? [Double] {
            tileInferenceMilliseconds.append(contentsOf: frameInference)
          }
          guard let enhancedValue = enhanced["cgImage"] else {
            throw EngineError("video_frame_encode", "The AI engine returned no enhanced frame.")
          }
          enhancedImage = enhancedValue as! CGImage
        }
        return enhancedImage
      }
      processedFrames += 1

      // Allocate the potentially 4K encoder buffer only after model inference
      // temporaries have left their autorelease pool. Holding it during CDA-VSR
      // inference added tens of megabytes to the peak and triggered iOS jetsam.
      guard let pool = adaptor.pixelBufferPool else {
        throw EngineError("video_encode_failed", "The output pixel buffer pool is unavailable.")
      }
      var outputBuffer: CVPixelBuffer?
      guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBuffer) == kCVReturnSuccess,
            let outputBuffer else {
        throw EngineError("video_memory", "The iPhone ran out of video processing memory.")
      }
      autoreleasepool {
        var enhancedFrame = CIImage(cgImage: enhancedImage)
        let fitScale = min(
          CGFloat(outputWidth) / enhancedFrame.extent.width,
          CGFloat(outputHeight) / enhancedFrame.extent.height
        )
        if abs(fitScale - 1) > 0.001 {
          enhancedFrame = enhancedFrame.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [kCIInputScaleKey: fitScale, kCIInputAspectRatioKey: 1.0]
          )
        }
        ciContext.render(
          enhancedFrame,
          to: outputBuffer,
          bounds: CGRect(x: 0, y: 0, width: outputWidth, height: outputHeight),
          colorSpace: CGColorSpaceCreateDeviceRGB()
        )
      }
      guard adaptor.append(outputBuffer, withPresentationTime: timestamp) else {
        throw EngineError("video_encode_failed", writer.error?.localizedDescription ?? "A video frame could not be encoded.")
      }
      emitProgress(min(0.92, max(0.01, CMTimeGetSeconds(timestamp) / durationSeconds * 0.92)))
      throttleForThermalState()
      if engine == "animeUltra" {
        currentSample = nextAnimeSample
        nextAnimeSample = readerOutput.copyNextSampleBuffer()
      } else {
        currentSample = readerOutput.copyNextSampleBuffer()
      }
    }

    writerInput.markAsFinished()
    let writerFinished = DispatchSemaphore(value: 0)
    writer.finishWriting { writerFinished.signal() }
    writerFinished.wait()
    guard reader.status == .completed, writer.status == .completed else {
      throw EngineError(
        "video_encode_failed",
        reader.error?.localizedDescription ?? writer.error?.localizedDescription ?? "Video processing did not finish."
      )
    }
    emitProgress(0.95)
    try preserveAudio(from: asset, processedVideoURL: silentURL, outputURL: finalURL)
    try? FileManager.default.removeItem(at: silentURL)
    emitProgress(1)
    let processingSeconds = max(
      ProcessInfo.processInfo.systemUptime - benchmarkStarted,
      0.001
    )
    let meanInference = tileInferenceMilliseconds.isEmpty
      ? 0
      : tileInferenceMilliseconds.reduce(0, +) / Double(tileInferenceMilliseconds.count)
    return [
      "path": finalURL.path,
      "originalWidth": sourceWidth,
      "originalHeight": sourceHeight,
      "outputWidth": outputWidth,
      "outputHeight": outputHeight,
      "durationSeconds": durationSeconds,
      "benchmark": [
        "processedFrames": processedFrames,
        "processingSeconds": processingSeconds,
        "processingFps": Double(processedFrames) / processingSeconds,
        "secondsPerVideoMinute": processingSeconds / durationSeconds * 60,
        "modelInferenceMeanMs": meanInference,
        "modelInferenceP50Ms": percentile(tileInferenceMilliseconds, 0.50),
        "modelInferenceP95Ms": percentile(tileInferenceMilliseconds, 0.95),
        "thermalStart": initialThermalState,
        "thermalEnd": thermalStateLabel(),
        "computeUnits": "Core ML all (CPU/GPU/Neural Engine selected per device)",
        "utilizationNote": "Capture CPU, GPU, Neural Engine, and peak memory with Xcode Instruments on a physical iPhone."
      ],
      "engine": engine == "superUltra"
        ? "SuperUltra — \(content.capitalized), \(detailMode.capitalized) (Core ML/Metal)"
        : (engine == "animeUltra"
          ? "AniUltraAnime — AnimeSR_v2 recurrent (Core ML/Metal)"
          : (engine == "realism"
            ? "AniRealism Test — CDA-VSR recurrent (ONNX Runtime/Core ML)"
            : (efficient
              ? "\(engineLabel) (Efficient Core ML)"
              : "\(engineLabel) (Maximum Core ML)")))
    ]
  }

  private final class AnimeCoreMLState {
    var width = 0
    var height = 0
    var feedback: MLMultiArray?
    var hidden: MLMultiArray?

    func reset() {
      feedback = nil
      hidden = nil
    }
  }

  private func preparedVideoFrame(
    _ sample: CMSampleBuffer,
    track: AVAssetTrack,
    context: CIContext,
    longestEdgeLimit: CGFloat?
  ) throws -> CGImage {
    guard let buffer = CMSampleBufferGetImageBuffer(sample) else {
      throw EngineError("video_frame_decode", "A video frame could not be decoded.")
    }
    var frame = CIImage(cvPixelBuffer: buffer).transformed(by: track.preferredTransform)
    frame = frame.transformed(
      by: CGAffineTransform(translationX: -frame.extent.minX, y: -frame.extent.minY)
    )
    if let longestEdgeLimit {
      let longestEdge = max(frame.extent.width, frame.extent.height)
      if longestEdge > longestEdgeLimit {
        frame = frame.applyingFilter(
          "CILanczosScaleTransform",
          parameters: [
            kCIInputScaleKey: longestEdgeLimit / longestEdge,
            kCIInputAspectRatioKey: 1.0
          ]
        )
      }
    }
    let width = max(32, (Int(frame.extent.width.rounded()) / 4) * 4)
    let height = max(32, (Int(frame.extent.height.rounded()) / 4) * 4)
    if Int(frame.extent.width.rounded()) != width || Int(frame.extent.height.rounded()) != height {
      frame = frame.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }
    guard let image = context.createCGImage(frame, from: frame.extent) else {
      throw EngineError("video_frame_decode", "A video frame could not be prepared for AI processing.")
    }
    return image
  }

  private func upscaleAnime(
    previous: CGImage,
    current: CGImage,
    next: CGImage,
    state: AnimeCoreMLState
  ) throws -> CGImage {
    let width = current.width
    let height = current.height
    let outputWidth = width * 4
    let outputHeight = height * 4
    if state.width != width || state.height != height {
      state.width = width
      state.height = height
      state.reset()
    }
    let frames = try MLMultiArray(
      shape: [1, 9, NSNumber(value: height), NSNumber(value: width)],
      dataType: .float32
    )
    try fillAnimeFrames(frames, images: [previous, current, next])
    let feedback: MLMultiArray
    if let existingFeedback = state.feedback {
      feedback = existingFeedback
    } else {
      feedback = try zeroArray(
        shape: [1, 3, NSNumber(value: outputHeight), NSNumber(value: outputWidth)]
      )
    }
    let hidden: MLMultiArray
    if let existingHidden = state.hidden {
      hidden = existingHidden
    } else {
      hidden = try zeroArray(
        shape: [1, 64, NSNumber(value: height), NSNumber(value: width)]
      )
    }
    let provider = try MLDictionaryFeatureProvider(
      dictionary: ["frames": frames, "feedback": feedback, "recurrent_state": hidden]
    )
    let prediction = try animeUltraModel.prediction(from: provider)
    guard
      let enhanced = prediction.featureValue(for: "enhanced")?.multiArrayValue,
      let nextState = prediction.featureValue(for: "next_state")?.multiArrayValue,
      enhanced.dataType == .float32,
      nextState.dataType == .float32,
      enhanced.shape.map({ $0.intValue }) == [1, 3, outputHeight, outputWidth],
      nextState.shape.map({ $0.intValue }) == [1, 64, height, width]
    else {
      throw EngineError("anime_output", "AnimeSR_v2 returned an invalid recurrent state.")
    }
    state.feedback = enhanced
    state.hidden = nextState
    return try imageFromAnimeOutput(enhanced, width: outputWidth, height: outputHeight)
  }

  private func zeroArray(shape: [NSNumber]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    memset(array.dataPointer, 0, array.count * MemoryLayout<Float32>.size)
    return array
  }

  private func fillAnimeFrames(_ array: MLMultiArray, images: [CGImage]) throws {
    let height = array.shape[2].intValue
    let width = array.shape[3].intValue
    let channelStride = array.strides[1].intValue
    let rowStride = array.strides[2].intValue
    let columnStride = array.strides[3].intValue
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
    for (frameIndex, image) in images.enumerated() {
      guard image.width == width, image.height == height,
            let pixels = normalizedRGBA(UIImage(cgImage: image)) else {
        throw EngineError("anime_input", "AnimeSR_v2 received mismatched video frames.")
      }
      for y in 0..<height {
        for x in 0..<width {
          let sourceIndex = (y * width + x) * 4
          let destination = y * rowStride + x * columnStride
          for channel in 0..<3 {
            pointer[(frameIndex * 3 + channel) * channelStride + destination] =
              Float32(pixels.pixels[sourceIndex + channel]) / 255
          }
        }
      }
    }
  }

  private func imageFromAnimeOutput(
    _ output: MLMultiArray,
    width: Int,
    height: Int
  ) throws -> CGImage {
    let channelStride = output.strides[1].intValue
    let rowStride = output.strides[2].intValue
    let columnStride = output.strides[3].intValue
    let pointer = output.dataPointer.bindMemory(to: Float32.self, capacity: output.count)
    var pixels = [UInt8](repeating: 255, count: width * height * 4)
    for y in 0..<height {
      for x in 0..<width {
        let destination = (y * width + x) * 4
        let source = y * rowStride + x * columnStride
        pixels[destination] = byte(pointer[source])
        pixels[destination + 1] = byte(pointer[channelStride + source])
        pixels[destination + 2] = byte(pointer[2 * channelStride + source])
      }
    }
    guard let context = CGContext(
      data: &pixels,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let image = context.makeImage() else {
      throw EngineError("anime_encode", "AniUltraAnime could not assemble its frame.")
    }
    return image
  }

  private func isSceneCut(_ previous: CGImage, _ current: CGImage) -> Bool {
    guard let first = normalizedRGBA(UIImage(cgImage: previous)),
          let second = normalizedRGBA(UIImage(cgImage: current)),
          first.width == second.width,
          first.height == second.height else { return true }
    let stepX = max(1, first.width / 32)
    let stepY = max(1, first.height / 18)
    var difference = 0.0
    var samples = 0
    for y in stride(from: 0, to: first.height, by: stepY) {
      for x in stride(from: 0, to: first.width, by: stepX) {
        let index = (y * first.width + x) * 4
        let firstLuma = 0.2126 * Double(first.pixels[index])
          + 0.7152 * Double(first.pixels[index + 1])
          + 0.0722 * Double(first.pixels[index + 2])
        let secondLuma = 0.2126 * Double(second.pixels[index])
          + 0.7152 * Double(second.pixels[index + 1])
          + 0.0722 * Double(second.pixels[index + 2])
        difference += abs(firstLuma - secondLuma) / 255
        samples += 1
      }
    }
    return samples > 0 && difference / Double(samples) > 0.28
  }

  private func percentile(_ values: [Double], _ fraction: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let ordered = values.sorted()
    let position = Int((Double(ordered.count - 1) * fraction).rounded())
    return ordered[max(0, min(position, ordered.count - 1))]
  }

  private func thermalStateLabel() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }

  private func throttleForThermalState() {
    let delay: TimeInterval
    switch ProcessInfo.processInfo.thermalState {
    case .critical: delay = 0.08
    case .serious: delay = 0.045
    case .fair: delay = 0.015
    default: delay = 0
    }
    if delay > 0 { Thread.sleep(forTimeInterval: delay) }
  }

  private func preserveAudio(
    from originalAsset: AVAsset,
    processedVideoURL: URL,
    outputURL: URL
  ) throws {
    guard let audioTrack = originalAsset.tracks(withMediaType: .audio).first else {
      try FileManager.default.moveItem(at: processedVideoURL, to: outputURL)
      return
    }
    let processedAsset = AVAsset(url: processedVideoURL)
    guard let processedTrack = processedAsset.tracks(withMediaType: .video).first else {
      throw EngineError("video_mux_failed", "The processed video track could not be reopened.")
    }
    let composition = AVMutableComposition()
    guard
      let compositionVideo = composition.addMutableTrack(
        withMediaType: .video,
        preferredTrackID: kCMPersistentTrackID_Invalid
      ),
      let compositionAudio = composition.addMutableTrack(
        withMediaType: .audio,
        preferredTrackID: kCMPersistentTrackID_Invalid
      )
    else {
      throw EngineError("video_mux_failed", "Audio preservation could not be configured.")
    }
    let duration = processedAsset.duration
    try compositionVideo.insertTimeRange(
      CMTimeRange(start: .zero, duration: duration),
      of: processedTrack,
      at: .zero
    )
    try compositionAudio.insertTimeRange(
      CMTimeRange(start: .zero, duration: CMTimeMinimum(duration, originalAsset.duration)),
      of: audioTrack,
      at: .zero
    )
    guard let exporter = AVAssetExportSession(
      asset: composition,
      presetName: AVAssetExportPresetPassthrough
    ) else {
      throw EngineError("video_mux_failed", "The final video exporter is unavailable.")
    }
    exporter.outputURL = outputURL
    exporter.outputFileType = .mp4
    exporter.shouldOptimizeForNetworkUse = false
    let exportFinished = DispatchSemaphore(value: 0)
    exporter.exportAsynchronously { exportFinished.signal() }
    exportFinished.wait()
    guard exporter.status == .completed else {
      throw EngineError("video_mux_failed", exporter.error?.localizedDescription ?? "Audio could not be preserved.")
    }
  }

  private func normalizedRGBA(_ image: UIImage) -> PixelImage? {
    let width = Int(image.size.width * image.scale)
    let height = Int(image.size.height * image.scale)
    guard width > 0, height > 0 else { return nil }
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    guard let context = CGContext(
      data: &pixels,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    // UIKit images use a top-left origin while a raw bitmap CGContext uses a
    // bottom-left origin. Flip the context once so the AI input is upright.
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    UIGraphicsPushContext(context)
    image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()
    return PixelImage(width: width, height: height, pixels: pixels)
  }

  private func fillInput(
    _ array: MLMultiArray,
    pixels: [UInt8],
    sourceWidth: Int,
    sourceHeight: Int,
    x0: Int,
    y0: Int,
    tileWidth: Int,
    tileHeight: Int,
    denoise: Double,
    modelInputSize: Int
  ) {
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
    let channelStride = array.strides[1].intValue
    let rowStride = array.strides[2].intValue
    let columnStride = array.strides[3].intValue
    let cleanup = Float32(max(0, min(1, denoise))) * 0.35
    for localY in 0..<modelInputSize {
      let sourceY = y0 + reflected(localY, length: tileHeight)
      for localX in 0..<modelInputSize {
        let sourceX = x0 + reflected(localX, length: tileWidth)
        let safeX = min(sourceX, sourceWidth - 1)
        let safeY = min(sourceY, sourceHeight - 1)
        let sourceIndex = (safeY * sourceWidth + safeX) * 4
        let alpha = max(Float32(pixels[sourceIndex + 3]) / 255, 1 / 255)
        let leftIndex = (safeY * sourceWidth + max(0, safeX - 1)) * 4
        let rightIndex = (safeY * sourceWidth + min(sourceWidth - 1, safeX + 1)) * 4
        let topIndex = (max(0, safeY - 1) * sourceWidth + safeX) * 4
        let bottomIndex = (min(sourceHeight - 1, safeY + 1) * sourceWidth + safeX) * 4
        let pixelOffset = localY * rowStride + localX * columnStride
        for channel in 0..<3 {
          let center = min(Float32(pixels[sourceIndex + channel]) / 255 / alpha, 1)
          var cleaned = center
          if cleanup > 0.001 {
            var sum = center * 4
            let leftAlpha = max(Float32(pixels[leftIndex + 3]) / 255, 1 / 255)
            let rightAlpha = max(Float32(pixels[rightIndex + 3]) / 255, 1 / 255)
            let topAlpha = max(Float32(pixels[topIndex + 3]) / 255, 1 / 255)
            let bottomAlpha = max(Float32(pixels[bottomIndex + 3]) / 255, 1 / 255)
            sum += min(Float32(pixels[leftIndex + channel]) / 255 / leftAlpha, 1)
            sum += min(Float32(pixels[rightIndex + channel]) / 255 / rightAlpha, 1)
            sum += min(Float32(pixels[topIndex + channel]) / 255 / topAlpha, 1)
            sum += min(Float32(pixels[bottomIndex + channel]) / 255 / bottomAlpha, 1)
            cleaned = center * (1 - cleanup) + (sum / 8) * cleanup
          }
          pointer[channel * channelStride + pixelOffset] = cleaned
        }
      }
    }
  }

  private func copyOutput(
    _ array: MLMultiArray,
    into destination: inout [UInt8],
    outputWidth: Int,
    requestedScale: Int,
    modelOutputScale: Int,
    tileX: Int,
    tileY: Int,
    coreX0: Int,
    coreY0: Int,
    coreX1: Int,
    coreY1: Int,
    sourcePixels: [UInt8],
    sourceWidth: Int,
    sourceHeight: Int,
    preserveTransparency: Bool,
    sharpness: Double,
    detail: Double,
    colorFidelity: Double
  ) {
    let downsample = modelOutputScale / requestedScale
    let channelStride = array.strides[1].intValue
    let rowStride = array.strides[2].intValue
    let columnStride = array.strides[3].intValue
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)

    func value(channel: Int, x: Int, y: Int) -> Float32 {
      let index = channel * channelStride + y * rowStride + x * columnStride
      return pointer[index]
    }

    func tune(
      _ ai: Float32,
      source: Float32,
      fidelity: Float32,
      reconstruction: Float32
    ) -> Float32 {
      let faithfulAI = ai * (0.7 + 0.3 * fidelity) + source * (0.3 * (1 - fidelity))
      return source + (faithfulAI - source) * reconstruction
    }

    for sourceY in coreY0..<coreY1 {
      for sourceX in coreX0..<coreX1 {
        for subY in 0..<requestedScale {
          for subX in 0..<requestedScale {
            let modelX = (sourceX - tileX) * modelOutputScale + subX * downsample
            let modelY = (sourceY - tileY) * modelOutputScale + subY * downsample
            var red: Float32 = 0
            var green: Float32 = 0
            var blue: Float32 = 0
            for sampleY in 0..<downsample {
              for sampleX in 0..<downsample {
                let x = modelX + sampleX
                let y = modelY + sampleY
                red += value(channel: 0, x: x, y: y)
                green += value(channel: 1, x: x, y: y)
                blue += value(channel: 2, x: x, y: y)
              }
            }
            let divisor = Float32(downsample * downsample)
            let outputX = sourceX * requestedScale + subX
            let outputY = sourceY * requestedScale + subY
            let destinationIndex = (outputY * outputWidth + outputX) * 4
            let sourceIndex = (sourceY * sourceWidth + sourceX) * 4
            let sourceAlpha = max(Float32(sourcePixels[sourceIndex + 3]) / 255, 1 / 255)
            let sourceRed = min(Float32(sourcePixels[sourceIndex]) / 255 / sourceAlpha, 1)
            let sourceGreen = min(Float32(sourcePixels[sourceIndex + 1]) / 255 / sourceAlpha, 1)
            let sourceBlue = min(Float32(sourcePixels[sourceIndex + 2]) / 255 / sourceAlpha, 1)
            let fidelity = Float32(max(0, min(1, colorFidelity)))
            let reconstruction = Float32(0.65 + 0.35 * max(0, min(1, detail)))
            var tunedRed = tune(red / divisor, source: sourceRed, fidelity: fidelity, reconstruction: reconstruction)
            var tunedGreen = tune(green / divisor, source: sourceGreen, fidelity: fidelity, reconstruction: reconstruction)
            var tunedBlue = tune(blue / divisor, source: sourceBlue, fidelity: fidelity, reconstruction: reconstruction)
            let luminance = tunedRed * 0.2126 + tunedGreen * 0.7152 + tunedBlue * 0.0722
            let crispness = Float32(1 + 0.12 * max(0, min(1, sharpness)))
            tunedRed = luminance + (tunedRed - luminance) * crispness
            tunedGreen = luminance + (tunedGreen - luminance) * crispness
            tunedBlue = luminance + (tunedBlue - luminance) * crispness
            let alpha = preserveTransparency
              ? interpolatedAlpha(
                  pixels: sourcePixels,
                  width: sourceWidth,
                  height: sourceHeight,
                  outputX: outputX,
                  outputY: outputY,
                  scale: requestedScale
                )
              : 1
            destination[destinationIndex] = byte(tunedRed * alpha)
            destination[destinationIndex + 1] = byte(tunedGreen * alpha)
            destination[destinationIndex + 2] = byte(tunedBlue * alpha)
            destination[destinationIndex + 3] = byte(alpha)
          }
        }
      }
    }
  }

  private func tileStarts(total: Int, tileSize: Int) -> [Int] {
    if total <= tileSize { return [0] }
    let stride = tileSize - overlap
    var starts = [Int]()
    var position = 0
    while position < total {
      if position + tileSize >= total {
        let final = total - tileSize
        if starts.last != final { starts.append(final) }
        break
      }
      starts.append(position)
      position += stride
    }
    return starts
  }

  private func reflected(_ index: Int, length: Int) -> Int {
    guard length > 1 else { return 0 }
    let period = 2 * length - 2
    let wrapped = index % period
    return wrapped < length ? wrapped : period - wrapped
  }

  private func byte(_ value: Float32) -> UInt8 {
    UInt8(max(0, min(255, Int((value * 255).rounded()))))
  }

  private func hasTransparency(_ pixels: [UInt8]) -> Bool {
    var index = 3
    while index < pixels.count {
      if pixels[index] < 255 { return true }
      index += 4
    }
    return false
  }

  private func interpolatedAlpha(
    pixels: [UInt8],
    width: Int,
    height: Int,
    outputX: Int,
    outputY: Int,
    scale: Int
  ) -> Float32 {
    let sourceX = (Double(outputX) + 0.5) / Double(scale) - 0.5
    let sourceY = (Double(outputY) + 0.5) / Double(scale) - 0.5
    let x0 = max(0, min(width - 1, Int(floor(sourceX))))
    let y0 = max(0, min(height - 1, Int(floor(sourceY))))
    let x1 = min(width - 1, x0 + 1)
    let y1 = min(height - 1, y0 + 1)
    let xWeight = Float32(max(0, min(1, sourceX - Double(x0))))
    let yWeight = Float32(max(0, min(1, sourceY - Double(y0))))
    func alpha(_ x: Int, _ y: Int) -> Float32 {
      Float32(pixels[(y * width + x) * 4 + 3]) / 255
    }
    let top = alpha(x0, y0) * (1 - xWeight) + alpha(x1, y0) * xWeight
    let bottom = alpha(x0, y1) * (1 - xWeight) + alpha(x1, y1) * xWeight
    return top * (1 - yWeight) + bottom * yWeight
  }

  private func emitProgress(_ progress: Double) {
    DispatchQueue.main.async { [weak self] in self?.progressSink?(progress) }
  }

  private func setCancelled(_ value: Bool) {
    lock.lock()
    cancelled = value
    lock.unlock()
  }

  private func isCancelled() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }
}

private struct PixelImage {
  let width: Int
  let height: Int
  let pixels: [UInt8]
}

private struct EngineError: Error {
  let code: String
  let message: String

  init(_ code: String, _ message: String) {
    self.code = code
    self.message = message
  }
}
