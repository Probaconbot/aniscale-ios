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

  private let tileSize = 256
  private let modelSize = 266
  private let nativeScale = 4
  private let overlap = 16
  private let lock = NSLock()
  private var cancelled = false
  private var progressSink: FlutterEventSink?
  private lazy var model: MLModel = {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    guard let url = Bundle.main.url(
      forResource: "RealESRGAN_anime_6B_266_fp16",
      withExtension: "mlmodelc"
    ) else {
      fatalError("AniScale AI model is missing from the app bundle")
    }
    return try! MLModel(contentsOf: url, configuration: configuration)
  }()

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
        setCancelled(false)
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            let response = try self.upscale(
              path: path,
              requestedScale: scale,
              preserveTransparency: preserveTransparency
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
        setCancelled(false)
        DispatchQueue.global(qos: .userInitiated).async {
          do {
            let response = try self.upscaleVideo(path: path, scale: scale)
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
    preserveTransparency: Bool
  ) throws -> [String: Any] {
    guard let image = UIImage(contentsOfFile: path) else {
      throw EngineError("decode_failed", "This image could not be decoded.")
    }
    return try upscale(
      image: image,
      requestedScale: requestedScale,
      preserveTransparency: preserveTransparency,
      reportsTileProgress: true
    )
  }

  private func upscale(
    image: UIImage,
    requestedScale: Int,
    preserveTransparency: Bool,
    reportsTileProgress: Bool
  ) throws -> [String: Any] {
    guard let source = normalizedRGBA(image) else {
      throw EngineError("decode_failed", "This image could not be decoded.")
    }
    let width = source.width
    let height = source.height
    let outputWidth = width * requestedScale
    let outputHeight = height * requestedScale
    guard outputWidth * outputHeight <= 40_000_000 else {
      throw EngineError(
        "image_too_large",
        "That output would exceed 40 megapixels. Choose 2× or use a smaller image."
      )
    }

    let xStarts = tileStarts(total: width)
    let yStarts = tileStarts(total: height)
    let totalTiles = xStarts.count * yStarts.count
    var output = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
    let inputArray = try MLMultiArray(
      shape: [1, 3, NSNumber(value: modelSize), NSNumber(value: modelSize)],
      dataType: .float32
    )
    var completed = 0

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
          tileHeight: tileHeight
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
        let prediction = try model.prediction(from: provider)
        guard
          let values = prediction.featureValue(for: "output")?.multiArrayValue
        else {
          throw EngineError("invalid_model", "The AI model returned an invalid result.")
        }

        guard
          values.dataType == .float32,
          values.shape.count == 4,
          values.shape[1].intValue == 3,
          values.shape[2].intValue == modelSize * nativeScale,
          values.shape[3].intValue == modelSize * nativeScale
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
          tileX: x0,
          tileY: y0,
          coreX0: coreX0,
          coreY0: coreY0,
          coreX1: coreX1,
          coreY1: coreY1,
          sourcePixels: source.pixels,
          sourceWidth: width,
          sourceHeight: height,
          preserveTransparency: preserveTransparency
        )
        completed += 1
        if reportsTileProgress {
          emitProgress(Double(completed) / Double(totalTiles))
        }
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
    let outputURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("aniscale_\(UUID().uuidString).png")
    guard
      let destination = CGImageDestinationCreateWithURL(
        outputURL as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
      )
    else {
      throw EngineError("encode_failed", "The enhanced image could not be saved.")
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw EngineError("encode_failed", "The enhanced image could not be saved.")
    }
    return [
      "path": outputURL.path,
      "originalWidth": width,
      "originalHeight": height,
      "engine": "Real-ESRGAN Anime 6B (Core ML)"
    ]
  }

  private func upscaleVideo(path: String, scale: Int) throws -> [String: Any] {
    let sourceURL = URL(fileURLWithPath: path)
    let asset = AVAsset(url: sourceURL)
    guard let videoTrack = asset.tracks(withMediaType: .video).first else {
      throw EngineError("video_decode_failed", "That file does not contain a readable video track.")
    }

    let transformedRect = CGRect(origin: .zero, size: videoTrack.naturalSize)
      .applying(videoTrack.preferredTransform)
    let sourceWidth = Int(abs(transformedRect.width).rounded())
    let sourceHeight = Int(abs(transformedRect.height).rounded())
    let requestedWidth = sourceWidth * scale
    let requestedHeight = sourceHeight * scale
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
    let bitrate = min(40_000_000, max(4_000_000, outputWidth * outputHeight * 4))
    let writerInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: outputWidth,
        AVVideoHeightKey: outputHeight,
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: bitrate,
          AVVideoExpectedSourceFrameRateKey: max(1, Int(videoTrack.nominalFrameRate.rounded()))
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
    // Use the 4× neural output when the final canvas retains at least 3× of
    // the source resolution. For 1080p sources, 2× already fills the 4K
    // hardware encoder limit and avoids generating a wasteful 8K frame first.
    let aiScale = outputWidth >= sourceWidth * 3 ? 4 : 2
    while let sample = readerOutput.copyNextSampleBuffer() {
      if isCancelled() {
        reader.cancelReading()
        writer.cancelWriting()
        try? FileManager.default.removeItem(at: silentURL)
        throw EngineError("cancelled", "Video upscaling was cancelled.")
      }
      guard let inputBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
      while !writerInput.isReadyForMoreMediaData {
        if isCancelled() { break }
        Thread.sleep(forTimeInterval: 0.004)
      }
      guard
        let pool = adaptor.pixelBufferPool
      else {
        throw EngineError("video_encode_failed", "The output pixel buffer pool is unavailable.")
      }
      var outputBuffer: CVPixelBuffer?
      guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &outputBuffer) == kCVReturnSuccess,
            let outputBuffer else {
        throw EngineError("video_memory", "The iPhone ran out of video processing memory.")
      }

      try autoreleasepool {
        var frame = CIImage(cvPixelBuffer: inputBuffer).transformed(by: videoTrack.preferredTransform)
        frame = frame.transformed(
          by: CGAffineTransform(translationX: -frame.extent.minX, y: -frame.extent.minY)
        )
        guard let frameImage = ciContext.createCGImage(frame, from: frame.extent) else {
          throw EngineError(
            "video_frame_decode",
            "A video frame could not be decoded for AI processing."
          )
        }
        let enhanced = try upscale(
          image: UIImage(cgImage: frameImage),
          requestedScale: aiScale,
          preserveTransparency: false,
          reportsTileProgress: false
        )
        guard let enhancedPath = enhanced["path"] as? String else {
          throw EngineError("video_frame_encode", "The AI engine returned no enhanced frame.")
        }
        let enhancedURL = URL(fileURLWithPath: enhancedPath)
        defer { try? FileManager.default.removeItem(at: enhancedURL) }
        guard var enhancedFrame = CIImage(contentsOf: enhancedURL) else {
          throw EngineError(
            "video_frame_encode",
            "An AI-enhanced frame could not be reopened."
          )
        }
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
      let timestamp = CMSampleBufferGetPresentationTimeStamp(sample)
      guard adaptor.append(outputBuffer, withPresentationTime: timestamp) else {
        throw EngineError("video_encode_failed", writer.error?.localizedDescription ?? "A video frame could not be encoded.")
      }
      emitProgress(min(0.92, max(0.01, CMTimeGetSeconds(timestamp) / durationSeconds * 0.92)))
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
    return [
      "path": finalURL.path,
      "outputWidth": outputWidth,
      "outputHeight": outputHeight,
      "durationSeconds": durationSeconds,
      "engine": "Real-ESRGAN Anime/3D (Core ML)"
    ]
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
    tileHeight: Int
  ) {
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
    let channelStride = array.strides[1].intValue
    let rowStride = array.strides[2].intValue
    let columnStride = array.strides[3].intValue
    for localY in 0..<modelSize {
      let sourceY = y0 + reflected(localY, length: tileHeight)
      for localX in 0..<modelSize {
        let sourceX = x0 + reflected(localX, length: tileWidth)
        let sourceIndex = (min(sourceY, sourceHeight - 1) * sourceWidth + min(sourceX, sourceWidth - 1)) * 4
        let alpha = max(Float32(pixels[sourceIndex + 3]) / 255, 1 / 255)
        let pixelOffset = localY * rowStride + localX * columnStride
        pointer[pixelOffset] = min(Float32(pixels[sourceIndex]) / 255 / alpha, 1)
        pointer[channelStride + pixelOffset] = min(Float32(pixels[sourceIndex + 1]) / 255 / alpha, 1)
        pointer[channelStride * 2 + pixelOffset] = min(Float32(pixels[sourceIndex + 2]) / 255 / alpha, 1)
      }
    }
  }

  private func copyOutput(
    _ array: MLMultiArray,
    into destination: inout [UInt8],
    outputWidth: Int,
    requestedScale: Int,
    tileX: Int,
    tileY: Int,
    coreX0: Int,
    coreY0: Int,
    coreX1: Int,
    coreY1: Int,
    sourcePixels: [UInt8],
    sourceWidth: Int,
    sourceHeight: Int,
    preserveTransparency: Bool
  ) {
    let downsample = nativeScale / requestedScale
    let channelStride = array.strides[1].intValue
    let rowStride = array.strides[2].intValue
    let columnStride = array.strides[3].intValue
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)

    func value(channel: Int, x: Int, y: Int) -> Float32 {
      let index = channel * channelStride + y * rowStride + x * columnStride
      return pointer[index]
    }

    for sourceY in coreY0..<coreY1 {
      for sourceX in coreX0..<coreX1 {
        for subY in 0..<requestedScale {
          for subX in 0..<requestedScale {
            let modelX = (sourceX - tileX) * nativeScale + subX * downsample
            let modelY = (sourceY - tileY) * nativeScale + subY * downsample
            var rgb = [Float32](repeating: 0, count: 3)
            for sampleY in 0..<downsample {
              for sampleX in 0..<downsample {
                for channel in 0..<3 {
                  rgb[channel] += value(
                    channel: channel,
                    x: modelX + sampleX,
                    y: modelY + sampleY
                  )
                }
              }
            }
            let divisor = Float32(downsample * downsample)
            let outputX = sourceX * requestedScale + subX
            let outputY = sourceY * requestedScale + subY
            let destinationIndex = (outputY * outputWidth + outputX) * 4
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
            destination[destinationIndex] = byte(rgb[0] / divisor * alpha)
            destination[destinationIndex + 1] = byte(rgb[1] / divisor * alpha)
            destination[destinationIndex + 2] = byte(rgb[2] / divisor * alpha)
            destination[destinationIndex + 3] = byte(alpha)
          }
        }
      }
    }
  }

  private func tileStarts(total: Int) -> [Int] {
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
