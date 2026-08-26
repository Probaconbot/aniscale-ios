import CoreML
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
    guard let image = UIImage(contentsOfFile: path), let source = normalizedRGBA(image) else {
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
          let outputName = model.modelDescription.outputDescriptionsByName.keys.first,
          let values = prediction.featureValue(for: outputName)?.multiArrayValue
        else {
          throw EngineError("invalid_model", "The AI model returned an invalid result.")
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
        emitProgress(Double(completed) / Double(totalTiles))
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
    let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: modelSize * modelSize * 3)
    let plane = modelSize * modelSize
    for localY in 0..<modelSize {
      let sourceY = y0 + reflected(localY, length: tileHeight)
      for localX in 0..<modelSize {
        let sourceX = x0 + reflected(localX, length: tileWidth)
        let sourceIndex = (min(sourceY, sourceHeight - 1) * sourceWidth + min(sourceX, sourceWidth - 1)) * 4
        let destinationIndex = localY * modelSize + localX
        let alpha = max(Float32(pixels[sourceIndex + 3]) / 255, 1 / 255)
        pointer[destinationIndex] = min(Float32(pixels[sourceIndex]) / 255 / alpha, 1)
        pointer[plane + destinationIndex] = min(Float32(pixels[sourceIndex + 1]) / 255 / alpha, 1)
        pointer[plane * 2 + destinationIndex] = min(Float32(pixels[sourceIndex + 2]) / 255 / alpha, 1)
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
    let modelOutputSize = modelSize * nativeScale
    let plane = modelOutputSize * modelOutputSize
    let downsample = nativeScale / requestedScale

    func value(channel: Int, x: Int, y: Int) -> Float32 {
      let index = channel * plane + y * modelOutputSize + x
      let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: plane * 3)
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
