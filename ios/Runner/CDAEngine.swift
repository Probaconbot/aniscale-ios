import CoreGraphics
import Foundation
import OnnxRuntimeBindings

final class CDAEngine {
  final class State {
    var width = 0
    var height = 0
    var low: NSMutableData?
    var high: NSMutableData?
    var framesSinceReset = 0

    func reset() {
      low = nil
      high = nil
      framesSinceReset = 0
    }
  }

  private let environment: ORTEnv
  private let initializer: ORTSession
  private let recurrent: ORTSession

  init() throws {
    environment = try ORTEnv(loggingLevel: .warning)
    let options = try ORTSessionOptions()
    try options.setIntraOpNumThreads(2)
    try options.setGraphOptimizationLevel(.all)
    if ORTIsCoreMLExecutionProviderAvailable() {
      let coreML = ORTCoreMLExecutionProviderOptions()
      coreML.enableOnSubgraphs = true
      coreML.createMLProgram = true
      try options.appendCoreMLExecutionProvider(with: coreML)
    }
    guard
      let initializerURL = Bundle.main.url(
        forResource: "AniRealism_cda_vsr_initializer",
        withExtension: "onnx"
      ),
      let recurrentURL = Bundle.main.url(
        forResource: "AniRealism_cda_vsr_recurrent",
        withExtension: "onnx"
      )
    else {
      throw NSError(
        domain: "AniScale.CDA",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "AniRealism test models are missing."]
      )
    }
    initializer = try ORTSession(
      env: environment,
      modelPath: initializerURL.path,
      sessionOptions: options
    )
    recurrent = try ORTSession(
      env: environment,
      modelPath: recurrentURL.path,
      sessionOptions: options
    )
  }

  func process(previous: CGImage?, current: CGImage, state: State) throws -> CGImage {
    let width = current.width
    let height = current.height
    if state.width != width || state.height != height {
      state.width = width
      state.height = height
      state.reset()
    }
    let frame = try rgbPlanes(current)
    let frameValue = try tensor(
      frame,
      shape: [1, 3, NSNumber(value: height), NSNumber(value: width)]
    )
    let outputs: [String: ORTValue]
    let names: Set<String> = ["output", "next_state_low", "next_state_high"]
    if let previous, let low = state.low, let high = state.high {
      let priors = try decodedPriors(previous: previous, current: current)
      let motionValue = try tensor(
        priors.motion,
        shape: [1, 2, NSNumber(value: height), NSNumber(value: width)]
      )
      let residualValue = try tensor(
        priors.residual,
        shape: [1, 1, NSNumber(value: height), NSNumber(value: width)]
      )
      let lowValue = try ORTValue(
        tensorData: low,
        elementType: .float,
        shape: [1, 64, NSNumber(value: height), NSNumber(value: width)]
      )
      let highValue = try ORTValue(
        tensorData: high,
        elementType: .float,
        shape: [1, 64, NSNumber(value: height), NSNumber(value: width)]
      )
      outputs = try recurrent.run(
        withInputs: [
          "frame": frameValue,
          "motion": motionValue,
          "residual": residualValue,
          "state_low": lowValue,
          "state_high": highValue
        ],
        outputNames: names,
        runOptions: nil
      )
    } else {
      outputs = try initializer.run(
        withInputs: ["frame": frameValue],
        outputNames: names,
        runOptions: nil
      )
    }
    guard
      let outputValue = outputs["output"],
      let lowValue = outputs["next_state_low"],
      let highValue = outputs["next_state_high"]
    else {
      throw NSError(
        domain: "AniScale.CDA",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: "CDA-VSR returned incomplete output."]
      )
    }
    state.low = try lowValue.tensorData()
    state.high = try highValue.tensorData()
    state.framesSinceReset += 1
    return try image(
      from: try outputValue.tensorData(),
      width: width * 4,
      height: height * 4
    )
  }

  private func tensor(_ values: [Float], shape: [NSNumber]) throws -> ORTValue {
    let data = values.withUnsafeBufferPointer { buffer in
      NSMutableData(
        bytes: buffer.baseAddress!,
        length: buffer.count * MemoryLayout<Float>.size
      )
    }
    return try ORTValue(tensorData: data, elementType: .float, shape: shape)
  }

  private func rgbPlanes(_ image: CGImage) throws -> [Float] {
    let width = image.width
    let height = image.height
    let count = width * height
    var rgba = [UInt8](repeating: 0, count: count * 4)
    guard let context = CGContext(
      data: &rgba,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue |
        CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
      throw NSError(domain: "AniScale.CDA", code: 3)
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    var planes = [Float](repeating: 0, count: count * 3)
    for index in 0..<count {
      planes[index] = Float(rgba[index * 4]) / 255
      planes[count + index] = Float(rgba[index * 4 + 1]) / 255
      planes[count * 2 + index] = Float(rgba[index * 4 + 2]) / 255
    }
    return planes
  }

  private func image(from data: NSMutableData, width: Int, height: Int) throws -> CGImage {
    let count = width * height
    guard data.length >= count * 3 * MemoryLayout<Float>.size else {
      throw NSError(domain: "AniScale.CDA", code: 4)
    }
    let source = data.bytes.bindMemory(to: Float.self, capacity: count * 3)
    var rgba = [UInt8](repeating: 255, count: count * 4)
    for index in 0..<count {
      rgba[index * 4] = UInt8((min(1, max(0, source[index])) * 255).rounded())
      rgba[index * 4 + 1] = UInt8((min(1, max(0, source[count + index])) * 255).rounded())
      rgba[index * 4 + 2] = UInt8((min(1, max(0, source[count * 2 + index])) * 255).rounded())
    }
    let output = Data(rgba)
    guard
      let provider = CGDataProvider(data: output as CFData),
      let result = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(
          rawValue: CGImageAlphaInfo.premultipliedLast.rawValue |
            CGBitmapInfo.byteOrder32Big.rawValue
        ),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
      )
    else {
      throw NSError(domain: "AniScale.CDA", code: 5)
    }
    return result
  }

  private func decodedPriors(
    previous: CGImage,
    current: CGImage
  ) throws -> (motion: [Float], residual: [Float]) {
    let width = current.width
    let height = current.height
    let plane = width * height
    let previousRGB = try rgbPlanes(previous)
    let currentRGB = try rgbPlanes(current)
    var previousLuma = [Float](repeating: 0, count: plane)
    var currentLuma = [Float](repeating: 0, count: plane)
    for index in 0..<plane {
      previousLuma[index] = 0.2126 * previousRGB[index]
        + 0.7152 * previousRGB[plane + index]
        + 0.0722 * previousRGB[plane * 2 + index]
      currentLuma[index] = 0.2126 * currentRGB[index]
        + 0.7152 * currentRGB[plane + index]
        + 0.0722 * currentRGB[plane * 2 + index]
    }
    var motion = [Float](repeating: 0, count: plane * 2)
    var residual = [Float](repeating: 0, count: plane)
    let block = 16
    let radius = 4
    let sampleStride = 4
    for originY in stride(from: 0, to: height, by: block) {
      for originX in stride(from: 0, to: width, by: block) {
        var bestX = 0
        var bestY = 0
        var bestScore = Float.infinity
        for deltaY in -radius...radius {
          for deltaX in -radius...radius {
            var score: Float = 0
            var samples = 0
            for sampleY in stride(from: 0, to: block, by: sampleStride) {
              let y = originY + sampleY
              if y >= height { continue }
              for sampleX in stride(from: 0, to: block, by: sampleStride) {
                let x = originX + sampleX
                if x >= width { continue }
                let referenceX = x + deltaX
                let referenceY = y + deltaY
                if referenceX >= 0 && referenceX < width && referenceY >= 0 && referenceY < height {
                  score += abs(
                    currentLuma[y * width + x] - previousLuma[referenceY * width + referenceX]
                  )
                } else {
                  score += 1
                }
                samples += 1
              }
            }
            score /= Float(max(1, samples))
            let magnitude = abs(deltaX) + abs(deltaY)
            let bestMagnitude = abs(bestX) + abs(bestY)
            if score < bestScore - 0.0000001 ||
              (abs(score - bestScore) <= 0.0000001 && magnitude < bestMagnitude) {
              bestScore = score
              bestX = deltaX
              bestY = deltaY
            }
          }
        }
        for y in originY..<min(height, originY + block) {
          for x in originX..<min(width, originX + block) {
            let index = y * width + x
            motion[index] = Float(bestX)
            motion[plane + index] = Float(bestY)
            let referenceX = min(width - 1, max(0, x + bestX))
            let referenceY = min(height - 1, max(0, y + bestY))
            residual[index] = abs(
              currentLuma[index] - previousLuma[referenceY * width + referenceX]
            )
          }
        }
      }
    }
    return (motion, residual)
  }
}
