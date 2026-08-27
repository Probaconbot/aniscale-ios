import Flutter
import UIKit

final class NativeLiquidGlassFactory: NSObject, FlutterPlatformViewFactory {
  private let messenger: FlutterBinaryMessenger

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
    FlutterStandardMessageCodec.sharedInstance()
  }

  func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    NativeLiquidGlassView(frame: frame, arguments: args)
  }
}

final class NativeLiquidGlassView: NSObject, FlutterPlatformView {
  private let rootView: UIView

  init(frame: CGRect, arguments: Any?) {
    let parameters = arguments as? [String: Any]
    let radius = parameters?["cornerRadius"] as? Double ?? 18
    let tintValue = (parameters?["tint"] as? NSNumber)?.uint32Value ?? 0x99141725
    let interactive = parameters?["interactive"] as? Bool ?? true

    rootView = UIView(frame: frame)
    rootView.backgroundColor = .clear
    rootView.isUserInteractionEnabled = false
    rootView.layer.cornerRadius = radius
    rootView.layer.cornerCurve = .continuous
    rootView.clipsToBounds = true

    let effectView: UIVisualEffectView
    if #available(iOS 26.0, *) {
      let glass = UIGlassEffect(style: .regular)
      glass.isInteractive = interactive
      glass.tintColor = UIColor(argb: tintValue).withAlphaComponent(0.16)
      effectView = UIVisualEffectView(effect: glass)
    } else {
      effectView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterialDark))
      effectView.contentView.backgroundColor = UIColor(argb: tintValue).withAlphaComponent(0.24)
    }
    effectView.frame = rootView.bounds
    effectView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    effectView.isUserInteractionEnabled = false
    rootView.addSubview(effectView)
    super.init()
  }

  func view() -> UIView { rootView }
}

private extension UIColor {
  convenience init(argb: UInt32) {
    let alpha = CGFloat((argb >> 24) & 0xff) / 255
    let red = CGFloat((argb >> 16) & 0xff) / 255
    let green = CGFloat((argb >> 8) & 0xff) / 255
    let blue = CGFloat(argb & 0xff) / 255
    self.init(red: red, green: green, blue: blue, alpha: alpha)
  }
}
