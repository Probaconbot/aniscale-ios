import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let upscaleEngine = UpscaleEngine()

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    let engineRegistrar = engineBridge.pluginRegistry.registrar(forPlugin: "AniScaleUpscaleEngine")!
    upscaleEngine.register(with: engineRegistrar.messenger())
    let glassRegistrar = engineBridge.pluginRegistry.registrar(forPlugin: "AniScaleNativeLiquidGlass")!
    glassRegistrar.register(
      NativeLiquidGlassFactory(messenger: glassRegistrar.messenger()),
      withId: "app.aniscale/native_liquid_glass"
    )
  }
}
