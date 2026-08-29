package app.aniscale.aniscale

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.os.Build
import android.os.PowerManager
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class MainActivity : FlutterActivity() {
    companion object {
        init {
            System.loadLibrary("aniscale_ai")
        }

        @JvmStatic external fun nativeInitialize(assetManager: AssetManager): Boolean
        @JvmStatic external fun nativeUsesVulkan(): Boolean
        @JvmStatic external fun nativeCancel()
        @JvmStatic external fun nativeUpscale(
            bitmap: Bitmap,
            engine: String,
            scale: Int,
            tileSize: Int,
            sharpening: Float,
        ): Bitmap?
        @JvmStatic external fun nativeFillYuv420(
            bitmap: Bitmap,
            yBuffer: ByteBuffer,
            yRowStride: Int,
            yPixelStride: Int,
            uBuffer: ByteBuffer,
            uRowStride: Int,
            uPixelStride: Int,
            vBuffer: ByteBuffer,
            vRowStride: Int,
            vPixelStride: Int,
        )
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val cancelled = AtomicBoolean(false)
    private var progressSink: EventChannel.EventSink? = null
    @Volatile private var initialized = false
    private val ultraEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val ultraSession by lazy {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val model = assets.open("models/AniUltraScale_experimental_2x.onnx").use {
            it.readBytes()
        }
        ultraEnvironment.createSession(model, options)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        MethodChannel(messenger, "app.aniscale/upscaler").setMethodCallHandler(::handleMethod)
        EventChannel(messenger, "app.aniscale/upscaler_progress").setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    progressSink = events
                }

                override fun onCancel(arguments: Any?) {
                    progressSink = null
                }
            },
        )
    }

    override fun onDestroy() {
        cancelled.set(true)
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun handleMethod(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "cancel" -> {
                cancelled.set(true)
                nativeCancel()
                result.success(null)
            }
            "upscaleImage" -> runImage(call, result)
            "upscaleVideo" -> runVideo(call, result)
            else -> result.notImplemented()
        }
    }

    private fun ensureModels() {
        if (!initialized) initialized = nativeInitialize(assets)
        check(initialized) { "AniScale Android AI models could not be loaded." }
    }

    private fun runImage(call: MethodCall, result: MethodChannel.Result) {
        val path = call.argument<String>("path")
        val scale = call.argument<Int>("scale") ?: 2
        val tileSize = call.argument<Int>("tileSize") ?: 192
        val outputFormat = call.argument<String>("outputFormat") ?: "automatic"
        val engine = call.argument<String>("engine") ?: "fusion"
        val performance = call.argument<Int>("performance") ?: 0
        val sharpness = call.argument<Double>("sharpness") ?: 0.2
        val detail = call.argument<Double>("detail") ?: 0.5
        if (path.isNullOrBlank() || scale !in listOf(2, 4) ||
            engine !in listOf("fusion", "render", "turbo")) {
            result.error("bad_arguments", "Invalid image path or scale.", null)
            return
        }
        cancelled.set(false)
        worker.execute {
            try {
                ensureModels()
                emitProgress(0.04)
                val decoded = BitmapFactory.decodeFile(path)
                    ?: error("This image could not be decoded.")
                val originalWidth = decoded.width
                val originalHeight = decoded.height
                val outputPixelLimit = when (performance) {
                    1 -> 18_000_000
                    2 -> 8_847_360
                    else -> 12_000_000
                }
                val requestedEdgeLimit = if (performance == 2) 1600 else null
                val inputEdgeLimit = if (nativeUsesVulkan()) {
                    requestedEdgeLimit
                } else {
                    min(requestedEdgeLimit ?: 1024, 1024)
                }
                val source = fitInput(decoded, scale, outputPixelLimit, inputEdgeLimit)
                if (source !== decoded) decoded.recycle()
                val input = source.copy(Bitmap.Config.ARGB_8888, false)
                if (source !== input) source.recycle()
                emitProgress(0.12)
                val sharpenAmount = (sharpness * 0.8 + detail * 0.22)
                    .coerceIn(0.08, 0.42).toFloat()
                val enhanced = nativeUpscale(
                    input,
                    engine,
                    scale,
                    tileSize,
                    sharpenAmount,
                ) ?: error("The selected AniScale model returned no image.")
                input.recycle()
                check(!cancelled.get()) { "Image upscaling was cancelled." }
                val usePng = outputFormat == "png" || outputFormat == "automatic"
                val extension = if (usePng) "png" else "jpg"
                val output = File(
                    enhancementDirectory(),
                    "aniscale_${System.currentTimeMillis()}.$extension",
                )
                FileOutputStream(output).use { stream ->
                    enhanced.compress(
                        if (usePng) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG,
                        if (usePng) 100 else 96,
                        stream,
                    )
                }
                val response = hashMapOf<String, Any>(
                    "path" to output.absolutePath,
                    "originalWidth" to originalWidth,
                    "originalHeight" to originalHeight,
                    "outputWidth" to enhanced.width,
                    "outputHeight" to enhanced.height,
                    "engine" to "${engineLabel(engine)} (${backendLabel(engine)})",
                )
                enhanced.recycle()
                emitProgress(1.0)
                runOnUiThread { result.success(response) }
            } catch (error: Throwable) {
                runOnUiThread {
                    result.error("android_image_engine_failed", error.message, null)
                }
            }
        }
    }

    private fun runVideo(call: MethodCall, result: MethodChannel.Result) {
        val path = call.argument<String>("path")
        val scale = call.argument<Int>("scale") ?: 2
        val targetScale = call.argument<Number>("targetScale")?.toDouble() ?: scale.toDouble()
        val efficient = call.argument<Boolean>("efficient") ?: true
        val tileSize = call.argument<Int>("tileSize") ?: 192
        val engine = call.argument<String>("engine") ?: "fusion"
        val content = call.argument<String>("content") ?: "auto"
        val detailMode = call.argument<String>("detailMode") ?: "natural"
        val codec = call.argument<String>("codec") ?: "hevc"
        if (path.isNullOrBlank() || scale !in listOf(2, 4) ||
            targetScale !in listOf(1.5, 2.0, 3.0, 4.0) ||
            engine !in listOf("fusion", "render", "turbo", "clean", "ultra", "superUltra") ||
            content !in listOf("auto", "live", "anime") ||
            detailMode !in listOf("natural", "detailed", "sharp") ||
            codec !in listOf("hevc", "h264")) {
            result.error("bad_arguments", "Invalid video request.", null)
            return
        }
        cancelled.set(false)
        worker.execute {
            try {
                when (engine) {
                    "ultra" -> ultraSession
                    "clean" -> Unit
                    else -> ensureModels()
                }
                val response = processVideo(
                    path,
                    scale,
                    targetScale,
                    efficient,
                    tileSize,
                    engine,
                    content,
                    detailMode,
                    codec,
                )
                runOnUiThread { result.success(response) }
            } catch (error: Throwable) {
                runOnUiThread {
                    result.error("android_video_engine_failed", error.message, null)
                }
            }
        }
    }

    private fun processVideo(
        path: String,
        scale: Int,
        targetScale: Double,
        efficient: Boolean,
        tileSize: Int,
        engine: String,
        content: String,
        detailMode: String,
        codec: String,
    ): Map<String, Any> {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(path)
        val durationMs = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_DURATION,
        )?.toLongOrNull() ?: error("The video duration could not be read.")
        val fps = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE,
        )?.toDoubleOrNull()?.coerceIn(1.0, 60.0) ?: 30.0
        val metadataFrames = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT,
        )?.toIntOrNull()
        val frameCount = (metadataFrames ?: ((durationMs / 1000.0) * fps).roundToInt())
            .coerceAtLeast(1)
        val first = frameAt(retriever, 0, fps)
            ?: error("The first video frame could not be decoded.")
        val sourceWidth = first.width
        val sourceHeight = first.height
        val target = fitOutput(sourceWidth, sourceHeight, targetScale)
        val outputWidth = target.first
        val outputHeight = target.second
        val maxInputEdge = if (!nativeUsesVulkan()) {
            if (engine == "turbo") 480 else 384
        } else when (engine) {
            "ultra" -> 320
            "superUltra" -> if (efficient) 720 else 960
            "turbo" -> if (efficient) 720 else 960
            "render" -> if (efficient) 640 else 896
            else -> if (efficient) 768 else 960
        }
        val fittedFirst = fitInput(first, scale, 8_847_360, maxInputEdge)
        if (fittedFirst !== first) first.recycle()
        val inputWidth = fittedFirst.width
        val inputHeight = fittedFirst.height
        val output = File(
            enhancementDirectory(),
            "aniscale_video_${System.currentTimeMillis()}.mp4",
        )
        if (output.exists()) output.delete()

        val audioExtractor = MediaExtractor()
        audioExtractor.setDataSource(path)
        var sourceAudioTrack = -1
        var audioFormat: MediaFormat? = null
        for (index in 0 until audioExtractor.trackCount) {
            val candidate = audioExtractor.getTrackFormat(index)
            if (candidate.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                sourceAudioTrack = index
                audioFormat = candidate
                break
            }
        }

        val requestedMime = if (codec == "hevc") {
            MediaFormat.MIMETYPE_VIDEO_HEVC
        } else {
            MediaFormat.MIMETYPE_VIDEO_AVC
        }
        val encoderMime = if (hasHardwareEncoder(requestedMime)) {
            requestedMime
        } else {
            MediaFormat.MIMETYPE_VIDEO_AVC
        }
        val format = MediaFormat.createVideoFormat(
            encoderMime,
            outputWidth,
            outputHeight,
        ).apply {
            setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
            )
            setInteger(MediaFormat.KEY_FRAME_RATE, fps.roundToInt())
            setInteger(
                MediaFormat.KEY_BIT_RATE,
                min(
                    60_000_000,
                    max(
                        4_000_000,
                        (outputWidth * outputHeight * fps *
                            if (encoderMime == MediaFormat.MIMETYPE_VIDEO_HEVC) 0.11 else 0.18
                        ).roundToInt(),
                    ),
                ),
            )
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val encoder = MediaCodec.createEncoderByType(encoderMime)
        val muxer = MediaMuxer(output.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val muxState = MuxState()
        try {
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            var sourceFrame: Bitmap? = fittedFirst
            for (frameIndex in 0 until frameCount) {
                check(!cancelled.get()) { "Video upscaling was cancelled." }
                val decoded = sourceFrame ?: frameAt(retriever, frameIndex, fps)
                sourceFrame = null
                if (decoded == null) continue
                val fitted = if (decoded.width == inputWidth && decoded.height == inputHeight) {
                    decoded
                } else {
                    Bitmap.createScaledBitmap(decoded, inputWidth, inputHeight, true).also {
                        decoded.recycle()
                    }
                }
                val prepared = if (engine == "clean") cleanupFrame(fitted) else fitted
                if (prepared !== fitted) fitted.recycle()
                val rgba = prepared.copy(Bitmap.Config.ARGB_8888, false)
                prepared.recycle()
                val modelSharpness = when (engine) {
                    "superUltra" -> when (detailMode) {
                        "sharp" -> 0.16f
                        "detailed" -> 0.08f
                        else -> 0.0f
                    }
                    "render" -> 0.18f
                    "turbo" -> 0.22f
                    else -> 0.28f
                }
                val enhanced = when (engine) {
                    "ultra" -> upscaleUltraExperimental(rgba)
                    // Avoid neural re-amplification of residual scanlines.
                    "clean" -> Bitmap.createScaledBitmap(
                        rgba,
                        outputWidth,
                        outputHeight,
                        true,
                    )
                    "superUltra" -> nativeUpscale(
                        rgba,
                        if (content == "anime") "superultra_anime" else "superultra_live",
                        scale,
                        tileSize,
                        modelSharpness,
                    )
                    else -> nativeUpscale(
                        rgba,
                        engine,
                        scale,
                        tileSize,
                        modelSharpness,
                    )
                } ?: error("The selected AniScale model failed on frame ${frameIndex + 1}.")
                rgba.recycle()
                val outputFrame = if (
                    enhanced.width == outputWidth && enhanced.height == outputHeight
                ) {
                    enhanced
                } else {
                    Bitmap.createScaledBitmap(enhanced, outputWidth, outputHeight, true).also {
                        enhanced.recycle()
                    }
                }
                queueFrame(encoder, outputFrame, frameIndex, fps, muxer, muxState, audioFormat)
                outputFrame.recycle()
                applyThermalBackoff(frameIndex)
                emitProgress(0.02 + (frameIndex + 1).toDouble() / frameCount * 0.92)
            }
            queueEndOfStream(encoder, frameCount, fps, muxer, muxState, audioFormat)
            if (sourceAudioTrack >= 0 && muxState.audioTrack >= 0) {
                copyAudio(audioExtractor, sourceAudioTrack, muxer, muxState.audioTrack)
            }
            check(muxState.started) { "The Android video encoder produced no output." }
        } finally {
            retriever.release()
            audioExtractor.release()
            try {
                encoder.stop()
            } catch (_: Throwable) {
            }
            encoder.release()
            if (muxState.started) {
                try {
                    muxer.stop()
                } catch (_: Throwable) {
                }
            }
            muxer.release()
        }
        check(!cancelled.get()) { "Video upscaling was cancelled." }
        emitProgress(1.0)
        return hashMapOf(
            "path" to output.absolutePath,
            "originalWidth" to sourceWidth,
            "originalHeight" to sourceHeight,
            "outputWidth" to outputWidth,
            "outputHeight" to outputHeight,
            "durationSeconds" to durationMs / 1000.0,
            "engine" to if (engine == "superUltra") {
                "SuperUltra — ${contentLabel(content)}, ${detailMode.replaceFirstChar { it.uppercase() }} (${backendLabel(engine)})"
            } else {
                "${engineLabel(engine)} (${backendLabel(engine)})"
            },
        )
    }

    private fun enhancementDirectory(): File = File(filesDir, "enhancements").apply {
        check(exists() || mkdirs()) { "AniScale could not create its local history folder." }
    }

    private fun backendLabel(engine: String? = null): String = if (engine == "ultra") {
        "ONNX Runtime — untrained"
    } else if (engine == "superUltra") {
        if (nativeUsesVulkan()) "SPAN ncnn Vulkan FP16" else "SPAN ncnn CPU"
    } else if (engine == "clean") {
        "Native restoration + Lanczos"
    } else if (nativeUsesVulkan()) {
        "Android ncnn Vulkan FP16"
    } else {
        "Android ncnn CPU"
    }

    private fun engineLabel(engine: String): String = when (engine) {
        "render" -> "AniScale Render"
        "turbo" -> "AniScale Turbo"
        "clean" -> "AniScale Clean"
        "ultra" -> "AniUltraScale Experimental"
        "superUltra" -> "SuperUltra"
        else -> "AniScale Fusion"
    }

    private fun contentLabel(content: String): String = when (content) {
        "anime" -> "Anime"
        "live" -> "Live Action"
        else -> "Auto"
    }

    private fun hasHardwareEncoder(mime: String): Boolean = try {
        MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.any { info ->
            info.isEncoder && info.supportedTypes.any { it.equals(mime, ignoreCase = true) } &&
                !info.name.startsWith("OMX.google", ignoreCase = true) &&
                !info.name.startsWith("c2.android", ignoreCase = true)
        }
    } catch (_: Throwable) {
        false
    }

    private fun applyThermalBackoff(frameIndex: Int) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || frameIndex % 6 != 0) return
        val power = getSystemService(PowerManager::class.java)
        val pauseMs = when {
            power.currentThermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL -> 80L
            power.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 45L
            power.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> 15L
            else -> 0L
        }
        if (pauseMs > 0) Thread.sleep(pauseMs)
    }

    /**
     * Mobile-safe cleanup pass for patterned/green-cast footage. It blends a
     * vertical three-tap filter to suppress scanlines, neutralizes excessive
     * green, and applies restrained local contrast before Fusion upscaling.
     */
    private fun cleanupFrame(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val source = IntArray(width * height)
        val horizontal = IntArray(source.size)
        val output = IntArray(source.size)
        bitmap.getPixels(source, 0, width, 0, 0, width, height)
        // First remove the vertical display-line component with a five-tap
        // horizontal low-pass. Two passes are faster than a full 5x3 kernel.
        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x
                val p2l = source[y * width + max(0, x - 2)]
                val p1l = source[y * width + max(0, x - 1)]
                val p = source[index]
                val p1r = source[y * width + min(width - 1, x + 1)]
                val p2r = source[y * width + min(width - 1, x + 2)]
                val r = (((p shr 16) and 255) * 2 + ((p1l shr 16) and 255) * 2 + ((p1r shr 16) and 255) * 2 + ((p2l shr 16) and 255) + ((p2r shr 16) and 255)) / 8
                val g = (((p shr 8) and 255) * 2 + ((p1l shr 8) and 255) * 2 + ((p1r shr 8) and 255) * 2 + ((p2l shr 8) and 255) + ((p2r shr 8) and 255)) / 8
                val b = ((p and 255) * 2 + (p1l and 255) * 2 + (p1r and 255) * 2 + (p2l and 255) + (p2r and 255)) / 8
                horizontal[index] = (p and -0x1000000) or (r shl 16) or (g shl 8) or b
            }
        }
        // Then suppress horizontal scanlines and perform green-cast removal.
        for (y in 0 until height) {
            val above = max(0, y - 1)
            val below = min(height - 1, y + 1)
            for (x in 0 until width) {
                val index = y * width + x
                val p = horizontal[index]
                val pa = horizontal[above * width + x]
                val pb = horizontal[below * width + x]
                var r = (((p shr 16) and 255) * 2 + ((pa shr 16) and 255) + ((pb shr 16) and 255)) / 4
                var g = (((p shr 8) and 255) * 2 + ((pa shr 8) and 255) + ((pb shr 8) and 255)) / 4
                var b = ((p and 255) * 2 + (pa and 255) + (pb and 255)) / 4
                val neutral = (r + b) / 2
                if (g > neutral) g = (neutral + (g - neutral) * 0.12f).roundToInt()
                r = ((r - 128) * 1.03f + 132).roundToInt().coerceIn(0, 255)
                g = ((g - 128) * 1.01f + 126).roundToInt().coerceIn(0, 255)
                b = ((b - 128) * 1.03f + 133).roundToInt().coerceIn(0, 255)
                output[index] = (source[index] and -0x1000000) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(output, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun upscaleUltraExperimental(bitmap: Bitmap): Bitmap {
        val width = 320
        val height = 180
        val outputWidth = width * 2
        val outputHeight = height * 2
        val source = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)
        if (source !== bitmap) source.recycle()
        val plane = width * height
        val input = FloatArray(5 * 3 * plane)
        for (frame in 0 until 5) {
            val frameOffset = frame * 3 * plane
            for (index in pixels.indices) {
                val pixel = pixels[index]
                input[frameOffset + index] = ((pixel shr 16) and 0xff) / 255f
                input[frameOffset + plane + index] = ((pixel shr 8) and 0xff) / 255f
                input[frameOffset + 2 * plane + index] = (pixel and 0xff) / 255f
            }
        }
        OnnxTensor.createTensor(
            ultraEnvironment,
            FloatBuffer.wrap(input),
            longArrayOf(1, 5, 3, height.toLong(), width.toLong()),
        ).use { framesTensor ->
            OnnxTensor.createTensor(
                ultraEnvironment,
                FloatBuffer.wrap(floatArrayOf(1f, 0.82f)),
                longArrayOf(1, 2),
            ).use { controlsTensor ->
                ultraSession.run(
                    mapOf("frames" to framesTensor, "controls" to controlsTensor),
                ).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<Array<Array<FloatArray>>>
                    val channels = output[0]
                    val outputPixels = IntArray(outputWidth * outputHeight)
                    for (y in 0 until outputHeight) {
                        for (x in 0 until outputWidth) {
                            val red = (channels[0][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                            val green = (channels[1][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                            val blue = (channels[2][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                            outputPixels[y * outputWidth + x] =
                                (0xff shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    }
                    return Bitmap.createBitmap(
                        outputPixels,
                        outputWidth,
                        outputHeight,
                        Bitmap.Config.ARGB_8888,
                    )
                }
            }
        }
    }

    private fun fitOutput(width: Int, height: Int, scale: Double): Pair<Int, Int> {
        val requestedWidth = width * scale
        val requestedHeight = height * scale
        val edgeRatio = 3840.0 / max(requestedWidth, requestedHeight)
        val pixelRatio = sqrt(8_847_360.0 / (requestedWidth.toDouble() * requestedHeight))
        val ratio = min(1.0, min(edgeRatio, pixelRatio))
        val outputWidth = max(2, ((requestedWidth * ratio).roundToInt() / 2) * 2)
        val outputHeight = max(2, ((requestedHeight * ratio).roundToInt() / 2) * 2)
        return outputWidth to outputHeight
    }

    private fun frameAt(
        retriever: MediaMetadataRetriever,
        index: Int,
        fps: Double,
    ): Bitmap? = try {
        retriever.getFrameAtIndex(index)
    } catch (_: Throwable) {
        retriever.getFrameAtTime(
            (index * 1_000_000.0 / fps).toLong(),
            MediaMetadataRetriever.OPTION_CLOSEST,
        )
    }

    private fun fitInput(
        bitmap: Bitmap,
        scale: Int,
        outputPixelLimit: Int,
        inputEdgeLimit: Int?,
    ): Bitmap {
        val requestedPixels = bitmap.width.toDouble() * bitmap.height * scale * scale
        val pixelRatio = sqrt(outputPixelLimit / requestedPixels).coerceAtMost(1.0)
        val edgeRatio = inputEdgeLimit?.let {
            it.toDouble() / max(bitmap.width, bitmap.height)
        }?.coerceAtMost(1.0) ?: 1.0
        val ratio = min(pixelRatio, edgeRatio)
        if (ratio >= 0.999) return bitmap
        var width = max(2, (bitmap.width * ratio).roundToInt())
        var height = max(2, (bitmap.height * ratio).roundToInt())
        if (width % 2 != 0) width--
        if (height % 2 != 0) height--
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private data class MuxState(
        var videoTrack: Int = -1,
        var audioTrack: Int = -1,
        var started: Boolean = false,
        var endReached: Boolean = false,
    )

    private fun queueFrame(
        encoder: MediaCodec,
        bitmap: Bitmap,
        frameIndex: Int,
        fps: Double,
        muxer: MediaMuxer,
        state: MuxState,
        audioFormat: MediaFormat?,
    ) {
        while (true) {
            val inputIndex = encoder.dequeueInputBuffer(10_000)
            if (inputIndex >= 0) {
                val image = encoder.getInputImage(inputIndex)
                    ?: error("The Android encoder did not provide a YUV input image.")
                fillYuv420(bitmap, image)
                image.close()
                val presentationTime = (frameIndex * 1_000_000.0 / fps).toLong()
                encoder.queueInputBuffer(
                    inputIndex,
                    0,
                    bitmap.width * bitmap.height * 3 / 2,
                    presentationTime,
                    0,
                )
                break
            }
            drainEncoder(encoder, muxer, state, audioFormat, false)
        }
        drainEncoder(encoder, muxer, state, audioFormat, false)
    }

    private fun queueEndOfStream(
        encoder: MediaCodec,
        frameCount: Int,
        fps: Double,
        muxer: MediaMuxer,
        state: MuxState,
        audioFormat: MediaFormat?,
    ) {
        while (true) {
            val index = encoder.dequeueInputBuffer(10_000)
            if (index >= 0) {
                encoder.queueInputBuffer(
                    index,
                    0,
                    0,
                    (frameCount * 1_000_000.0 / fps).toLong(),
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                )
                break
            }
            drainEncoder(encoder, muxer, state, audioFormat, false)
        }
        while (!state.endReached) {
            drainEncoder(encoder, muxer, state, audioFormat, true)
        }
    }

    private fun drainEncoder(
        encoder: MediaCodec,
        muxer: MediaMuxer,
        state: MuxState,
        audioFormat: MediaFormat?,
        wait: Boolean,
    ) {
        val info = MediaCodec.BufferInfo()
        while (true) {
            val index = encoder.dequeueOutputBuffer(info, if (wait) 10_000 else 0)
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> return
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    check(!state.started) { "The video output format changed twice." }
                    state.videoTrack = muxer.addTrack(encoder.outputFormat)
                    if (audioFormat != null) state.audioTrack = muxer.addTrack(audioFormat)
                    muxer.start()
                    state.started = true
                }
                index >= 0 -> {
                    val buffer = encoder.getOutputBuffer(index)
                    if (buffer != null && info.size > 0 &&
                        info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        check(state.started) { "The video muxer was not started." }
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        muxer.writeSampleData(state.videoTrack, buffer, info)
                    }
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        state.endReached = true
                    }
                    encoder.releaseOutputBuffer(index, false)
                    if (state.endReached) return
                }
            }
        }
    }

    private fun fillYuv420(bitmap: Bitmap, image: android.media.Image) {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        nativeFillYuv420(
            bitmap,
            yPlane.buffer,
            yPlane.rowStride,
            yPlane.pixelStride,
            uPlane.buffer,
            uPlane.rowStride,
            uPlane.pixelStride,
            vPlane.buffer,
            vPlane.rowStride,
            vPlane.pixelStride,
        )
    }

    private fun copyAudio(
        extractor: MediaExtractor,
        sourceTrack: Int,
        muxer: MediaMuxer,
        destinationTrack: Int,
    ) {
        extractor.selectTrack(sourceTrack)
        val format = extractor.getTrackFormat(sourceTrack)
        val capacity = if (format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
            max(1_048_576, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE))
        } else 1_048_576
        val buffer = ByteBuffer.allocateDirect(capacity)
        val info = MediaCodec.BufferInfo()
        while (!cancelled.get()) {
            buffer.clear()
            val size = extractor.readSampleData(buffer, 0)
            if (size < 0) break
            info.set(0, size, extractor.sampleTime, extractor.sampleFlags)
            muxer.writeSampleData(destinationTrack, buffer, info)
            extractor.advance()
        }
    }

    private fun emitProgress(value: Double) {
        runOnUiThread { progressSink?.success(value.coerceIn(0.0, 1.0)) }
    }
}
