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
    private val animeEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val animeSession by lazy {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        val model = assets.open("models/AniUltraAnime_v2_recurrent.onnx").use {
            it.readBytes()
        }
        animeEnvironment.createSession(model, options)
    }
    private fun cdaSession(assetName: String): OrtSession {
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        val model = assets.open("models/$assetName").use { it.readBytes() }
        return animeEnvironment.createSession(model, options)
    }
    private val cdaInitializerSession by lazy {
        cdaSession("AniRealism_cda_vsr_initializer.onnx")
    }
    private val cdaRecurrentSession by lazy {
        cdaSession("AniRealism_cda_vsr_recurrent.onnx")
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
            engine !in listOf("fusion", "render", "turbo", "superUltra", "animeUltra", "realism") ||
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
                    "animeUltra" -> animeSession
                    "realism" -> {
                        cdaInitializerSession
                        cdaRecurrentSession
                    }
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
            "animeUltra" -> if (efficient) 360 else 480
            "realism" -> if (efficient) 384 else 480
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
            var sourceFrame: Bitmap? = if (engine == "animeUltra") null else fittedFirst
            var previousAnimeFrame: Bitmap? = null
            var currentAnimeFrame: Bitmap? = if (engine == "animeUltra") fittedFirst else null
            var nextAnimeFrame: Bitmap? = if (engine == "animeUltra" && frameCount > 1) {
                frameAt(retriever, 1, fps)?.let { decoded ->
                    if (decoded.width == inputWidth && decoded.height == inputHeight) {
                        decoded
                    } else {
                        Bitmap.createScaledBitmap(decoded, inputWidth, inputHeight, true).also {
                            decoded.recycle()
                        }
                    }
                }
            } else {
                null
            }
            val animeState = AnimeRuntimeState()
            val cdaState = CdaRuntimeState()
            var previousCdaFrame: Bitmap? = null
            for (frameIndex in 0 until frameCount) {
                check(!cancelled.get()) { "Video upscaling was cancelled." }
                if (engine == "animeUltra") {
                    val current = currentAnimeFrame ?: break
                    val next = nextAnimeFrame ?: current
                    val previous = previousAnimeFrame ?: current
                    if (previousAnimeFrame != null && isSceneCut(previous, current)) {
                        animeState.reset()
                    }
                    val enhanced = upscaleAnime(previous, current, next, animeState)
                    previousAnimeFrame?.recycle()
                    previousAnimeFrame = current
                    currentAnimeFrame = nextAnimeFrame
                    nextAnimeFrame = if (frameIndex + 2 < frameCount) {
                        frameAt(retriever, frameIndex + 2, fps)?.let { decoded ->
                            if (decoded.width == inputWidth && decoded.height == inputHeight) {
                                decoded
                            } else {
                                Bitmap.createScaledBitmap(
                                    decoded,
                                    inputWidth,
                                    inputHeight,
                                    true,
                                ).also { decoded.recycle() }
                            }
                        }
                    } else {
                        null
                    }
                    val outputFrame = if (
                        enhanced.width == outputWidth && enhanced.height == outputHeight
                    ) {
                        enhanced
                    } else {
                        Bitmap.createScaledBitmap(
                            enhanced,
                            outputWidth,
                            outputHeight,
                            true,
                        ).also { enhanced.recycle() }
                    }
                    queueFrame(
                        encoder,
                        outputFrame,
                        frameIndex,
                        fps,
                        muxer,
                        muxState,
                        audioFormat,
                    )
                    outputFrame.recycle()
                    applyThermalBackoff(frameIndex)
                    emitProgress(0.02 + (frameIndex + 1).toDouble() / frameCount * 0.92)
                    continue
                }
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
                val rgba = fitted.copy(Bitmap.Config.ARGB_8888, false)
                fitted.recycle()
                if (engine == "realism") {
                    if (previousCdaFrame != null && isSceneCut(previousCdaFrame!!, rgba)) {
                        cdaState.reset()
                    }
                    if (cdaState.framesSinceReset >= 25) cdaState.reset()
                    val enhanced = upscaleCda(previousCdaFrame, rgba, cdaState)
                    previousCdaFrame?.recycle()
                    previousCdaFrame = rgba
                    val outputFrame = if (
                        enhanced.width == outputWidth && enhanced.height == outputHeight
                    ) {
                        enhanced
                    } else {
                        Bitmap.createScaledBitmap(
                            enhanced,
                            outputWidth,
                            outputHeight,
                            true,
                        ).also { enhanced.recycle() }
                    }
                    queueFrame(
                        encoder,
                        outputFrame,
                        frameIndex,
                        fps,
                        muxer,
                        muxState,
                        audioFormat,
                    )
                    outputFrame.recycle()
                    applyThermalBackoff(frameIndex)
                    emitProgress(0.02 + (frameIndex + 1).toDouble() / frameCount * 0.92)
                    continue
                }
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
            listOfNotNull(previousAnimeFrame, currentAnimeFrame, nextAnimeFrame)
                .distinctBy { System.identityHashCode(it) }
                .forEach { if (!it.isRecycled) it.recycle() }
            previousCdaFrame?.let { if (!it.isRecycled) it.recycle() }
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
            } else if (engine == "animeUltra") {
                "AniUltraAnime — AnimeSR_v2 recurrent (${backendLabel(engine)})"
            } else if (engine == "realism") {
                "AniRealism Test — CDA-VSR recurrent (${backendLabel(engine)})"
            } else {
                "${engineLabel(engine)} (${backendLabel(engine)})"
            },
        )
    }

    private fun enhancementDirectory(): File = File(filesDir, "enhancements").apply {
        check(exists() || mkdirs()) { "AniScale could not create its local history folder." }
    }

    private fun backendLabel(engine: String? = null): String = if (engine == "animeUltra" || engine == "realism") {
        "ONNX Runtime Mobile"
    } else if (engine == "superUltra") {
        if (nativeUsesVulkan()) "SPAN ncnn Vulkan FP16" else "SPAN ncnn CPU"
    } else if (nativeUsesVulkan()) {
        "Android ncnn Vulkan FP16"
    } else {
        "Android ncnn CPU"
    }

    private fun engineLabel(engine: String): String = when (engine) {
        "render" -> "AniScale Render"
        "turbo" -> "AniScale Turbo"
        "animeUltra" -> "AniUltraAnime"
        "realism" -> "AniRealism Test"
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

    private data class AnimeRuntimeState(
        var width: Int = 0,
        var height: Int = 0,
        var feedback: FloatArray = FloatArray(0),
        var hidden: FloatArray = FloatArray(0),
    ) {
        fun ensureSize(newWidth: Int, newHeight: Int) {
            if (width == newWidth && height == newHeight && feedback.isNotEmpty()) return
            width = newWidth
            height = newHeight
            feedback = FloatArray(3 * newWidth * 4 * newHeight * 4)
            hidden = FloatArray(64 * newWidth * newHeight)
        }

        fun reset() {
            feedback.fill(0f)
            hidden.fill(0f)
        }
    }

    private fun upscaleAnime(
        previous: Bitmap,
        current: Bitmap,
        next: Bitmap,
        state: AnimeRuntimeState,
    ): Bitmap {
        val width = current.width
        val height = current.height
        val plane = width * height
        val outputWidth = width * 4
        val outputHeight = height * 4
        state.ensureSize(width, height)
        val frames = FloatArray(9 * plane)
        appendRgbPlanes(previous, frames, 0)
        appendRgbPlanes(current, frames, 3 * plane)
        appendRgbPlanes(next, frames, 6 * plane)
        OnnxTensor.createTensor(
            animeEnvironment,
            FloatBuffer.wrap(frames),
            longArrayOf(1, 9, height.toLong(), width.toLong()),
        ).use { framesTensor ->
            OnnxTensor.createTensor(
                animeEnvironment,
                FloatBuffer.wrap(state.feedback),
                longArrayOf(1, 3, outputHeight.toLong(), outputWidth.toLong()),
            ).use { feedbackTensor ->
                OnnxTensor.createTensor(
                    animeEnvironment,
                    FloatBuffer.wrap(state.hidden),
                    longArrayOf(1, 64, height.toLong(), width.toLong()),
                ).use { stateTensor ->
                    animeSession.run(
                        mapOf(
                            "frames" to framesTensor,
                            "feedback" to feedbackTensor,
                            "state" to stateTensor,
                        ),
                    ).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val enhanced = result[0].value as Array<Array<Array<FloatArray>>>
                        @Suppress("UNCHECKED_CAST")
                        val nextState = result[1].value as Array<Array<Array<FloatArray>>>
                        val enhancedChannels = enhanced[0]
                        val stateChannels = nextState[0]
                        state.feedback = flattenChannels(
                            enhancedChannels,
                            3,
                            outputHeight,
                            outputWidth,
                        )
                        state.hidden = flattenChannels(stateChannels, 64, height, width)
                        val outputPixels = IntArray(outputWidth * outputHeight)
                        for (y in 0 until outputHeight) {
                            for (x in 0 until outputWidth) {
                                val red = (enhancedChannels[0][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                                val green = (enhancedChannels[1][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                                val blue = (enhancedChannels[2][y][x].coerceIn(0f, 1f) * 255).roundToInt()
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
    }

    private fun appendRgbPlanes(bitmap: Bitmap, destination: FloatArray, offset: Int) {
        val width = bitmap.width
        val height = bitmap.height
        val plane = width * height
        val pixels = IntArray(plane)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (index in pixels.indices) {
            val pixel = pixels[index]
            destination[offset + index] = ((pixel shr 16) and 0xff) / 255f
            destination[offset + plane + index] = ((pixel shr 8) and 0xff) / 255f
            destination[offset + 2 * plane + index] = (pixel and 0xff) / 255f
        }
    }

    private fun flattenChannels(
        source: Array<Array<FloatArray>>,
        channels: Int,
        height: Int,
        width: Int,
    ): FloatArray {
        val output = FloatArray(channels * height * width)
        val plane = height * width
        for (channel in 0 until channels) {
            for (y in 0 until height) {
                source[channel][y].copyInto(output, channel * plane + y * width)
            }
        }
        return output
    }

    private data class CdaRuntimeState(
        var width: Int = 0,
        var height: Int = 0,
        var low: FloatArray = FloatArray(0),
        var high: FloatArray = FloatArray(0),
        var framesSinceReset: Int = 0,
    ) {
        fun reset() {
            low = FloatArray(0)
            high = FloatArray(0)
            framesSinceReset = 0
        }
    }

    private fun upscaleCda(
        previous: Bitmap?,
        current: Bitmap,
        state: CdaRuntimeState,
    ): Bitmap {
        val width = current.width
        val height = current.height
        val plane = width * height
        if (state.width != width || state.height != height) {
            state.width = width
            state.height = height
            state.reset()
        }
        val frame = FloatArray(3 * plane)
        appendRgbPlanes(current, frame, 0)
        val inputShape = longArrayOf(1, 3, height.toLong(), width.toLong())
        val outputNames = setOf("output", "next_state_low", "next_state_high")
        val result = if (state.low.isEmpty() || state.high.isEmpty() || previous == null) {
            OnnxTensor.createTensor(
                animeEnvironment,
                FloatBuffer.wrap(frame),
                inputShape,
            ).use { frameTensor ->
                cdaInitializerSession.run(
                    mapOf("frame" to frameTensor),
                    outputNames,
                )
            }
        } else {
            val priors = decodedCdaPriors(previous, current)
            OnnxTensor.createTensor(
                animeEnvironment,
                FloatBuffer.wrap(frame),
                inputShape,
            ).use { frameTensor ->
                OnnxTensor.createTensor(
                    animeEnvironment,
                    FloatBuffer.wrap(priors.first),
                    longArrayOf(1, 2, height.toLong(), width.toLong()),
                ).use { motionTensor ->
                    OnnxTensor.createTensor(
                        animeEnvironment,
                        FloatBuffer.wrap(priors.second),
                        longArrayOf(1, 1, height.toLong(), width.toLong()),
                    ).use { residualTensor ->
                        OnnxTensor.createTensor(
                            animeEnvironment,
                            FloatBuffer.wrap(state.low),
                            longArrayOf(1, 64, height.toLong(), width.toLong()),
                        ).use { lowTensor ->
                            OnnxTensor.createTensor(
                                animeEnvironment,
                                FloatBuffer.wrap(state.high),
                                longArrayOf(1, 64, height.toLong(), width.toLong()),
                            ).use { highTensor ->
                                cdaRecurrentSession.run(
                                    mapOf(
                                        "frame" to frameTensor,
                                        "motion" to motionTensor,
                                        "residual" to residualTensor,
                                        "state_low" to lowTensor,
                                        "state_high" to highTensor,
                                    ),
                                    outputNames,
                                )
                            }
                        }
                    }
                }
            }
        }
        result.use {
            @Suppress("UNCHECKED_CAST")
            val output = it.get("output").orElseThrow().value as Array<Array<Array<FloatArray>>>
            @Suppress("UNCHECKED_CAST")
            val nextLow = it.get("next_state_low").orElseThrow().value as Array<Array<Array<FloatArray>>>
            @Suppress("UNCHECKED_CAST")
            val nextHigh = it.get("next_state_high").orElseThrow().value as Array<Array<Array<FloatArray>>>
            val outputChannels = output[0]
            state.low = flattenChannels(nextLow[0], 64, height, width)
            state.high = flattenChannels(nextHigh[0], 64, height, width)
            state.framesSinceReset++
            return bitmapFromChannels(outputChannels, width * 4, height * 4)
        }
    }

    private fun bitmapFromChannels(
        channels: Array<Array<FloatArray>>,
        width: Int,
        height: Int,
    ): Bitmap {
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val red = (channels[0][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                val green = (channels[1][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                val blue = (channels[2][y][x].coerceIn(0f, 1f) * 255).roundToInt()
                pixels[y * width + x] =
                    (0xff shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun decodedCdaPriors(
        previous: Bitmap,
        current: Bitmap,
    ): Pair<FloatArray, FloatArray> {
        val width = current.width
        val height = current.height
        val plane = width * height
        val previousPixels = IntArray(plane)
        val currentPixels = IntArray(plane)
        previous.getPixels(previousPixels, 0, width, 0, 0, width, height)
        current.getPixels(currentPixels, 0, width, 0, 0, width, height)
        val previousLuma = FloatArray(plane)
        val currentLuma = FloatArray(plane)
        for (index in 0 until plane) {
            previousLuma[index] = pixelLuma(previousPixels[index])
            currentLuma[index] = pixelLuma(currentPixels[index])
        }
        val motion = FloatArray(2 * plane)
        val residual = FloatArray(plane)
        val blockSize = 16
        val radius = 4
        val sampleStride = 4
        for (originY in 0 until height step blockSize) {
            for (originX in 0 until width step blockSize) {
                var bestX = 0
                var bestY = 0
                var bestScore = Float.POSITIVE_INFINITY
                for (deltaY in -radius..radius) {
                    for (deltaX in -radius..radius) {
                        var score = 0f
                        var count = 0
                        for (sampleY in 0 until blockSize step sampleStride) {
                            val y = originY + sampleY
                            if (y >= height) continue
                            for (sampleX in 0 until blockSize step sampleStride) {
                                val x = originX + sampleX
                                if (x >= width) continue
                                val referenceX = x + deltaX
                                val referenceY = y + deltaY
                                score += if (
                                    referenceX in 0 until width && referenceY in 0 until height
                                ) {
                                    kotlin.math.abs(
                                        currentLuma[y * width + x] -
                                            previousLuma[referenceY * width + referenceX],
                                    )
                                } else {
                                    1f
                                }
                                count++
                            }
                        }
                        score /= max(1, count)
                        val candidateMagnitude = kotlin.math.abs(deltaX) + kotlin.math.abs(deltaY)
                        val bestMagnitude = kotlin.math.abs(bestX) + kotlin.math.abs(bestY)
                        if (
                            score < bestScore - 1e-7f ||
                            (kotlin.math.abs(score - bestScore) <= 1e-7f &&
                                candidateMagnitude < bestMagnitude)
                        ) {
                            bestScore = score
                            bestX = deltaX
                            bestY = deltaY
                        }
                    }
                }
                for (y in originY until min(height, originY + blockSize)) {
                    for (x in originX until min(width, originX + blockSize)) {
                        val index = y * width + x
                        motion[index] = bestX.toFloat()
                        motion[plane + index] = bestY.toFloat()
                        val referenceX = (x + bestX).coerceIn(0, width - 1)
                        val referenceY = (y + bestY).coerceIn(0, height - 1)
                        residual[index] = kotlin.math.abs(
                            currentLuma[index] - previousLuma[referenceY * width + referenceX],
                        )
                    }
                }
            }
        }
        return motion to residual
    }

    private fun pixelLuma(pixel: Int): Float = (
        0.2126f * ((pixel shr 16) and 255) +
            0.7152f * ((pixel shr 8) and 255) +
            0.0722f * (pixel and 255)
        ) / 255f

    private fun isSceneCut(previous: Bitmap, current: Bitmap): Boolean {
        val stepX = max(1, current.width / 32)
        val stepY = max(1, current.height / 18)
        var difference = 0.0
        var samples = 0
        for (y in 0 until current.height step stepY) {
            for (x in 0 until current.width step stepX) {
                val first = previous.getPixel(x, y)
                val second = current.getPixel(x, y)
                val firstLuma = 0.2126 * ((first shr 16) and 255) +
                    0.7152 * ((first shr 8) and 255) + 0.0722 * (first and 255)
                val secondLuma = 0.2126 * ((second shr 16) and 255) +
                    0.7152 * ((second shr 8) and 255) + 0.0722 * (second and 255)
                difference += kotlin.math.abs(firstLuma - secondLuma) / 255.0
                samples++
            }
        }
        return samples > 0 && difference / samples > 0.28
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
