package app.aniscale.aniscale

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.os.Build
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
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
        @JvmStatic external fun nativeUpscale(
            bitmap: Bitmap,
            engine: String,
            scale: Int,
            tileSize: Int,
        ): Bitmap?
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val cancelled = AtomicBoolean(false)
    private var progressSink: EventChannel.EventSink? = null
    @Volatile private var initialized = false

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
        if (path.isNullOrBlank() || scale !in listOf(2, 4)) {
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
                val source = fitInput(decoded, scale, 18_000_000, null)
                if (source !== decoded) decoded.recycle()
                val input = source.copy(Bitmap.Config.ARGB_8888, false)
                if (source !== input) source.recycle()
                emitProgress(0.12)
                val enhanced = nativeUpscale(input, "fusion", scale, tileSize)
                    ?: error("AniScale Fusion could not process this image.")
                input.recycle()
                check(!cancelled.get()) { "Image upscaling was cancelled." }
                val usePng = outputFormat == "png" || outputFormat == "automatic"
                val extension = if (usePng) "png" else "jpg"
                val output = File(cacheDir, "aniscale_${System.currentTimeMillis()}.$extension")
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
                    "engine" to "AniScale Fusion (Android ncnn Vulkan)",
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
        val efficient = call.argument<Boolean>("efficient") ?: true
        val tileSize = call.argument<Int>("tileSize") ?: 192
        val engine = call.argument<String>("engine") ?: "fusion"
        if (path.isNullOrBlank() || scale !in listOf(2, 4) ||
            engine !in listOf("fusion", "render", "turbo")) {
            result.error("bad_arguments", "Invalid video request.", null)
            return
        }
        cancelled.set(false)
        worker.execute {
            try {
                ensureModels()
                val response = processVideo(path, scale, efficient, tileSize, engine)
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
        efficient: Boolean,
        tileSize: Int,
        engine: String,
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
        val maxInputEdge = when {
            engine == "turbo" -> 1080
            efficient -> 720
            else -> 1080
        }
        val fittedFirst = fitInput(first, scale, 8_847_360, maxInputEdge)
        if (fittedFirst !== first) first.recycle()
        val inputWidth = fittedFirst.width
        val inputHeight = fittedFirst.height
        val outputWidth = inputWidth * scale
        val outputHeight = inputHeight * scale
        val output = File(cacheDir, "aniscale_video_${System.currentTimeMillis()}.mp4")
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

        val format = MediaFormat.createVideoFormat(
            MediaFormat.MIMETYPE_VIDEO_AVC,
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
                min(40_000_000, max(4_000_000, outputWidth * outputHeight * 4)),
            )
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
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
                val rgba = fitted.copy(Bitmap.Config.ARGB_8888, false)
                fitted.recycle()
                val enhanced = nativeUpscale(rgba, engine, scale, tileSize)
                    ?: error("The selected AniScale model failed on frame ${frameIndex + 1}.")
                rgba.recycle()
                queueFrame(encoder, enhanced, frameIndex, fps, muxer, muxState, audioFormat)
                enhanced.recycle()
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
        val label = when (engine) {
            "render" -> "AniScale Render"
            "turbo" -> "AniScale Turbo"
            else -> "AniScale Fusion"
        }
        return hashMapOf(
            "path" to output.absolutePath,
            "outputWidth" to outputWidth,
            "outputHeight" to outputHeight,
            "durationSeconds" to durationMs / 1000.0,
            "engine" to "$label (Android ncnn Vulkan)",
        )
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
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        for (y in 0 until height) {
            for (x in 0 until width) {
                val color = pixels[y * width + x]
                val red = color shr 16 and 0xff
                val green = color shr 8 and 0xff
                val blue = color and 0xff
                val luma = ((66 * red + 129 * green + 25 * blue + 128) shr 8) + 16
                yPlane.buffer.put(
                    y * yPlane.rowStride + x * yPlane.pixelStride,
                    luma.coerceIn(0, 255).toByte(),
                )
                if (x % 2 == 0 && y % 2 == 0) {
                    val u = ((-38 * red - 74 * green + 112 * blue + 128) shr 8) + 128
                    val v = ((112 * red - 94 * green - 18 * blue + 128) shr 8) + 128
                    val uvX = x / 2
                    val uvY = y / 2
                    uPlane.buffer.put(
                        uvY * uPlane.rowStride + uvX * uPlane.pixelStride,
                        u.coerceIn(0, 255).toByte(),
                    )
                    vPlane.buffer.put(
                        uvY * vPlane.rowStride + uvX * vPlane.pixelStride,
                        v.coerceIn(0, 255).toByte(),
                    )
                }
            }
        }
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
