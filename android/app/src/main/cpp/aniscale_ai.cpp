#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpu.h"
#include "mat.h"
#include "net.h"

namespace {

std::unique_ptr<ncnn::Net> active_net;
std::string active_model;
AAssetManager* app_assets = nullptr;
std::mutex inference_mutex;
std::atomic<bool> cancelled{false};
bool gpu_ready = false;

std::unique_ptr<ncnn::Net> load_net(
    AAssetManager* manager,
    const char* param,
    const char* weights
) {
    auto net = std::make_unique<ncnn::Net>();
    net->opt.use_vulkan_compute = gpu_ready;
    net->opt.use_fp16_packed = true;
    net->opt.use_fp16_storage = true;
    net->opt.use_fp16_arithmetic = gpu_ready;
    net->opt.use_local_pool_allocator = true;
    // Keep the phone responsive and avoid saturating every CPU core when a
    // Vulkan driver is unavailable.
    net->opt.num_threads = 2;
    if (gpu_ready) net->set_vulkan_device(0);
    if (net->load_param(manager, param) != 0 ||
        net->load_model(manager, weights) != 0) {
        return nullptr;
    }
    return net;
}

bool select_net(const std::string& engine, int requested_scale, int& native_scale) {
    std::string model;
    const char* param = nullptr;
    const char* weights = nullptr;
    if (engine == "turbo") {
        native_scale = requested_scale == 2 ? 2 : 4;
        model = requested_scale == 2 ? "turbo2x" : "turbo4x";
        param = requested_scale == 2
            ? "models/realesr-animevideov3-x2.param"
            : "models/realesr-animevideov3-x4.param";
        weights = requested_scale == 2
            ? "models/realesr-animevideov3-x2.bin"
            : "models/realesr-animevideov3-x4.bin";
    } else if (engine == "render") {
        native_scale = 4;
        model = "render";
        param = "models/realesrgan-x4plus.param";
        weights = "models/realesrgan-x4plus.bin";
    } else {
        native_scale = 4;
        model = "fusion";
        param = "models/realesrgan-x4plus-anime.param";
        weights = "models/realesrgan-x4plus-anime.bin";
    }
    if (active_net && active_model == model) return true;

    // Only one network is kept resident. Loading all four models together was
    // enough to exhaust RAM/GPU memory on mid-range Android devices.
    active_net.reset();
    active_model.clear();
    active_net = load_net(app_assets, param, weights);
    if (!active_net) return false;
    active_model = model;
    return true;
}

unsigned char byte_from_float(float value) {
    return static_cast<unsigned char>(
        std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f)
    );
}

jobject make_bitmap(JNIEnv* env, int width, int height) {
    jclass bitmap_class = env->FindClass("android/graphics/Bitmap");
    jclass config_class = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb_field = env->GetStaticFieldID(
        config_class,
        "ARGB_8888",
        "Landroid/graphics/Bitmap$Config;"
    );
    jobject argb = env->GetStaticObjectField(config_class, argb_field);
    jmethodID create = env->GetStaticMethodID(
        bitmap_class,
        "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;"
    );
    return env->CallStaticObjectMethod(bitmap_class, create, width, height, argb);
}

void throw_runtime(JNIEnv* env, const char* message) {
    jclass type = env->FindClass("java/lang/RuntimeException");
    if (type) env->ThrowNew(type, message);
}

float sharpened(float center, float left, float right, float above, float below, float strength) {
    const float neighbors = (left + right + above + below) * 0.25f;
    return center + (center - neighbors) * strength;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeInitialize(
    JNIEnv* env,
    jobject,
    jobject asset_manager
) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    if (app_assets) return JNI_TRUE;

    AAssetManager* manager = AAssetManager_fromJava(env, asset_manager);
    if (!manager) return JNI_FALSE;
    gpu_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;
    app_assets = manager;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeUsesVulkan(JNIEnv*, jobject) {
    return gpu_ready ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeCancel(JNIEnv*, jobject) {
    cancelled.store(true);
}

extern "C" JNIEXPORT jobject JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeUpscale(
    JNIEnv* env,
    jobject,
    jobject bitmap,
    jstring engine_value,
    jint requested_scale,
    jint requested_tile,
    jfloat sharpening
) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    cancelled.store(false);
    if (!app_assets) {
        throw_runtime(env, "The Android AI runtime is not initialized.");
        return nullptr;
    }

    const char* engine_chars = env->GetStringUTFChars(engine_value, nullptr);
    std::string engine(engine_chars ? engine_chars : "fusion");
    if (engine_chars) env->ReleaseStringUTFChars(engine_value, engine_chars);

    int native_scale = 4;
    if ((requested_scale != 2 && requested_scale != 4) ||
        !select_net(engine, requested_scale, native_scale)) {
        throw_runtime(env, "The selected AniScale model could not be loaded. Close other apps and try again.");
        return nullptr;
    }
    ncnn::Net* net = active_net.get();

    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return nullptr;
    }
    void* source_pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &source_pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return nullptr;
    }

    const int width = static_cast<int>(info.width);
    const int height = static_cast<int>(info.height);
    int tile_size = static_cast<int>(requested_tile);
    if (tile_size == 0) tile_size = engine == "turbo" && gpu_ready ? 320 : 256;
    tile_size = std::clamp(tile_size, 64, engine == "turbo" ? 384 : 256);
    const int overlap = 10;
    const int output_width = width * requested_scale;
    const int output_height = height * requested_scale;
    jobject output_bitmap = make_bitmap(env, output_width, output_height);
    AndroidBitmapInfo output_info{};
    if (!output_bitmap ||
        AndroidBitmap_getInfo(env, output_bitmap, &output_info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, bitmap);
        throw_runtime(env, "Android could not allocate the enhanced image. Try 2x or Faster mode.");
        return nullptr;
    }
    void* output_pixels = nullptr;
    if (!output_bitmap ||
        AndroidBitmap_lockPixels(env, output_bitmap, &output_pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, bitmap);
        return nullptr;
    }
    std::fill_n(
        static_cast<unsigned char*>(output_pixels),
        output_info.stride * output_height,
        static_cast<unsigned char>(0)
    );

    const auto* source = static_cast<const unsigned char*>(source_pixels);
    auto* destination = static_cast<unsigned char*>(output_pixels);
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    bool failed = false;

    for (int core_y0 = 0; core_y0 < height && !failed && !cancelled.load(); core_y0 += tile_size) {
        const int core_y1 = std::min(core_y0 + tile_size, height);
        const int tile_y0 = std::max(0, core_y0 - overlap);
        const int tile_y1 = std::min(height, core_y1 + overlap);
        for (int core_x0 = 0; core_x0 < width; core_x0 += tile_size) {
            if (cancelled.load()) break;
            const int core_x1 = std::min(core_x0 + tile_size, width);
            const int tile_x0 = std::max(0, core_x0 - overlap);
            const int tile_x1 = std::min(width, core_x1 + overlap);
            const int tile_width = tile_x1 - tile_x0;
            const int tile_height = tile_y1 - tile_y0;
            std::vector<unsigned char> tile(tile_width * tile_height * 4);
            for (int y = 0; y < tile_height; ++y) {
                const unsigned char* row = source + (tile_y0 + y) * info.stride + tile_x0 * 4;
                std::copy_n(row, tile_width * 4, tile.data() + y * tile_width * 4);
            }

            ncnn::Mat input = ncnn::Mat::from_pixels(
                tile.data(),
                ncnn::Mat::PIXEL_RGBA2RGB,
                tile_width,
                tile_height
            );
            input.substract_mean_normalize(nullptr, norm);
            ncnn::Extractor extractor = net->create_extractor();
            extractor.set_light_mode(true);
            if (extractor.input("data", input) != 0) {
                failed = true;
                break;
            }
            ncnn::Mat prediction;
            if (extractor.extract("output", prediction) != 0 || prediction.empty()) {
                failed = true;
                break;
            }
            ncnn::Mat unpacked;
            ncnn::convert_packing(prediction, unpacked, 1);
            ncnn::Mat scaled;
            const ncnn::Mat* result = &unpacked;
            if (native_scale != requested_scale) {
                ncnn::resize_bicubic(
                    unpacked,
                    scaled,
                    tile_width * requested_scale,
                    tile_height * requested_scale
                );
                result = &scaled;
            }
            if (result->w < tile_width * requested_scale ||
                result->h < tile_height * requested_scale || result->c < 3) {
                failed = true;
                break;
            }

            const int source_offset_x = (core_x0 - tile_x0) * requested_scale;
            const int source_offset_y = (core_y0 - tile_y0) * requested_scale;
            const int copy_width = (core_x1 - core_x0) * requested_scale;
            const int copy_height = (core_y1 - core_y0) * requested_scale;
            const ncnn::Mat red_channel = result->channel(0);
            const ncnn::Mat green_channel = result->channel(1);
            const ncnn::Mat blue_channel = result->channel(2);
            for (int y = 0; y < copy_height; ++y) {
                const int sy = source_offset_y + y;
                const int above_y = std::max(0, sy - 1);
                const int below_y = std::min(result->h - 1, sy + 1);
                const float* red = red_channel.row(sy);
                const float* green = green_channel.row(sy);
                const float* blue = blue_channel.row(sy);
                const float* red_above = red_channel.row(above_y);
                const float* green_above = green_channel.row(above_y);
                const float* blue_above = blue_channel.row(above_y);
                const float* red_below = red_channel.row(below_y);
                const float* green_below = green_channel.row(below_y);
                const float* blue_below = blue_channel.row(below_y);
                unsigned char* row = destination +
                    (core_y0 * requested_scale + y) * output_info.stride + core_x0 * requested_scale * 4;
                for (int x = 0; x < copy_width; ++x) {
                    const int sx = source_offset_x + x;
                    const int left_x = std::max(0, sx - 1);
                    const int right_x = std::min(result->w - 1, sx + 1);
                    row[x * 4] = byte_from_float(sharpened(
                        red[sx], red[left_x], red[right_x], red_above[sx], red_below[sx], sharpening
                    ));
                    row[x * 4 + 1] = byte_from_float(sharpened(
                        green[sx], green[left_x], green[right_x], green_above[sx], green_below[sx], sharpening
                    ));
                    row[x * 4 + 2] = byte_from_float(sharpened(
                        blue[sx], blue[left_x], blue[right_x], blue_above[sx], blue_below[sx], sharpening
                    ));
                    row[x * 4 + 3] = 255;
                }
            }
        }
    }

    AndroidBitmap_unlockPixels(env, output_bitmap);
    AndroidBitmap_unlockPixels(env, bitmap);
    if (cancelled.load()) {
        throw_runtime(env, "Upscaling was cancelled.");
        return nullptr;
    }
    if (failed) {
        throw_runtime(env, "The selected model ran out of working memory. Try Automatic tile size or Faster mode.");
        return nullptr;
    }
    return output_bitmap;
}

extern "C" JNIEXPORT void JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeFillYuv420(
    JNIEnv* env,
    jobject,
    jobject bitmap,
    jobject y_buffer,
    jint y_row_stride,
    jint y_pixel_stride,
    jobject u_buffer,
    jint u_row_stride,
    jint u_pixel_stride,
    jobject v_buffer,
    jint v_row_stride,
    jint v_pixel_stride
) {
    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throw_runtime(env, "The encoder could not read an enhanced frame.");
        return;
    }
    auto* y_out = static_cast<unsigned char*>(env->GetDirectBufferAddress(y_buffer));
    auto* u_out = static_cast<unsigned char*>(env->GetDirectBufferAddress(u_buffer));
    auto* v_out = static_cast<unsigned char*>(env->GetDirectBufferAddress(v_buffer));
    if (!y_out || !u_out || !v_out) {
        AndroidBitmap_unlockPixels(env, bitmap);
        throw_runtime(env, "The encoder did not provide direct YUV buffers.");
        return;
    }
    const auto* source = static_cast<const unsigned char*>(pixels);
    for (int y = 0; y < static_cast<int>(info.height); ++y) {
        const unsigned char* row = source + y * info.stride;
        for (int x = 0; x < static_cast<int>(info.width); ++x) {
            const int red = row[x * 4];
            const int green = row[x * 4 + 1];
            const int blue = row[x * 4 + 2];
            const int luma = ((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16;
            y_out[y * y_row_stride + x * y_pixel_stride] = static_cast<unsigned char>(std::clamp(luma, 0, 255));
            if ((x & 1) == 0 && (y & 1) == 0) {
                const int chroma_x = x / 2;
                const int chroma_y = y / 2;
                const int u = ((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128;
                const int v = ((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128;
                u_out[chroma_y * u_row_stride + chroma_x * u_pixel_stride] = static_cast<unsigned char>(std::clamp(u, 0, 255));
                v_out[chroma_y * v_row_stride + chroma_x * v_pixel_stride] = static_cast<unsigned char>(std::clamp(v, 0, 255));
            }
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    active_net.reset();
    active_model.clear();
    app_assets = nullptr;
    if (gpu_ready) ncnn::destroy_gpu_instance();
    gpu_ready = false;
}
