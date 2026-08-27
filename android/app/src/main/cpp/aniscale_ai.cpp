#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gpu.h"
#include "mat.h"
#include "net.h"

namespace {

std::unique_ptr<ncnn::Net> fusion;
std::unique_ptr<ncnn::Net> render;
std::unique_ptr<ncnn::Net> turbo2x;
std::unique_ptr<ncnn::Net> turbo4x;
std::mutex inference_mutex;
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
    net->opt.use_fp16_arithmetic = false;
    net->opt.num_threads = 4;
    if (gpu_ready) net->set_vulkan_device(0);
    if (net->load_param(manager, param) != 0 ||
        net->load_model(manager, weights) != 0) {
        return nullptr;
    }
    return net;
}

ncnn::Net* select_net(const std::string& engine, int requested_scale, int& native_scale) {
    if (engine == "turbo") {
        native_scale = requested_scale == 2 ? 2 : 4;
        return requested_scale == 2 ? turbo2x.get() : turbo4x.get();
    }
    native_scale = 4;
    return engine == "render" ? render.get() : fusion.get();
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

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeInitialize(
    JNIEnv* env,
    jobject,
    jobject asset_manager
) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    if (fusion && render && turbo2x && turbo4x) return JNI_TRUE;

    AAssetManager* manager = AAssetManager_fromJava(env, asset_manager);
    if (!manager) return JNI_FALSE;
    gpu_ready = ncnn::create_gpu_instance() == 0 && ncnn::get_gpu_count() > 0;

    fusion = load_net(
        manager,
        "models/realesrgan-x4plus-anime.param",
        "models/realesrgan-x4plus-anime.bin"
    );
    render = load_net(
        manager,
        "models/realesrgan-x4plus.param",
        "models/realesrgan-x4plus.bin"
    );
    turbo2x = load_net(
        manager,
        "models/realesr-animevideov3-x2.param",
        "models/realesr-animevideov3-x2.bin"
    );
    turbo4x = load_net(
        manager,
        "models/realesr-animevideov3-x4.param",
        "models/realesr-animevideov3-x4.bin"
    );
    return fusion && render && turbo2x && turbo4x ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jobject JNICALL
Java_app_aniscale_aniscale_MainActivity_nativeUpscale(
    JNIEnv* env,
    jobject,
    jobject bitmap,
    jstring engine_value,
    jint requested_scale,
    jint requested_tile
) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    if (!fusion || !render || !turbo2x || !turbo4x) return nullptr;

    const char* engine_chars = env->GetStringUTFChars(engine_value, nullptr);
    std::string engine(engine_chars ? engine_chars : "fusion");
    env->ReleaseStringUTFChars(engine_value, engine_chars);

    int native_scale = 4;
    ncnn::Net* net = select_net(engine, requested_scale, native_scale);
    if (!net || (requested_scale != 2 && requested_scale != 4)) return nullptr;

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
    const int tile_size = std::clamp(static_cast<int>(requested_tile), 64, 256);
    const int overlap = 10;
    const int output_width = width * requested_scale;
    const int output_height = height * requested_scale;
    jobject output_bitmap = make_bitmap(env, output_width, output_height);
    void* output_pixels = nullptr;
    if (!output_bitmap ||
        AndroidBitmap_lockPixels(env, output_bitmap, &output_pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        AndroidBitmap_unlockPixels(env, bitmap);
        return nullptr;
    }
    std::fill_n(
        static_cast<unsigned char*>(output_pixels),
        output_width * output_height * 4,
        static_cast<unsigned char>(0)
    );

    const auto* source = static_cast<const unsigned char*>(source_pixels);
    auto* destination = static_cast<unsigned char*>(output_pixels);
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    bool failed = false;

    for (int core_y0 = 0; core_y0 < height && !failed; core_y0 += tile_size) {
        const int core_y1 = std::min(core_y0 + tile_size, height);
        const int tile_y0 = std::max(0, core_y0 - overlap);
        const int tile_y1 = std::min(height, core_y1 + overlap);
        for (int core_x0 = 0; core_x0 < width; core_x0 += tile_size) {
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

            const int source_offset_x = (core_x0 - tile_x0) * requested_scale;
            const int source_offset_y = (core_y0 - tile_y0) * requested_scale;
            const int copy_width = (core_x1 - core_x0) * requested_scale;
            const int copy_height = (core_y1 - core_y0) * requested_scale;
            for (int y = 0; y < copy_height; ++y) {
                const float* red = result->channel(0).row(source_offset_y + y);
                const float* green = result->channel(1).row(source_offset_y + y);
                const float* blue = result->channel(2).row(source_offset_y + y);
                unsigned char* row = destination +
                    ((core_y0 * requested_scale + y) * output_width + core_x0 * requested_scale) * 4;
                for (int x = 0; x < copy_width; ++x) {
                    const int sx = source_offset_x + x;
                    row[x * 4] = byte_from_float(red[sx]);
                    row[x * 4 + 1] = byte_from_float(green[sx]);
                    row[x * 4 + 2] = byte_from_float(blue[sx]);
                    row[x * 4 + 3] = 255;
                }
            }
        }
    }

    AndroidBitmap_unlockPixels(env, output_bitmap);
    AndroidBitmap_unlockPixels(env, bitmap);
    return failed ? nullptr : output_bitmap;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    std::lock_guard<std::mutex> guard(inference_mutex);
    turbo4x.reset();
    turbo2x.reset();
    render.reset();
    fusion.reset();
    if (gpu_ready) ncnn::destroy_gpu_instance();
    gpu_ready = false;
}
