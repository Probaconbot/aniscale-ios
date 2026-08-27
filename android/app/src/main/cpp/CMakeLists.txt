cmake_minimum_required(VERSION 3.22.1)
project(aniscale_ai)

set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_library(aniscale_ai SHARED aniscale_ai.cpp)

target_link_libraries(
    aniscale_ai
    ncnn
    android
    jnigraphics
    log
)
