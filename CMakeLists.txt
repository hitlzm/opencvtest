cmake_minimum_required(VERSION 3.10)

project(opencvtest LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "E:/opencv-4.2.0/DNN/install")
# set(OPENCV_BIN_DIR "${OPENCV_DIR}/bin")
# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(Qt5 COMPONENTS Core Quick REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT OPENCV_BIN_DIR)
    # 尝试从 OpenCV_DIR 推导
    get_filename_component(OPENCV_BIN_DIR "${OpenCV_DIR}/x64/mingw/bin" ABSOLUTE)
endif()


# ══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime 配置（根据实际安装路径修改）
# ══════════════════════════════════════════════════════════════════════════════
#
# ONNX Runtime 头文件目录（包含 onnxruntime_cxx_api.h 等）
# 源码目录: E:/onnxruntime/include/
set(ONNXRUNTIME_INCLUDE_DIR "E:/onnxruntime/include")

# ONNX Runtime 库文件路径
# 根据编译配置选择对应的 .lib / .dll.a / .so 文件
# MinGW 示例: E:/onnxruntime/build/Windows/Release/onnxruntime.dll
#            或 E:/onnxruntime/build/Windows/Release/libonnxruntime.dll.a
# MSVC 示例:  E:/onnxruntime/build/Windows/Release/onnxruntime.lib
set(ONNXRUNTIME_LIB_DIR "E:/onnxruntime/build/Windows/Release")
set(ONNXRUNTIME_LIB_NAME "onnxruntime")   # 不含前缀/后缀的库名

# ONNX Runtime DLL 目录（运行时所需，用于 POST_BUILD 拷贝）
# 若 DLL 与 .lib 不在同一目录，可单独指定
set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_LIB_DIR}")


if(ANDROID)
    add_library(opencvtest SHARED
      main.cpp
      qml.qrc
    )
else()
    add_executable(opencvtest
      main.cpp
      opencv/objectdetector.cpp
      opencv/videoprocessor.cpp
      opencv/videoprocessor.h
      opencv/cvconvert.h
      opencvimage/imageprovider.h
      opencvimage/framesource.h
      DNN/dnnvideoprocessor.cpp
      DNN/yolodetector.cpp
      DNN/onnxdetector.cpp
      DNN/onnxvideoprocessor.cpp
      ONNXRUNTIME/onnxyolodetector.cpp
      ONNXRUNTIME/streamprocessor.cpp
      ONNXRUNTIME/streamprocessor.h
      qml.qrc
    )
endif()

target_compile_definitions(opencvtest
  PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
target_link_libraries(opencvtest
  PRIVATE Qt5::Core Qt5::Quick
   ${OpenCV_LIBS}          # 链接所有必需的OpenCV库
   )

# ── 头文件路径 ───────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}          # OpenCV 的头文件路径
    ${ONNXRUNTIME_INCLUDE_DIR}      # ONNX Runtime 的头文件路径
    ${CMAKE_CURRENT_SOURCE_DIR}     # 项目根目录（方便跨目录 #include）
)

# ── ONNX Runtime 库链接 ──────────────────────────────────────
# 方式1: 直接链接库文件（推荐）
# target_link_libraries(opencvtest PRIVATE
#     "${ONNXRUNTIME_LIB_DIR}/lib${ONNXRUNTIME_LIB_NAME}.dll.a"  # MinGW
#     # "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_LIB_NAME}.lib"     # MSVC
# )

# 方式2: 通过 -L 和 -l 链接
target_link_directories(opencvtest PRIVATE "${ONNXRUNTIME_LIB_DIR}")
target_link_libraries(opencvtest PRIVATE "${ONNXRUNTIME_LIB_NAME}")

# ── 构建后拷贝 DLL 到可执行文件目录 ──────────────────────────
add_custom_command(TARGET opencvtest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${OPENCV_BIN_DIR}"
        "$<TARGET_FILE_DIR:opencvtest>"
    COMMENT "Copying OpenCV DLLs from ${OPENCV_BIN_DIR}"
)

# ONNX Runtime DLL 拷贝（若 DLL 与 .lib 在同一目录则已被上述命令覆盖）
# 如果 ONNX Runtime DLL 在单独目录，取消下面注释：
# add_custom_command(TARGET opencvtest POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll"
#         "$<TARGET_FILE_DIR:opencvtest>"
#     COMMENT "Copying ONNX Runtime DLL"
# )
