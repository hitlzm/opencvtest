cmake_minimum_required(VERSION 3.10)

project(opencvtest LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "E:/opencv-4.2.0/mingw2/install")
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
      qml.qrc
    )
endif()

target_compile_definitions(opencvtest
  PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
target_link_libraries(opencvtest
  PRIVATE Qt5::Core Qt5::Quick 
   ${OpenCV_LIBS}          # 链接所有必需的OpenCV库
   )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}  # OpenCV的头文件路径
)

# add_custom_command(TARGET opencvtest POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#         "${OPENCV_BIN_DIR}"
#         "$<TARGET_FILE_DIR:opencvtest>"
#     COMMENT "Copying OpenCV DLLs from ${OPENCV_BIN_DIR}"
# )
