// cvconvert.h — cv::Mat ↔ QImage 转换工具
#ifndef CVCONVERT_H
#define CVCONVERT_H

#include <QImage>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief cv::Mat → QImage 转换（优化版）
 *
 * 先分配 QImage（自有缓冲区），再用 cv::cvtColor 直接将 BGR→RGB
 * 写入 QImage 的像素内存。相比旧版 image.rgbSwapped().copy()：
 *  - 只用一次分配（QImage 自身），无需 .copy() 二次分配
 *  - BGR→RGB 转换由 OpenCV SIMD（SSE/AVX/NEON）加速
 *  - 灰度图同理：分配 QImage 后用 cv::Mat wrapper + copyTo 一次性写入
 */
inline QImage cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
        case CV_8UC3: {
            // 直接分配目标 QImage（Format_RGB888，自有内存）
            QImage image(mat.cols, mat.rows, QImage::Format_RGB888);
            // 构造 cv::Mat 头，指向 QImage 的缓冲区（零拷贝包装）
            cv::Mat wrapper(mat.rows, mat.cols, CV_8UC3,
                            image.bits(), image.bytesPerLine());
            // SIMD 加速：BGR → RGB，直接写入 QImage 缓冲
            cv::cvtColor(mat, wrapper, cv::COLOR_BGR2RGB);
            return image;
        }
        case CV_8UC1: {
            QImage image(mat.cols, mat.rows, QImage::Format_Grayscale8);
            cv::Mat wrapper(mat.rows, mat.cols, CV_8UC1,
                            image.bits(), image.bytesPerLine());
            mat.copyTo(wrapper);
            return image;
        }
        default:
            return QImage();
    }
}

#endif // CVCONVERT_H
