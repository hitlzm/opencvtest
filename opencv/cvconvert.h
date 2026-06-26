// cvconvert.h — cv::Mat ↔ QImage 转换工具
#ifndef CVCONVERT_H
#define CVCONVERT_H

#include <QImage>
#include <opencv2/opencv.hpp>

inline QImage cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type()) {
        case CV_8UC3: { // 8-bit, 3 channels (BGR)
            QImage image(mat.data, mat.cols, mat.rows, mat.step,
                         QImage::Format_RGB888);
            return image.rgbSwapped().copy(); // BGR→RGB + 深拷贝
        }
        case CV_8UC1: { // 8-bit, 1 channel (Grayscale)
            QImage image(mat.data, mat.cols, mat.rows, mat.step,
                         QImage::Format_Grayscale8);
            return image.copy(); // 深拷贝
        }
        default:
            return QImage();
    }
}

#endif // CVCONVERT_H
