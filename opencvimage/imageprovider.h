// imageprovider.h
#ifndef IMAGEPROVIDER_H
#define IMAGEPROVIDER_H

#include <QQuickImageProvider>
#include <QImage>
#include <QDebug>
#include "opencv/cvconvert.h"

class ImageProvider : public QQuickImageProvider
{
public:
    ImageProvider() : QQuickImageProvider(QQuickImageProvider::Image)
    {
        // 初始化为 1x1 透明图，防止 QML 拿到 null QImage 崩溃
        m_currentImage = QImage(1, 1, QImage::Format_ARGB32);
        m_currentImage.fill(Qt::transparent);
    }

    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override {
        Q_UNUSED(requestedSize);

        // 去掉 query string（Timer 会拼接 "?random"），只取 id 部分
        QString cleanId = id.split('?').first();

        if (cleanId == "frame" || cleanId == "loaded") {
            if (size) *size = m_currentImage.size();
            return m_currentImage;
        }
        return QImage();
    }

    // QImage 直接更新（来自 VideoProcessor 信号）
    void updateImage(const QImage &img) {
        m_currentImage = img;
    }

    // cv::Mat 间接更新（静态图片等场景）
    void updateImage(const cv::Mat &mat) {
        m_currentImage = cvMatToQImage(mat);
        qDebug() << "[ImageProvider] Updated from cv::Mat, size:" << m_currentImage.size();
    }

private:
    QImage m_currentImage;
};

#endif // IMAGEPROVIDER_H
