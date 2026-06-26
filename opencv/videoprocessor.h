// videoprocessor.h — 视频读取 + 目标检测工作线程
#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <atomic>
#include <opencv2/videoio.hpp>
#include "ObjectDetector.h"

class VideoProcessor : public QObject
{
    Q_OBJECT
public:
    explicit VideoProcessor(QObject *parent = nullptr);
    ~VideoProcessor();

    /// 加载视频文件（只调用一次）
    bool openVideo(const QString &videoPath);

    /// 添加一个模板图片（可多次调用，加载多个模板）
    bool addTemplateFile(const QString &path, const std::string &name = "");

    /// 便捷：打开视频 + 加载一个模板
    bool init(const QString &videoPath, const QString &templatePath);

    /// 设置目标帧率（默认 30）
    void setTargetFps(int fps);

    /// ObjectDetector 参数转发
    void setMatchThreshold(float t)   { m_detector.setMatchThreshold(t); }
    void setRansacThreshold(float t)  { m_detector.setRansacThreshold(t); }
    int  templateCount() const        { return m_detector.templateCount(); }

public slots:
    void start();         // 开始逐帧处理
    void processFrame();  // 处理一帧（由 QTimer 驱动）
    void stop();          // 停止处理

signals:
    void frameReady(const QImage &frame);  // 处理完的帧（含检测框）
    void error(const QString &msg);
    void finished();

private:
    ObjectDetector m_detector;
    cv::VideoCapture m_capture;
    QTimer *m_timer = nullptr;       // 工作线程中的定时器
    std::atomic<bool> m_running;
    int m_targetFps = 30;            // 目标帧率上限
};

#endif // VIDEOPROCESSOR_H
