#include "videoprocessor.h"
#include "cvconvert.h"
#include <QDebug>
#include <QThread>
#include <opencv2/imgproc.hpp>

VideoProcessor::VideoProcessor(QObject *parent)
    : QObject(parent), m_running(false)
{
}

VideoProcessor::~VideoProcessor()
{
    stop();
}

void VideoProcessor::setTargetFps(int fps)
{
    if (fps > 0 && fps <= 60)
        m_targetFps = fps;
}

// ── 视频 ──────────────────────────────────────────────────

bool VideoProcessor::openVideo(const QString &videoPath)
{
    m_capture.open(videoPath.toLocal8Bit().constData());
    if (!m_capture.isOpened()) {
        emit error("Cannot open video: " + videoPath);
        return false;
    }
    qDebug() << "[VideoProcessor] Video opened:" << videoPath;
    return true;
}

// ── 模板 ──────────────────────────────────────────────────

bool VideoProcessor::addTemplateFile(const QString &path, const std::string &name)
{
    cv::Mat templ = cv::imread(path.toLocal8Bit().constData());
    if (templ.empty()) {
        emit error("Failed to load template: " + path);
        return false;
    }
    if (!m_detector.addTemplate(templ, name)) {
        emit error("Failed to extract features from: " + path);
        return false;
    }
    return true;
}

// ── 初始化（便捷） ─────────────────────────────────────────

// bool VideoProcessor::init(const QString &videoPath, const QString &templatePath)
// {
//     // 加载模板
//     if (!addTemplateFile(templatePath))
//         return false;

//     m_detector.setMatchThreshold(0.7f);
//     m_detector.setRansacThreshold(4.0f);

//     // 打开视频
//     if (!openVideo(videoPath))
//         return false;

//     qDebug() << "[VideoProcessor] Initialized OK, templates:"
//              << m_detector.templateCount()
//              << "target FPS:" << m_targetFps;
//     return true;
// }

// ── 运行时 ────────────────────────────────────────────────

void VideoProcessor::start()
{
    m_running = true;

    if (!m_timer) {
        m_timer = new QTimer(this);
        m_timer->setSingleShot(true);
        QObject::connect(m_timer, &QTimer::timeout,
                         this, &VideoProcessor::processFrame);
    }

    qDebug() << "[VideoProcessor] Started, thread:" << QThread::currentThread();
    processFrame();
}

void VideoProcessor::stop()
{
    m_running = false;
    if (m_timer)
        m_timer->stop();
}

void VideoProcessor::processFrame()
{
    if (!m_running) {
        qDebug() << "[VideoProcessor] Stopped";
        emit finished();
        return;
    }

    // ── 读取一帧 ──
    cv::Mat frame;
    if (!m_capture.read(frame)) {
        qDebug() << "[VideoProcessor] Video ended";
        emit finished();
        return;
    }

    // ── 检测目标（遍历所有模板，选最优）并绘制检测框 ──
    cv::Rect bbox;
    if (m_detector.detect(frame, bbox)) {
        cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
    }

    // ── 转换为 QImage 发送到主线程 ──
    QImage img = cvMatToQImage(frame);
    if (!img.isNull()) {
        emit frameReady(img);
    }

    // ── 按目标帧率调度下一帧 ──
    int interval = 1000 / m_targetFps;
    if (interval < 5)
        interval = 5;

    m_timer->start(interval);
}
