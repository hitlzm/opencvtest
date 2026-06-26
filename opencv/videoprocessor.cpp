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

bool VideoProcessor::init(const QString &videoPath, const QString &templatePath)
{
    // 1. 加载模板
    cv::Mat templ = cv::imread(templatePath.toLocal8Bit().constData());
    if (!m_detector.setTemplate(templ)) {
        emit error("Failed to set template: " + templatePath);
        return false;
    }

    m_detector.setMatchThreshold(0.7f);
    m_detector.setRansacThreshold(4.0f);

    // 2. 打开视频
    m_capture.open(videoPath.toLocal8Bit().constData());
    if (!m_capture.isOpened()) {
        emit error("Cannot open video: " + videoPath);
        return false;
    }

    qDebug() << "[VideoProcessor] Initialized OK, target FPS:" << m_targetFps;
    return true;
}

void VideoProcessor::start()
{
    m_running = true;

    // 在工作线程内创建单次触发定时器
    if (!m_timer) {
        m_timer = new QTimer(this);
        m_timer->setSingleShot(true);
        QObject::connect(m_timer, &QTimer::timeout,
                         this, &VideoProcessor::processFrame);
    }

    qDebug() << "[VideoProcessor] Started, thread:" << QThread::currentThread();
    processFrame(); // 立即处理第一帧
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

    // ── 检测目标并绘制检测框 ──
    cv::Rect bbox;
    if (m_detector.detect(frame, bbox)) {
        cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);   //画绿色方框
    }

    // ── 转换为 QImage 发送到主线程 ──
    QImage img = cvMatToQImage(frame);
    if (!img.isNull()) {
        emit frameReady(img);
    }

    // ── 按目标帧率调度下一帧 ──
    // 计算每帧间隔（ms），减去本帧处理耗时。若已超时则立即触发下一帧。
    int interval = 1000 / m_targetFps;
    if (interval < 5)
        interval = 5; // 最低 5ms，防止事件循环饿死

    m_timer->start(interval);
}
