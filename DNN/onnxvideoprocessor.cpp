#include "onnxvideoprocessor.h"
#include "../opencv/cvconvert.h"
#include <QDebug>
#include <QThread>
#include <opencv2/imgproc.hpp>

// ========== 构造 / 析构 ==========
ONNXVideoProcessor::ONNXVideoProcessor(QObject *parent)
    : QObject(parent), m_running(false)
{
}

ONNXVideoProcessor::~ONNXVideoProcessor()
{
    stop();
}

// ========== 参数 ==========
void ONNXVideoProcessor::setTargetFps(int fps)
{
    if (fps > 0 && fps <= 60)
        m_targetFps = fps;
}

void ONNXVideoProcessor::setConfThreshold(float t) { m_detector.setConfThreshold(t); }
void ONNXVideoProcessor::setNmsThreshold(float t)  { m_detector.setNmsThreshold(t); }

void ONNXVideoProcessor::setInputSize(int width, int height)
{
    m_detector.setInputSize(width, height);
}

void ONNXVideoProcessor::setBackend(int backend, int target)
{
    m_detector.setBackend(backend, target);
}

// ========== 初始化 ==========
bool ONNXVideoProcessor::openVideo(const QString &videoPath)
{
    // 强制 FFmpeg 软件解码，避免 OMX/DXVA2/VAAPI 等硬件解码器兼容性问题
    // 注意：OpenCV 4.5+ 可用 set(CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE)
    //       此处用环境变量禁用 FFmpeg 硬解（兼容 OpenCV 4.2）
    qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel=none");
    m_capture.open(videoPath.toLocal8Bit().constData(), cv::CAP_FFMPEG);
    if (!m_capture.isOpened()) {
        emit error("Cannot open video: " + videoPath);
        return false;
    }
    qDebug() << "[ONNXVideoProcessor] Video opened:" << videoPath;
    return true;
}

bool ONNXVideoProcessor::loadONNXModel(const QString &onnxPath,
                                        const QString &namesPath)
{
    // 1. 加载 ONNX 模型
    if (!m_detector.loadModel(onnxPath.toStdString())) {
        emit error("Failed to load ONNX model: " + onnxPath);
        return false;
    }

    // 2. 加载类别名称
    if (!namesPath.isEmpty()) {
        if (!m_detector.loadClassNames(namesPath.toStdString())) {
            qWarning() << "[ONNXVideoProcessor] Class names file not loaded:"
                       << namesPath << "(detection will show class index only)";
        }
    }

    qDebug() << "[ONNXVideoProcessor] ONNX model ready,"
             << "classes:" << m_detector.classCount()
             << "input size:" << m_detector.inputWidth() << "x" << m_detector.inputHeight();
    return true;
}

// ========== 运行时 ==========
void ONNXVideoProcessor::start()
{
    m_running = true;

    if (!m_timer) {
        m_timer = new QTimer(this);
        m_timer->setSingleShot(true);
        QObject::connect(m_timer, &QTimer::timeout,
                         this, &ONNXVideoProcessor::processFrame);
    }

    qDebug() << "[ONNXVideoProcessor] Started, thread:" << QThread::currentThread();
    processFrame();
}

void ONNXVideoProcessor::stop()
{
    m_running = false;
    if (m_timer)
        m_timer->stop();
}

void ONNXVideoProcessor::processFrame()
{
    if (!m_running) {
        qDebug() << "[ONNXVideoProcessor] Stopped";
        emit finished();
        return;
    }

    // ── 1. 读取一帧 ──
    cv::Mat frame;
    if (!m_capture.read(frame)) {
        qDebug() << "[ONNXVideoProcessor] Video ended";
        emit finished();
        return;
    }

    // ── 2. ONNX DNN 目标检测 ──
    std::vector<ONNXDetection> detections;
    m_detector.detect(frame, detections);

    // ── 3. 绘制检测框 ──
    if (m_drawBoxes) {
        for (const auto &det : detections) {
            // 绘制边界框
            cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);

            // 绘制标签背景和文字
            std::string label = det.className
                                + " " + std::to_string(int(det.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size labelSize = cv::getTextSize(label,
                                                  cv::FONT_HERSHEY_SIMPLEX,
                                                  0.5, 1, &baseline);

            cv::Rect labelRect(det.bbox.x, det.bbox.y - labelSize.height - 5,
                               labelSize.width, labelSize.height + 5);
            // 防止标签超出图像顶部
            if (labelRect.y < 0) {
                labelRect.y = det.bbox.y + det.bbox.height + 5;
            }

            cv::rectangle(frame, labelRect, cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label,
                        cv::Point(labelRect.x, labelRect.y + labelSize.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }

    // ── 4. 发送到主线程 ──
    QImage img = cvMatToQImage(frame);
    if (!img.isNull()) {
        emit frameReady(img);
    }

    // 同时发送检测结果（可选）
    if (!detections.empty()) {
        emit detectionsReady(detections);
    }

    // ── 5. 按目标帧率调度下一帧 ──
    int interval = 1000 / m_targetFps;
    if (interval < 5)
        interval = 5;

    m_timer->start(interval);
}
