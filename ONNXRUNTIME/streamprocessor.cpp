// streamprocessor.cpp — 从本地视频取帧 → ONNX Runtime 检测 → 广播显示
//
// ══════════════════════════════════════════════════════════════════════════════
//  数据流
// ══════════════════════════════════════════════════════════════════════════════
//
//  本文件运行在【工作线程】中。
//  使用 cv::VideoCapture 读取本地视频文件，ONNX Runtime 进行推理。
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │ 步骤① 读取帧                             （工作线程 / QTimer）            │
// │   m_capture.read(frame)                                                   │
// │   → cv::Mat (BGR)                                                        │
// │                                                                          │
// │ 步骤② 基础图像处理【可选】               （工作线程）                      │
// │   applyImageProcessing(frame)                                            │
// │                                                                          │
// │ 步骤③ ONNX Runtime YOLOv3-tiny 推理      （工作线程）                      │
// │   m_detector.detect(frame, detections)                                    │
// │                                                                          │
// │ 步骤④ 绘制检测框                         （工作线程）                      │
// │   drawDetections(frame, detections)                                       │
// │                                                                          │
// │ 步骤⑤ 显示缩放【可选】                   （工作线程）                      │
// │   cv::resize → display size                                              │
// │                                                                          │
// │ 步骤⑥ cv::Mat → QImage 转换 + 广播       （工作线程 → 主线程）             │
// │   cvMatToQImage(frame) → emit frameReady(img)                            │
// └──────────────────────────────────────────────────────────────────────────┘
//
// ══════════════════════════════════════════════════════════════════════════════

#include "streamprocessor.h"
#include "opencv/cvconvert.h"
#include <QDebug>
#include <QElapsedTimer>
#include <QThread>
#include <opencv2/imgproc.hpp>

// ══════════════════════════════════════════════════════════════════════════════
// 构造 / 析构
// ══════════════════════════════════════════════════════════════════════════════

StreamProcessor::StreamProcessor(QObject *parent)
    : QObject(parent)
    , m_running(false)
{
}

StreamProcessor::~StreamProcessor() {
    stop();
    if (m_capture.isOpened())
        m_capture.release();
}

// ══════════════════════════════════════════════════════════════════════════════
// 视频源
// ══════════════════════════════════════════════════════════════════════════════

bool StreamProcessor::openVideo(const QString &videoPath) {
    // 强制 FFmpeg 软件解码，避免 OMX/DXVA2/VAAPI 等硬件解码器兼容性问题
    // 注意：OpenCV 4.5+ 可用 set(CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE)
    //       此处用环境变量禁用 FFmpeg 硬解（兼容 OpenCV 4.2）
    qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel=none");
    m_capture.open(videoPath.toLocal8Bit().constData(), cv::CAP_FFMPEG);
    if (!m_capture.isOpened()) {
        emit errorOccurred("Cannot open video: " + videoPath);
        return false;
    }
    qDebug() << "[StreamProcessor] Video opened:" << videoPath;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// 参数设置
// ══════════════════════════════════════════════════════════════════════════════

void StreamProcessor::setTargetFps(int fps) {
    if (fps > 0 && fps <= 60)
        m_targetFps = fps;
}

void StreamProcessor::setConfThreshold(float t) { m_detector.setConfThreshold(t); }
void StreamProcessor::setNmsThreshold(float t)  { m_detector.setNmsThreshold(t); }

void StreamProcessor::setInputSize(int width, int height) {
    m_detector.setInputSize(width, height);
}

// ══════════════════════════════════════════════════════════════════════════════
// 模型加载
// ══════════════════════════════════════════════════════════════════════════════

bool StreamProcessor::loadYoloModel(const QString &onnxPath,
                                    const QString &namesPath) {
    if (!m_detector.loadModel(onnxPath.toStdString())) {
        emit errorOccurred(QString("Failed to load ONNX model: %1").arg(onnxPath));
        return false;
    }

    if (!namesPath.isEmpty()) {
        if (!m_detector.loadClassNames(namesPath.toStdString())) {
            qWarning() << "[StreamProcessor] Class names file not loaded:"
                       << namesPath << "(will show class index instead)";
        }
    }

    qDebug() << "[StreamProcessor] YOLOv3-tiny ONNX model ready,"
             << "classes:" << m_detector.classCount()
             << "input:" << 416 << "x" << 416;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// 基础图像处理
// ══════════════════════════════════════════════════════════════════════════════

void StreamProcessor::applyImageProcessing(cv::Mat &frame) {
    if (frame.empty()) return;

    if (m_denoise) {
        cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0.5);
    }

    if (m_autoEnhance) {
        cv::Mat lab;
        cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> labChannels(3);
        cv::split(lab, labChannels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(labChannels[0], labChannels[0]);
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, frame, cv::COLOR_Lab2BGR);
    }

    if (m_histEq) {
        cv::Mat yuv;
        cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV);
        std::vector<cv::Mat> yuvChannels(3);
        cv::split(yuv, yuvChannels);
        cv::equalizeHist(yuvChannels[0], yuvChannels[0]);
        cv::merge(yuvChannels, yuv);
        cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 绘制检测框
// ══════════════════════════════════════════════════════════════════════════════

void StreamProcessor::drawDetections(
    cv::Mat &frame,
    const std::vector<OnnxDetection> &detections)
{
    // 找置信度最高的检测框，记录其中心坐标
    const OnnxDetection *bestDet = nullptr;
    float bestConf = 0.0f;

    for (const auto &det : detections) {
        cv::rectangle(frame, det.bbox, m_boxColor, 2);

        std::string label = det.className
                            + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        int baseline = 0;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Size labelSize = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX,
            fontScale, thickness, &baseline);

        int labelY = det.bbox.y - labelSize.height - 5;
        if (labelY < 0)
            labelY = det.bbox.y + det.bbox.height + 5;

        cv::Rect labelRect(det.bbox.x, labelY,
                          labelSize.width, labelSize.height + 5);
        if (labelRect.x + labelRect.width > frame.cols)
            labelRect.x = frame.cols - labelRect.width - 2;
        if (labelRect.x < 0)
            labelRect.x = 2;

        cv::rectangle(frame, labelRect, m_boxColor, cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(labelRect.x, labelRect.y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);

        // 追踪最高置信度目标
        if (det.confidence > bestConf) {
            bestConf = det.confidence;
            bestDet = &det;
        }
    }

    // 存储最高置信度检测框的中心像素坐标
    if (bestDet) {
        m_centerX = bestDet->bbox.x + bestDet->bbox.width / 2;
        m_centerY = bestDet->bbox.y + bestDet->bbox.height / 2;
    } else {
        m_centerX = -1;
        m_centerY = -1;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 运行时控制
// ══════════════════════════════════════════════════════════════════════════════

void StreamProcessor::start() {
    m_running = true;

    if (!m_timer) {
        m_timer = new QTimer(this);
        m_timer->setSingleShot(true);
        QObject::connect(m_timer, &QTimer::timeout,
                         this, &StreamProcessor::processFrame);
    }

    qDebug() << "[StreamProcessor] Started, thread:" << QThread::currentThread();
    processFrame();
}

void StreamProcessor::stop() {
    m_running = false;
    if (m_timer)
        m_timer->stop();
}

// ══════════════════════════════════════════════════════════════════════════════
// 核心帧处理循环
//
// QTimer 驱动，工作线程中执行。
// 从本地视频文件读取帧 → 图像预处理 → ONNX Runtime 推理 → 绘制 → 广播。
// ══════════════════════════════════════════════════════════════════════════════

void StreamProcessor::processFrame() {

    if (!m_running) {
        qDebug() << "[StreamProcessor] Stopped";
        emit finished();
        return;
    }

    // 固定目标帧间隔（ms）
    const int targetInterval = std::max(5, 1000 / m_targetFps);
    // ── 帧率控制起点：记录本次处理开始时间 ──
    QElapsedTimer frameTimer;
    frameTimer.start();

    // ── ① 从本地视频文件读取一帧 ──
    cv::Mat frame;
    if (!m_capture.read(frame) || frame.empty()) {
        // 视频结束或读取失败
        qDebug() << "[StreamProcessor] Video ended or read failed";
        emit finished();
        return;
    }

    // ── ② 基础图像处理 ──
    applyImageProcessing(frame);

    // ── ③ ONNX Runtime YOLOv3-tiny 目标检测 ──
    std::vector<OnnxDetection> detections;
    if (isModelLoaded()) {
        m_detector.detect(frame, detections);
    }

    // ── ④ 绘制检测框 ──
    if (m_drawBoxes && !detections.empty()) {
        drawDetections(frame, detections);
    }

    // ── ⑤ 显示缩放（仅影响最终输出，不改变 ONNX 输入分辨率）──
    if (m_displayWidth > 0 && m_displayHeight > 0) {
        cv::resize(frame, frame, cv::Size(m_displayWidth, m_displayHeight),
                   0, 0, cv::INTER_LINEAR);
    }

    // ── ⑥ cv::Mat → QImage 转换（使用 opencv/cvconvert.h）──
    QImage img = cvMatToQImage(frame);

    // ── ⑦ 广播到主线程显示 ──
    if (!img.isNull()) {
        emit frameReady(img);
    }

    if (!detections.empty()) {
        emit detectionsReady(detections);
    }

    // ── ⑧ 帧率控制：按实际耗时动态调节 ──
    //  耗时 < 目标间隔 → 等剩余时间，精准控帧
    //  耗时 ≥ 目标间隔 → 立即下一帧，不积压

    int elapsed  = static_cast<int>(frameTimer.elapsed());
    int remaining = targetInterval - elapsed;

    if (remaining > 0) {
        m_timer->start(remaining);
    } else {
        m_timer->start(0);  // 已超时，立即进入下一轮
    }
}
