// streamprocessor.h — 视频帧 ONNX YOLOv3-tiny 检测处理器
//
// 从本地视频文件读取帧，经 ONNX Runtime 推理引擎进行目标检测，
// 绘制检测结果后通过 frameReady 信号对外广播。
//
// 数据流：
//   cv::VideoCapture::read()            — 读取本地视频帧
//        ↓
//   StreamProcessor::processFrame()      — QTimer 驱动，工作线程
//        ↓  cv::Mat(BGR)
//        ↓  图像预处理 / ONNX Runtime 检测 / 绘制检测框
//        ↓  cv::Mat → QImage(RGB)  (via cvMatToQImage)
//        ↓
//   emit frameReady(img)                 — 广播到主线程显示
//
// 典型用法：
// @code
//   QThread *thread = new QThread;
//   StreamProcessor *proc = new StreamProcessor;
//   proc->openVideo("video.mp4");
//   proc->loadYoloModel("yolov3-tiny.onnx", "coco.names");
//   proc->moveToThread(thread);
//   QObject::connect(thread, &QThread::started, proc, &StreamProcessor::start);
//   QObject::connect(proc, &StreamProcessor::frameReady, ...);
//   thread->start();
// @endcode

#ifndef STREAMPROCESSOR_H
#define STREAMPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <atomic>
#include <opencv2/videoio.hpp>
#include "onnxyolodetector.h"

/**
 * @brief 视频帧处理工作线程（本地视频 + ONNX Runtime 检测）
 *
 * 运行在独立 QThread 中。通过 cv::VideoCapture 读取本地视频文件，
 * 使用 ONNX Runtime 推理引擎进行 YOLOv3-tiny 目标检测。
 *
 * OpenCV 用于：
 *  1. cv::VideoCapture 读取本地视频帧
 *  2. 可选的基础图像处理（去噪、增强等）
 *  3. 绘制检测框和标签
 *  4. cv::Mat → QImage 转换 (cvMatToQImage)
 *
 * ONNX Runtime 用于：
 *  1. 加载 .onnx 模型文件
 *  2. 执行 YOLOv3-tiny 前向推理
 *  3. 后处理解码检测结果
 */
class StreamProcessor : public QObject
{
    Q_OBJECT
public:
    explicit StreamProcessor(QObject *parent = nullptr);
    ~StreamProcessor() override;

    // ── 视频源 ────────────────────────────────────────────
    /// 打开本地视频文件
    bool openVideo(const QString &videoPath);

    // ── 模型加载 ──────────────────────────────────────────
    /// 加载 ONNX 模型 + 可选的类别名称文件
    bool loadYoloModel(const QString &onnxPath,
                       const QString &namesPath = QString());

    // ── 参数调节 ──────────────────────────────────────────
    void setTargetFps(int fps);
    void setConfThreshold(float t);
    void setNmsThreshold(float t);
    void setInputSize(int width, int height);

    void setDrawBoxes(bool draw) { m_drawBoxes = draw; }
    void setBoxColor(int b, int g, int r) {
        m_boxColor = cv::Scalar(b, g, r);
    }

    // ── 基础图像处理开关 ─────────────────────────────────
    void setAutoEnhance(bool enable)  { m_autoEnhance = enable; }
    void setHistogramEqualization(bool enable) { m_histEq = enable; }
    void setDenoise(bool enable)      { m_denoise = enable; }
    void setDisplaySize(int width, int height) {
        m_displayWidth = width;
        m_displayHeight = height;
    }

    // ── 状态查询 ──────────────────────────────────────────
    bool isModelLoaded() const { return m_detector.isLoaded(); }
    bool isVideoOpened() const { return m_capture.isOpened(); }

    /// 最新检测框中置信度最高的目标中心像素坐标（-1 表示无检测结果）
    int centerX() const { return m_centerX; }
    int centerY() const { return m_centerY; }

public slots:
    void start();
    void processFrame();
    void stop();

signals:
    void frameReady(const QImage &frame);
    void detectionsReady(const std::vector<OnnxDetection> &detections);
    void errorOccurred(const QString &msg);
    void finished();

private:
    void applyImageProcessing(cv::Mat &frame);
    void drawDetections(cv::Mat &frame,
                        const std::vector<OnnxDetection> &detections);

    OnnxYoloDetector m_detector;
    cv::VideoCapture m_capture;              // 本地视频读取器

    QTimer *m_timer = nullptr;
    std::atomic<bool> m_running{false};

    int m_targetFps = 30;

    // 绘制参数
    bool m_drawBoxes = true;
    cv::Scalar m_boxColor = cv::Scalar(0, 255, 0);

    // 图像处理参数
    bool m_autoEnhance = false;
    bool m_histEq     = false;
    bool m_denoise    = false;
    int  m_displayWidth  = 0;
    int  m_displayHeight = 0;

    // 检测框中心点（最高置信度目标）
    // 外引导模式时可根据CCD视场角得到转台的方位角与俯仰角应转动的角度
    int m_centerX = -1;
    int m_centerY = -1;
};

#endif // STREAMPROCESSOR_H
