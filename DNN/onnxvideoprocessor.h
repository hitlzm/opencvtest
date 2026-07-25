// onnxvideoprocessor.h — 基于 ONNX-DNN 的视频处理器（工作线程）
#ifndef ONNXVIDEOPROCESSOR_H
#define ONNXVIDEOPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <atomic>
#include <opencv2/videoio.hpp>
#include "onnxdetector.h"

/**
 * @brief 视频读取 + ONNX DNN 目标检测工作线程
 *
 * 与 DnnVideoProcessor（YOLOv4 Darknet）对称设计，可在 main.cpp 中
 * 按需替换。内部持有 ONNXDetector 实例，通过 QTimer 驱动逐帧
 * 处理，检测结果绘制到图像上后以 QImage 形式发送到主线程显示。
 *
 * 典型用法：
 * @code
 *   QThread *thread = new QThread;
 *   ONNXVideoProcessor *proc = new ONNXVideoProcessor;
 *   proc->openVideo("video.mp4");
 *   proc->loadONNXModel("yolov5s.onnx", "coco.names");
 *   proc->moveToThread(thread);
 *   // ... 连接信号/槽，启动线程
 * @endcode
 */
class ONNXVideoProcessor : public QObject
{
    Q_OBJECT
public:
    explicit ONNXVideoProcessor(QObject *parent = nullptr);
    ~ONNXVideoProcessor();

    // ── 初始化 ────────────────────────────────────────────
    /// 打开视频文件
    bool openVideo(const QString &videoPath);

    /**
     * @brief 一键加载 ONNX 模型 + 类别名称
     * @param onnxPath   .onnx 模型文件路径
     * @param namesPath  类别名称文件路径（如 coco.names），可选
     * @return true 全部加载成功
     */
    bool loadONNXModel(const QString &onnxPath,
                       const QString &namesPath = QString());

    /// 获取内部检测器引用，以便直接调节参数
    ONNXDetector &detector() { return m_detector; }

    // ── 参数调节 ──────────────────────────────────────────
    void setTargetFps(int fps);                         // 目标帧率，默认 30
    void setConfThreshold(float t);                     // 置信度阈值，默认 0.5
    void setNmsThreshold(float t);                      // NMS 阈值，默认 0.4
    void setInputSize(int width, int height);           // 网络输入尺寸

    /// 是否绘制检测框和标签（默认 true）
    void setDrawBoxes(bool draw) { m_drawBoxes = draw; }

    /// 设置后端（OpenCV CPU / CUDA 等）
    void setBackend(int backend, int target);

    bool isModelLoaded() const { return m_detector.isLoaded(); }

public slots:
    void start();            // 开始逐帧处理
    void processFrame();     // 处理一帧（由 QTimer 驱动）
    void stop();             // 停止处理

signals:
    /// 处理完成的帧图像（已绘制检测框）
    void frameReady(const QImage &frame);

    /// 每帧的检测结果（可选：如需在 QML 侧展示标签列表）
    void detectionsReady(const std::vector<ONNXDetection> &detections);

    void error(const QString &msg);
    void finished();

private:
    ONNXDetector m_detector;
    cv::VideoCapture m_capture;

    QTimer *m_timer = nullptr;
    std::atomic<bool> m_running;
    int m_targetFps = 30;

    bool m_drawBoxes = true;
};

#endif // ONNXVIDEOPROCESSOR_H
