// dnnvideoprocessor.h — 基于 YOLOv4-DNN 的视频处理器（工作线程）
#ifndef DNNVIDEOPROCESSOR_H
#define DNNVIDEOPROCESSOR_H

#include <QObject>
#include <QImage>
#include <QTimer>
#include <atomic>
#include <opencv2/videoio.hpp>
#include "yolodetector.h"

/**
 * @brief 视频读取 + YOLOv4 DNN 目标检测工作线程
 *
 * 与 VideoProcessor（ORB 特征匹配）对称设计，可在 main.cpp 中
 * 按需替换。内部持有 YoloDetector 实例，通过 QTimer 驱动逐帧
 * 处理，检测结果绘制到图像上后以 QImage 形式发送到主线程显示。
 *
 * 典型用法：
 * @code
 *   QThread *thread = new QThread;
 *   DnnVideoProcessor *proc = new DnnVideoProcessor;
 *   proc->openVideo("video.mp4");
 *   proc->loadYoloModel("yolov4.cfg", "yolov4.weights", "coco.names");
 *   proc->moveToThread(thread);
 *   // ... 连接信号/槽，启动线程
 * @endcode
 */
class DnnVideoProcessor : public QObject
{
    Q_OBJECT
public:
    explicit DnnVideoProcessor(QObject *parent = nullptr);
    ~DnnVideoProcessor();

    // ── 初始化 ────────────────────────────────────────────
    /// 打开视频文件
    bool openVideo(const QString &videoPath);

    /**
     * @brief 一键加载 YOLOv4 模型 + 类别名称
     * @param cfgPath     .cfg 配置文件路径
     * @param weightsPath .weights 权重文件路径
     * @param namesPath   类别名称文件路径（如 coco.names）
     * @return true 全部加载成功
     */
    bool loadYoloModel(const QString &cfgPath,
                       const QString &weightsPath,
                       const QString &namesPath = QString());

    // ── 参数调节 ──────────────────────────────────────────
    void setTargetFps(int fps);                         // 目标帧率，默认 30
    void setConfThreshold(float t);                     // 置信度阈值，默认 0.5
    void setNmsThreshold(float t);                      // NMS 阈值，默认 0.4
    void setInputSize(int width, int height);           // 网络输入尺寸，默认 416x416

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
    void detectionsReady(const std::vector<YoloDetection> &detections);

    void error(const QString &msg);
    void finished();

private:
    YoloDetector m_detector;
    cv::VideoCapture m_capture;

    QTimer *m_timer = nullptr;
    std::atomic<bool> m_running;
    int m_targetFps = 30;

    bool m_drawBoxes = true;
};

#endif // DNNVIDEOPROCESSOR_H
