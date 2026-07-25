// onnxyolodetector.h — 基于 ONNX Runtime 的 YOLOv3-tiny 目标检测器
//
// 与 YoloDetector（OpenCV DNN Darknet 后端）接口兼容，可互换使用。
// 区别在于本类使用 ONNX Runtime 加载 .onnx 模型文件进行推理。
//
// 使用 PIMPL 模式隐藏 ONNX Runtime 头文件依赖，
// 使用者只需链接 onnxruntime 库即可。
//
// 典型用法：
// @code
//   OnnxYoloDetector detector;
//   detector.loadModel("yolov3-tiny.onnx");
//   detector.loadClassNames("coco.names");
//   detector.setConfThreshold(0.5f);
//
//   std::vector<OnnxDetection> detections;
//   detector.detect(frame, detections);
// @endcode

#ifndef ONNXYOLODETECTOR_H
#define ONNXYOLODETECTOR_H

#include <QObject>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * @brief 单个目标检测结果
 */
struct OnnxDetection {
    int classId = -1;
    float confidence = 0.0f;
    std::string className;
    cv::Rect bbox;
};

// PIMPL 实现类（内部持有 ONNX Runtime 对象，定义在 .cpp 中）
struct OnnxYoloDetectorImpl;

/**
 * @brief 基于 ONNX Runtime 的 YOLOv3-tiny 目标检测器
 *
 * 加载 .onnx 模型，对 BGR 图像执行前向推理，返回检测到的目标列表。
 *
 * YOLOv3-tiny 网络：
 *  - 输入:  [1, 3, H, W]  NCHW, 归一化 [0,1]
 *  - 输出1: [1, 3*(5+numClasses), H/32, W/32]  大尺度
 *  - 输出2: [1, 3*(5+numClasses), H/16, W/16]  小尺度
 *
 * 线程安全：detect() 访问 ONNX Session，不支持并发调用，
 * 调用侧需自行串行化。
 */
class OnnxYoloDetector : public QObject
{
    Q_OBJECT
public:
    explicit OnnxYoloDetector(QObject *parent = nullptr);
    ~OnnxYoloDetector() override;

    // ── 模型加载 ──────────────────────────────────────────
    /// 加载 YOLOv3-tiny ONNX 模型，自动枚举输入输出节点
    bool loadModel(const std::string &onnxPath);

    /// 加载类别名称文件（每行一个类别名）
    bool loadClassNames(const std::string &namesPath);

    /// 直接设置类别名称列表
    void setClassNames(const std::vector<std::string> &names);

    // ── 参数调节 ──────────────────────────────────────────
    void setConfThreshold(float thresh);         // 置信度阈值，默认 0.5
    void setNmsThreshold(float thresh);          // NMS IoU 阈值，默认 0.4
    void setInputSize(int width, int height);    // 网络输入尺寸，默认 416×416
    void setNumClasses(int n);                   // 类别数量，默认 80
    void setNumThreads(int n);                   // ONNX Runtime 线程数，默认 4

    // ── 检测入口 ──────────────────────────────────────────
    /**
     * @brief 在 BGR 图像上执行 YOLOv3-tiny 检测
     * @param frame       输入图像（BGR, CV_8UC3）
     * @param detections  输出检测结果列表
     * @return true 至少检测到一个目标
     */
    bool detect(const cv::Mat &frame, std::vector<OnnxDetection> &detections);

    bool isLoaded() const { return m_loaded; }
    int classCount() const { return static_cast<int>(m_classNames.size()); }

signals:
    void errorOccurred(const QString &message);

private:
    std::unique_ptr<OnnxYoloDetectorImpl> m_impl;

    std::vector<std::string> m_classNames;

    float m_confThreshold = 0.5f;
    float m_nmsThreshold = 0.4f;
    int   m_inputWidth    = 416;
    int   m_inputHeight   = 416;
    int   m_numClasses    = 80;
    int   m_numThreads    = 4;
    bool  m_loaded        = false;

    // LetterBox 预处理参数（detect 时计算，供后处理坐标还原用）
    float m_letterBoxScale = 1.0f;
    int   m_letterBoxPadX  = 0;
    int   m_letterBoxPadY  = 0;
};

#endif // ONNXYOLODETECTOR_H
