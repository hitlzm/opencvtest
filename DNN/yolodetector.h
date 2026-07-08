// yolodetector.h — 基于 OpenCV DNN 的 YOLOv4 目标检测器
#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <QObject>

/**
 * @brief 单个检测结果
 */
struct YoloDetection {
    int classId;                // 类别索引
    float confidence;           // 置信度 [0, 1]
    std::string className;      // 类别名称
    cv::Rect bbox;              // 边界框
};

/**
 * @brief 基于 OpenCV DNN 模块的 YOLOv4 检测器
 *
 * 使用 cv::dnn::readNetFromDarknet() 加载训练好的 YOLOv4 模型
 * （.cfg 配置文件 + .weights 权重文件），在图像/视频帧上执行
 * 目标检测，返回类别、置信度和边界框。
 *
 * 典型用法：
 * @code
 *   YoloDetector detector;
 *   detector.loadModel("yolov4.cfg", "yolov4.weights");
 *   detector.loadClassNames("coco.names");
 *   detector.setConfThreshold(0.5f);
 *   detector.setNmsThreshold(0.4f);
 *
 *   std::vector<YoloDetection> detections;
 *   detector.detect(frame, detections);
 *   for (auto &d : detections) {
 *       cv::rectangle(frame, d.bbox, cv::Scalar(0,255,0), 2);
 *   }
 * @endcode
 */
class YoloDetector : public QObject
{
    Q_OBJECT
public:
    explicit YoloDetector(QObject *parent = nullptr);
    ~YoloDetector();

    // ── 模型加载 ──────────────────────────────────────────
    /**
     * @brief 加载 YOLOv4 Darknet 模型
     * @param cfgPath     .cfg 配置文件路径
     * @param weightsPath .weights 权重文件路径
     * @return true 加载成功
     */
    bool loadModel(const std::string &cfgPath, const std::string &weightsPath);

    /**
     * @brief 加载类别名称文件（每行一个类别名，如 coco.names）
     * @param namesPath 类别文件路径
     * @return true 加载成功
     */
    bool loadClassNames(const std::string &namesPath);

    /**
     * @brief 直接设置类别名称列表
     */
    void setClassNames(const std::vector<std::string> &names);

    // ── 参数调节 ──────────────────────────────────────────
    void setConfThreshold(float thresh);        // 置信度阈值，默认 0.5
    void setNmsThreshold(float thresh);         // NMS IoU 阈值，默认 0.4
    void setInputSize(int width, int height);   // 网络输入尺寸，默认 416x416

    /**
     * @brief 设置 DNN 后端和目标设备
     * @param backend  cv::dnn::DNN_BACKEND_OPENCV / DNN_BACKEND_CUDA
     * @param target   cv::dnn::DNN_TARGET_CPU / DNN_TARGET_CUDA
     *
     * 默认使用 OpenCV + CPU。如果编译了 CUDA 支持，可设为 CUDA/CUDA 加速。
     */
    void setBackend(int backend, int target);

    // ── 检测入口 ──────────────────────────────────────────
    /**
     * @brief 在单帧图像上执行 YOLOv4 检测
     * @param frame       输入图像（BGR 彩色）
     * @param detections  输出检测结果列表
     * @return true 至少检测到一个目标
     */
    bool detect(const cv::Mat &frame, std::vector<YoloDetection> &detections);

    /// 模型是否已加载
    bool isLoaded() const { return m_loaded; }

    /// 当前类别数量
    int classCount() const { return (int)m_classNames.size(); }

private:
    cv::dnn::Net m_net;                         // DNN 网络
    std::vector<std::string> m_classNames;      // 类别名称表
    std::vector<std::string> m_outputNames;     // 输出层名称

    bool m_loaded = false;

    float m_confThreshold = 0.5f;               // 置信度阈值
    float m_nmsThreshold = 0.4f;                // NMS IoU 阈值
    int   m_inputWidth  = 416;                  // 网络输入宽度
    int   m_inputHeight = 416;                  // 网络输入高度

    // ── 内部辅助 ──────────────────────────────────────────
    /// 获取 YOLO 输出层的名称（YOLOv4 通常有 3 个 yolo 层）
    std::vector<std::string> getOutputNames();

    /// 后处理：解析网络输出 → 检测结果
    void postProcess(const cv::Mat &frame,
                     const std::vector<cv::Mat> &outputs,
                     std::vector<YoloDetection> &detections);
};

#endif // YOLODETECTOR_H
