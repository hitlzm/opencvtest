// onnxdetector.h — 基于 OpenCV DNN 的 ONNX 模型推理器
#ifndef ONNXDETECTOR_H
#define ONNXDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <QObject>

/**
 * @brief ONNX 模型单次检测结果
 */
struct ONNXDetection {
    int classId;                // 类别索引
    float confidence;           // 置信度 [0, 1]
    std::string className;      // 类别名称
    cv::Rect bbox;              // 边界框
};

/**
 * @brief 基于 OpenCV DNN 模块的 ONNX 模型推理器
 *
 * 使用 cv::dnn::readNetFromONNX() 加载 ONNX 格式模型（如
 * YOLOv5 / YOLOv8 / YOLOv11 等 PyTorch 模型导出的 .onnx），
 * 提供通用的前向推理和 YOLO 风格的目标检测后处理。
 *
 * 与 YoloDetector（cv::dnn::readNetFromDarknet）形成互补：
 * - YoloDetector 适合 YOLOv4 及之前的 Darknet 原生格式
 * - ONNXDetector 适合 PyTorch → ONNX 导出的现代 YOLO 系列
 *
 * 典型用法：
 * @code
 *   ONNXDetector detector;
 *   detector.loadModel("yolov5s.onnx");
 *   detector.loadClassNames("coco.names");
 *   detector.setConfThreshold(0.5f);
 *   detector.setNmsThreshold(0.4f);
 *
 *   std::vector<ONNXDetection> detections;
 *   detector.detect(frame, detections);
 *   for (auto &d : detections) {
 *       cv::rectangle(frame, d.bbox, cv::Scalar(0,255,0), 2);
 *   }
 * @endcode
 *
 * 也支持通用前向推理（不限于目标检测）：
 * @code
 *   std::vector<cv::Mat> outputs;
 *   detector.forward(frame, outputs);
 *   // 自行解析 outputs 中的数据
 * @endcode
 */
class ONNXDetector : public QObject
{
    Q_OBJECT
public:
    explicit ONNXDetector(QObject *parent = nullptr);
    ~ONNXDetector();

    // ── 模型加载 ──────────────────────────────────────────
    /**
     * @brief 加载 ONNX 模型
     * @param onnxPath  .onnx 模型文件路径
     * @return true 加载成功
     */
    bool loadModel(const std::string &onnxPath);

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

    // ── 预处理参数 ──────────────────────────────────────
    /**
     * @brief 设置网络输入尺寸（宽、高）
     *
     * ONNX 模型通常固定输入尺寸，须与导出时的 input size 一致。
     * 常用值: 416x416, 640x640, 640x384 等。
     */
    void setInputSize(int width, int height);

    /// 设置 blobFromImage 的 scale 因子（默认 1/255）
    void setScale(double scale);

    /// 设置 blobFromImage 的均值（默认全 0）
    void setMean(const cv::Scalar &mean);

    /// 是否交换 R/B 通道（默认 true）
    void setSwapRB(bool swap);

    /// 是否裁切（默认 false，等比缩放）
    void setCrop(bool crop);

    // ── 推理参数 ──────────────────────────────────────────
    void setConfThreshold(float thresh);        // 置信度阈值，默认 0.5
    void setNmsThreshold(float thresh);         // NMS IoU 阈值，默认 0.4

    /**
     * @brief 设置 DNN 后端和目标设备
     * @param backend  cv::dnn::DNN_BACKEND_OPENCV / DNN_BACKEND_CUDA
     * @param target   cv::dnn::DNN_TARGET_CPU / DNN_TARGET_CUDA
     *
     * 默认使用 OpenCV + CPU。如果编译了 CUDA 支持，可设为 CUDA/CUDA 加速。
     */
    void setBackend(int backend, int target);

    /**
     * @brief 某些 ONNX 模型已内置 NMS，设为 true 可跳过后处理中的 NMS
     */
    void setBuiltinNms(bool builtin) { m_builtinNms = builtin; }

    // ── 推理入口 ──────────────────────────────────────────

    /**
     * @brief 通用前向推理：输入图像，获取原始网络输出
     * @param frame   输入图像（BGR 彩色）
     * @param outputs 原始输出张量列表（各输出层）
     * @return true 推理成功
     *
     * 不进行任何后处理，适合分类、分割等非目标检测任务，
     * 或者用户需要自行解析输出格式时使用。
     */
    bool forward(const cv::Mat &frame, std::vector<cv::Mat> &outputs);

    /**
     * @brief 目标检测模式：前向推理 + YOLO 风格后处理
     * @param frame       输入图像（BGR 彩色）
     * @param detections  输出检测结果列表
     * @return true 至少检测到一个目标
     *
     * 适用于 YOLOv5/v8/v11 等 PyTorch → ONNX 导出的目标检测模型。
     * 自动处理常见的 ONNX 输出格式：
     *   - [1, N, 5+C]  3D 张量（行优先排列）
     *   - [1, 5+C, N]  3D 张量（列优先排列）
     *   - [N, 5+C]     2D 张量
     */
    bool detect(const cv::Mat &frame, std::vector<ONNXDetection> &detections);

    /// 模型是否已加载
    bool isLoaded() const { return m_loaded; }

    /// 当前类别数量
    int classCount() const { return (int)m_classNames.size(); }

    /// 获取输入尺寸
    int inputWidth()  const { return m_inputWidth; }
    int inputHeight() const { return m_inputHeight; }

    /// 获取输出层名称列表
    const std::vector<std::string>& outputNames() const { return m_outputNames; }

private:
    cv::dnn::Net m_net;                         // DNN 网络
    std::vector<std::string> m_classNames;      // 类别名称表
    std::vector<std::string> m_outputNames;     // 输出层名称

    bool m_loaded = false;

    // 预处理参数
    int        m_inputWidth  = 640;             // 默认 YOLOv5 输入尺寸
    int        m_inputHeight = 640;
    double     m_scale       = 1.0 / 255.0;
    cv::Scalar m_mean        = cv::Scalar(0, 0, 0);
    bool       m_swapRB      = true;
    bool       m_crop        = false;

    // 后处理参数
    float m_confThreshold = 0.5f;
    float m_nmsThreshold  = 0.4f;
    bool  m_builtinNms    = false;

    // ── 内部辅助 ──────────────────────────────────────────
    /// 获取所有未连接的输出层名称
    std::vector<std::string> getOutputNames();

    /// 判断输出张量是否为 "列优先" 排列（[1, 5+C, N] 而非 [1, N, 5+C]）
    static bool isTransposedLayout(const cv::Mat &output);

    /// YOLO 风格后处理：解析网络输出 → 检测结果
    void postProcess(const cv::Mat &frame,
                     const std::vector<cv::Mat> &outputs,
                     std::vector<ONNXDetection> &detections);
};

#endif // ONNXDETECTOR_H
