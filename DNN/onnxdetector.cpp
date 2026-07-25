#include "onnxdetector.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <QDebug>

// ========== 构造 / 析构 ==========
ONNXDetector::ONNXDetector(QObject *parent)
    : QObject(parent)
{
}

ONNXDetector::~ONNXDetector() {}

// ========== 参数设置 ==========
void ONNXDetector::setInputSize(int width, int height)
{
    m_inputWidth  = width;
    m_inputHeight = height;
}

void ONNXDetector::setScale(double scale)          { m_scale  = scale; }
void ONNXDetector::setMean(const cv::Scalar &mean) { m_mean   = mean; }
void ONNXDetector::setSwapRB(bool swap)             { m_swapRB = swap; }
void ONNXDetector::setCrop(bool crop)               { m_crop   = crop; }

void ONNXDetector::setConfThreshold(float thresh) { m_confThreshold = thresh; }
void ONNXDetector::setNmsThreshold(float thresh)  { m_nmsThreshold  = thresh; }

void ONNXDetector::setBackend(int backend, int target)
{
    m_net.setPreferableBackend(backend);
    m_net.setPreferableTarget(target);
}

// ========== 模型加载 ==========
bool ONNXDetector::loadModel(const std::string &onnxPath)
{
    try {
        m_net = cv::dnn::readNetFromONNX(onnxPath);

        // 默认使用 OpenCV 后端 + CPU
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        m_outputNames = getOutputNames();

        if (m_outputNames.empty()) {
            std::cerr << "[ONNXDetector] No output layers found in ONNX model." << std::endl;
            return false;
        }

        m_loaded = true;
        qDebug() << "[ONNXDetector] ONNX model loaded, output layers:" << m_outputNames.size();
        return true;

    } catch (const cv::Exception &e) {
        std::cerr << "[ONNXDetector] Failed to load ONNX model: " << e.what() << std::endl;
        return false;
    }
}

bool ONNXDetector::loadClassNames(const std::string &namesPath)
{
    std::ifstream file(namesPath);
    if (!file.is_open()) {
        std::cerr << "[ONNXDetector] Cannot open class names file: " << namesPath << std::endl;
        return false;
    }

    m_classNames.clear();
    std::string line;
    while (std::getline(file, line)) {
        // 去掉末尾回车
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        m_classNames.push_back(line);
    }
    file.close();

    qDebug() << "[ONNXDetector] Loaded" << m_classNames.size() << "class names";
    return !m_classNames.empty();
}

void ONNXDetector::setClassNames(const std::vector<std::string> &names)
{
    m_classNames = names;
}

// ========== 获取输出层名称 ==========
std::vector<std::string> ONNXDetector::getOutputNames()
{
    std::vector<std::string> names;
    std::vector<int> outLayers = m_net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = m_net.getLayerNames();

    // OpenCV 的 getLayerNames() 在部分版本返回 0-based 索引，
    // getUnconnectedOutLayers() 返回 1-based 索引，需要转换
    for (int idx : outLayers) {
        int i = (idx > 0 && idx <= (int)layersNames.size()) ? idx - 1 : idx;
        if (i >= 0 && i < (int)layersNames.size())
            names.push_back(layersNames[i]);
    }
    return names;
}

// ========== 前向推理 ==========
bool ONNXDetector::forward(const cv::Mat &frame, std::vector<cv::Mat> &outputs)
{
    outputs.clear();

    if (!m_loaded) {
        std::cerr << "[ONNXDetector] Model not loaded." << std::endl;
        return false;
    }
    if (frame.empty()) {
        std::cerr << "[ONNXDetector] Input frame is empty." << std::endl;
        return false;
    }

    // ── 1. 预处理：将输入图像转为 blob ──
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, m_scale,
                           cv::Size(m_inputWidth, m_inputHeight),
                           m_mean, m_swapRB, m_crop);

    // ── 2. 前向推理 ──
    m_net.setInput(blob);
    m_net.forward(outputs, m_outputNames);

    return !outputs.empty();
}

// ========== 检测入口 ==========
bool ONNXDetector::detect(const cv::Mat &frame, std::vector<ONNXDetection> &detections)
{
    detections.clear();

    std::vector<cv::Mat> outputs;
    if (!forward(frame, outputs))
        return false;

    // ── 后处理 ──
    postProcess(frame, outputs, detections);

    return !detections.empty();
}

// ========== 判断输出布局是否转置 ==========
bool ONNXDetector::isTransposedLayout(const cv::Mat &output)
{
    if (output.dims < 3)
        return false;

    // ONNX YOLO 输出的两种常见形状:
    //   [1, N, 5+C]  — 行优先，通道维度在最后（常见）
    //   [1, 5+C, N]  — 转置，通道维度在中间
    // 如果 size[1] 较小（<= 100）则很可能是 [1, 5+C, N] 格式
    int dim1 = output.size[1];
    int dim2 = output.size[2];

    // 通道数（5 + numClasses）通常 ≤ 100，预测框数 N 通常 > 100
    return (dim1 <= 100 && dim2 > 100);
}

// ========== YOLO 风格后处理 ==========
void ONNXDetector::postProcess(const cv::Mat &frame,
                                const std::vector<cv::Mat> &outputs,
                                std::vector<ONNXDetection> &detections)
{
    int frameW = frame.cols;
    int frameH = frame.rows;

    std::vector<int>       classIds;
    std::vector<float>     confidences;
    std::vector<cv::Rect>  boxes;

    // ── 遍历所有输出层 ──
    for (const auto &output : outputs) {
        const float *data = (const float *)output.data;

        int numPredictions = 0;   // N（预测框数量）
        int numValues      = 0;   // 5 + numClasses（每行值数量）

        if (output.dims == 3) {
            // 3D 输出: [1, N, 5+C] 或 [1, 5+C, N]
            if (isTransposedLayout(output)) {
                // shape: [1, 5+C, N]  — 转置格式
                numValues      = output.size[1];
                numPredictions = output.size[2];
                int colStep    = numPredictions;  // 跨预测步长

                for (int i = 0; i < numPredictions; ++i) {
                    float objConf = data[i + 4 * colStep];

                    float maxClassScore = 0;
                    int   maxClassId    = -1;
                    for (int c = 5; c < numValues; ++c) {
                        float score = data[i + c * colStep];
                        if (score > maxClassScore) {
                            maxClassScore = score;
                            maxClassId    = c - 5;
                        }
                    }

                    float confidence = objConf * maxClassScore;
                    if (confidence < m_confThreshold)
                        continue;

                    float cx = data[i + 0 * colStep];
                    float cy = data[i + 1 * colStep];
                    float w  = data[i + 2 * colStep];
                    float h  = data[i + 3 * colStep];

                    int left   = int((cx - w * 0.5f) * frameW);
                    int top    = int((cy - h * 0.5f) * frameH);
                    int width  = int(w * frameW);
                    int height = int(h * frameH);

                    classIds.push_back(maxClassId);
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            } else {
                // shape: [1, N, 5+C]  — 标准格式
                numPredictions = output.size[1];
                numValues      = output.size[2];

                for (int i = 0; i < numPredictions; ++i) {
                    const float *row = data + i * numValues;

                    float objConf = row[4];

                    float maxClassScore = 0;
                    int   maxClassId    = -1;
                    for (int c = 5; c < numValues; ++c) {
                        if (row[c] > maxClassScore) {
                            maxClassScore = row[c];
                            maxClassId    = c - 5;
                        }
                    }

                    float confidence = objConf * maxClassScore;
                    if (confidence < m_confThreshold)
                        continue;

                    float cx = row[0];
                    float cy = row[1];
                    float w  = row[2];
                    float h  = row[3];

                    int left   = int((cx - w * 0.5f) * frameW);
                    int top    = int((cy - h * 0.5f) * frameH);
                    int width  = int(w * frameW);
                    int height = int(h * frameH);

                    classIds.push_back(maxClassId);
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        } else if (output.dims == 2) {
            // 2D 输出: [N, 5+C]
            numPredictions = output.rows;
            numValues      = output.cols;

            for (int i = 0; i < numPredictions; ++i) {
                const float *row = data + i * numValues;

                float objConf = row[4];

                float maxClassScore = 0;
                int   maxClassId    = -1;
                for (int c = 5; c < numValues; ++c) {
                    if (row[c] > maxClassScore) {
                        maxClassScore = row[c];
                        maxClassId    = c - 5;
                    }
                }

                float confidence = objConf * maxClassScore;
                if (confidence < m_confThreshold)
                    continue;

                float cx = row[0];
                float cy = row[1];
                float w  = row[2];
                float h  = row[3];

                int left   = int((cx - w * 0.5f) * frameW);
                int top    = int((cy - h * 0.5f) * frameH);
                int width  = int(w * frameW);
                int height = int(h * frameH);

                classIds.push_back(maxClassId);
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        } else {
            // 1D 或未知维度，打印警告跳过
            std::cerr << "[ONNXDetector] Unexpected output dims: " << output.dims
                      << ", skipping this output layer." << std::endl;
            continue;
        }
    }

    // ── 非极大值抑制 (NMS) ──
    std::vector<int> indices;
    if (m_builtinNms) {
        // 模型已内置 NMS，直接使用全部结果（按原始顺序）
        indices.reserve(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i)
            indices.push_back((int)i);
    } else {
        cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
    }

    // ── 组装最终结果 ──
    detections.reserve(indices.size());
    for (int idx : indices) {
        ONNXDetection det;
        det.classId    = classIds[idx];
        det.confidence = confidences[idx];
        det.bbox       = boxes[idx];

        // 裁剪到图像范围
        det.bbox.x = std::max(0, det.bbox.x);
        det.bbox.y = std::max(0, det.bbox.y);
        if (det.bbox.x + det.bbox.width  > frameW) det.bbox.width  = frameW - det.bbox.x;
        if (det.bbox.y + det.bbox.height > frameH) det.bbox.height = frameH - det.bbox.y;

        // 查找类别名称
        if (det.classId >= 0 && det.classId < (int)m_classNames.size())
            det.className = m_classNames[det.classId];
        else
            det.className = "unknown";

        detections.push_back(det);
    }
}
