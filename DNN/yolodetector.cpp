#include "yolodetector.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <QDebug>

// ========== 构造 / 析构 ==========
YoloDetector::YoloDetector(QObject *parent)
    : QObject(parent)
{
}

YoloDetector::~YoloDetector() {}

// ========== 参数设置 ==========
void YoloDetector::setConfThreshold(float thresh) { m_confThreshold = thresh; }
void YoloDetector::setNmsThreshold(float thresh)  { m_nmsThreshold  = thresh; }

void YoloDetector::setInputSize(int width, int height)
{
    m_inputWidth  = width;
    m_inputHeight = height;
}

void YoloDetector::setBackend(int backend, int target)
{
    m_net.setPreferableBackend(backend);
    m_net.setPreferableTarget(target);
}

// ========== 模型加载 ==========
bool YoloDetector::loadModel(const std::string &cfgPath, const std::string &weightsPath)
{
    try {
        m_net = cv::dnn::readNetFromDarknet(cfgPath, weightsPath);

        // 默认使用 OpenCV 后端 + CPU
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        m_outputNames = getOutputNames();

        if (m_outputNames.empty()) {
            std::cerr << "[YoloDetector] No YOLO output layers found in model." << std::endl;
            return false;
        }

        m_loaded = true;
        qDebug() << "[YoloDetector] Model loaded, output layers:" << m_outputNames.size();
        return true;

    } catch (const cv::Exception &e) {
        std::cerr << "[YoloDetector] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

bool YoloDetector::loadClassNames(const std::string &namesPath)
{
    std::ifstream file(namesPath);
    if (!file.is_open()) {
        std::cerr << "[YoloDetector] Cannot open class names file: " << namesPath << std::endl;
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

    qDebug() << "[YoloDetector] Loaded" << m_classNames.size() << "class names";
    return !m_classNames.empty();
}

void YoloDetector::setClassNames(const std::vector<std::string> &names)
{
    m_classNames = names;
}

// ========== 获取 YOLO 输出层名称 ==========
std::vector<std::string> YoloDetector::getOutputNames()
{
    std::vector<std::string> names;
    std::vector<int> outLayers = m_net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = m_net.getLayerNames();

    // OpenCV 的 getLayerNames() 在部分版本返回 0-based 索引，
    // getUnconnectedOutLayers() 返回 1-based 索引，需要转换
    for (int idx : outLayers) {
        // 兼容不同 OpenCV 版本的索引差异
        int i = (idx > 0 && idx <= (int)layersNames.size()) ? idx - 1 : idx;
        if (i >= 0 && i < (int)layersNames.size())
            names.push_back(layersNames[i]);
    }
    return names;
}

// ========== 检测入口 ==========
bool YoloDetector::detect(const cv::Mat &frame, std::vector<YoloDetection> &detections)
{
    detections.clear();

    if (!m_loaded) {
        std::cerr << "[YoloDetector] Model not loaded." << std::endl;
        return false;
    }
    if (frame.empty()) {
        std::cerr << "[YoloDetector] Input frame is empty." << std::endl;
        return false;
    }

    // ── 1. 预处理：将输入图像转为 blob ──
    //    缩放至网络输入尺寸，归一化到 [0, 1]，交换 BGR → RGB
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob,
                           1.0 / 255.0,                  // scale
                           cv::Size(m_inputWidth, m_inputHeight),
                           cv::Scalar(0, 0, 0),          // mean (YOLOv4 不需减均值)
                           true,                          // swapRB (BGR → RGB)
                           false);                        // crop

    // ── 2. 前向推理 ──
    m_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_outputNames);

    // ── 3. 后处理 ──
    postProcess(frame, outputs, detections);

    return !detections.empty();
}

// ========== YOLO 输出后处理 ==========
void YoloDetector::postProcess(const cv::Mat &frame,
                               const std::vector<cv::Mat> &outputs,
                               std::vector<YoloDetection> &detections)
{
    int frameW = frame.cols;
    int frameH = frame.rows;

    std::vector<int>    classIds;
    std::vector<float>  confidences;
    std::vector<cv::Rect> boxes;

    // ── 遍历所有输出层（3 个尺度） ──
    for (const auto &output : outputs) {
        // YOLOv4 输出 shape: [N, 5 + numClasses]
        //   每行: [center_x, center_y, width, height, objectness, class_0, ...]
        const float *data = (const float *)output.data;

        for (int i = 0; i < output.rows; ++i, data += output.cols) {
            // 提取每个类别的置信度并找最大值
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

            // 置信度 = objectness * class_confidence
            float objectness = data[4];
            float finalConf = objectness * (float)confidence;

            if (finalConf < m_confThreshold)
                continue;

            // YOLO 输出坐标（相对于 0~1）
            float cx = data[0];
            float cy = data[1];
            float w  = data[2];
            float h  = data[3];

            // 映射回原始图像尺寸
            int left   = int((cx - w * 0.5f) * frameW);
            int top    = int((cy - h * 0.5f) * frameH);
            int width  = int(w * frameW);
            int height = int(h * frameH);

            classIds.push_back(classIdPoint.x);
            confidences.push_back(finalConf);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // ── 非极大值抑制 (NMS) ──
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    // ── 组装最终结果 ──
    detections.reserve(indices.size());
    for (int idx : indices) {
        YoloDetection det;
        det.classId    = classIds[idx];
        det.confidence = confidences[idx];
        det.bbox       = boxes[idx];

        // 裁剪到图像范围
        if (det.bbox.x < 0) det.bbox.x = 0;
        if (det.bbox.y < 0) det.bbox.y = 0;
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
