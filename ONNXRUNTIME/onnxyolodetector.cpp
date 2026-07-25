// onnxyolodetector.cpp — ONNX Runtime YOLOv3-tiny 检测器实现
//
// ══════════════════════════════════════════════════════════════════════════════
//  YOLOv3-tiny ONNX 推理数据流
// ══════════════════════════════════════════════════════════════════════════════
//
//  步骤① 预处理（preprocess）
//    原图 BGR (H,W,3) → cv::dnn::blobFromImage
//    → resize 416×416 + BGR→RGB + 归一化 [0,1] + HWC→NCHW
//    → float blob [1, 3, 416, 416]
//
//  步骤② ONNX 推理（session->Run）
//    输入: float[1×3×416×416]
//    输出0: float[1, 255, 13, 13]  stride=32, 大目标尺度
//    输出1: float[1, 255, 26, 26]  stride=16, 小目标尺度
//    (255 = 3 anchors × (5 + 80 classes) for COCO)
//
//  步骤③ 后处理（postProcess）
//    a. 遍历两个输出尺度的每个网格单元
//    b. 对每层每个锚点解码 tx,ty,tw,th → 边界框坐标
//    c. 计算 objectness × class_score → 置信度
//    d. 过滤低置信度候选框
//    e. 映射回原始图像尺寸
//    f. NMS 去重 → 输出最终检测结果
//
// ══════════════════════════════════════════════════════════════════════════════
//  后续可优化部分，替换std::exp，自己实现NMS非极大值抑制
#include "onnxyolodetector.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <QDebug>

// ══════════════════════════════════════════════════════════════════════════════
// PIMPL 实现 — 封装所有 ONNX Runtime C++ 对象
// ══════════════════════════════════════════════════════════════════════════════
struct OnnxYoloDetectorImpl {
    // ONNX Runtime 环境（全局单例式的 logging 状态）
    std::unique_ptr<Ort::Env> env;
    // 会话选项
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    // 推理会话（绑定到 .onnx 模型文件）
    std::unique_ptr<Ort::Session> session;

    // 输入/输出节点名称（从模型动态获取，不硬编码）
    std::string inputName;
    std::vector<std::string> outputNames;

    // 输入形状：{1, 3, H, W}
    std::vector<int64_t> inputShape;
    // 输出形状列表
    std::vector<std::vector<int64_t>> outputShapes;

    void release() {
        session.reset();
        sessionOptions.reset();
        env.reset();
        inputName.clear();
        outputNames.clear();
        inputShape.clear();
        outputShapes.clear();
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// YOLOv3-tiny 锚点框（COCO 数据集，416×416 输入）
//
// 两个检测尺度各有 3 个锚点，按从小到大排列。
// 尺度 0: stride=32, 检测大目标（13×13 网格）
// 尺度 1: stride=16, 检测小目标（26×26 网格）
// ══════════════════════════════════════════════════════════════════════════════
// static constexpr float kAnchors[2][3][2] = {
//     // 尺度 0 (13×13) — 大锚点
//     {{116, 90}, {156, 198}, {373, 326}},
//     // 尺度 1 (26×26) — 小锚点
//     {{30, 61}, {62, 45}, {59, 119}}
// };

// static constexpr int kStrides[2] = {32, 16};
//后续换模型的话，可能还需要按着cfg文件改Anchor
static constexpr float kAnchors[2][3][2] =
{
    {
        {81.f, 82.f},
        {135.f,169.f},
        {344.f,319.f}
    },

    {
        {10.f,14.f},
        {23.f,27.f},
        {37.f,58.f}
    }
};
static constexpr int kStrides[2] = {32,16};

// ══════════════════════════════════════════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════════════════════════════════════════

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ══════════════════════════════════════════════════════════════════════════════
// 构造 / 析构
// ══════════════════════════════════════════════════════════════════════════════

OnnxYoloDetector::OnnxYoloDetector(QObject *parent)
    : QObject(parent)
    , m_impl(std::make_unique<OnnxYoloDetectorImpl>())
{
}

OnnxYoloDetector::~OnnxYoloDetector() {
    m_impl->release();
}

// ══════════════════════════════════════════════════════════════════════════════
// 参数设置
// ══════════════════════════════════════════════════════════════════════════════

void OnnxYoloDetector::setConfThreshold(float thresh) { m_confThreshold = thresh; }
void OnnxYoloDetector::setNmsThreshold(float thresh)  { m_nmsThreshold  = thresh; }
void OnnxYoloDetector::setNumThreads(int n)            { m_numThreads    = n; }

void OnnxYoloDetector::setInputSize(int width, int height) {
    m_inputWidth  = width;
    m_inputHeight = height;
}

void OnnxYoloDetector::setNumClasses(int n) {
    if (n > 0) m_numClasses = n;
}

// ══════════════════════════════════════════════════════════════════════════════
// 模型加载
// ══════════════════════════════════════════════════════════════════════════════

bool OnnxYoloDetector::loadModel(const std::string &onnxPath) {
    m_impl->release();
    m_loaded = false;

    try {
        // ── 创建 ONNX Runtime 环境 ──
        m_impl->env = std::make_unique<Ort::Env>(
            ORT_LOGGING_LEVEL_WARNING,
            "OnnxYoloDetector"
        );

        // ── 配置会话选项 ──
        m_impl->sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_impl->sessionOptions->SetIntraOpNumThreads(m_numThreads);
        // 设置图优化级别
        m_impl->sessionOptions->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED
        );

        // ── 加载模型并创建会话 ──
        // 注意：Windows 下 onnxPath 使用 UTF-8 路径
        //       若路径含中文，请使用 std::wstring 版本
        m_impl->session = std::make_unique<Ort::Session>(
            *m_impl->env,
            onnxPath.c_str(),
            *m_impl->sessionOptions
        );

        // ── 获取输入信息 ──
        Ort::AllocatorWithDefaultOptions allocator;
        {
            size_t numInputs = m_impl->session->GetInputCount();
            if (numInputs != 1) {
                emit errorOccurred(QString("Expected 1 input, got %1").arg(numInputs));
                m_impl->release();
                return false;
            }

            // 获取输入名称
            auto namePtr = m_impl->session->GetInputNameAllocated(0, allocator);
            m_impl->inputName = namePtr.get();

            // 获取输入形状
            Ort::TypeInfo typeInfo = m_impl->session->GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_impl->inputShape = tensorInfo.GetShape();

            // 动态维度（batch=1, H=W=-1 表示任意）替换为实际值
            if (m_impl->inputShape.size() == 4) {
                if (m_impl->inputShape[0] == -1) m_impl->inputShape[0] = 1;
                if (m_impl->inputShape[2] == -1) m_impl->inputShape[2] = m_inputHeight;
                if (m_impl->inputShape[3] == -1) m_impl->inputShape[3] = m_inputWidth;
            }

            qDebug() << "[OnnxYolo] Input:" << QString::fromStdString(m_impl->inputName)
                     << "shape: [" << m_impl->inputShape[0]
                     << m_impl->inputShape[1]
                     << m_impl->inputShape[2]
                     << m_impl->inputShape[3] << "]";
        }

        // ── 获取输出信息 ──
        {
            size_t numOutputs = m_impl->session->GetOutputCount();
            if (numOutputs < 1) {
                emit errorOccurred("Model has no outputs");
                m_impl->release();
                return false;
            }

            m_impl->outputNames.resize(numOutputs);
            m_impl->outputShapes.resize(numOutputs);

            for (size_t i = 0; i < numOutputs; ++i) {
                auto namePtr = m_impl->session->GetOutputNameAllocated(i, allocator);
                m_impl->outputNames[i] = namePtr.get();

                Ort::TypeInfo typeInfo = m_impl->session->GetOutputTypeInfo(i);
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                m_impl->outputShapes[i] = tensorInfo.GetShape();

                qDebug() << "[OnnxYolo] Output" << i
                         << ":" << QString::fromStdString(m_impl->outputNames[i])
                         << "shape: [" << m_impl->outputShapes[i][0]
                         << m_impl->outputShapes[i][1]
                         << m_impl->outputShapes[i][2]
                         << m_impl->outputShapes[i][3] << "]";
            }
        }

        m_loaded = true;
        qDebug() << "[OnnxYolo] Model loaded successfully:" << QString::fromStdString(onnxPath);
        return true;

    } catch (const Ort::Exception &e) {
        emit errorOccurred(QString("ONNX Runtime error: %1").arg(e.what()));
        m_impl->release();
        return false;
    } catch (const std::exception &e) {
        emit errorOccurred(QString("Failed to load model: %1").arg(e.what()));
        m_impl->release();
        return false;
    }
}

bool OnnxYoloDetector::loadClassNames(const std::string &namesPath) {
    std::ifstream file(namesPath);
    if (!file.is_open()) {
        emit errorOccurred(QString("Cannot open class names file: %1")
                          .arg(QString::fromStdString(namesPath)));
        return false;
    }

    m_classNames.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        m_classNames.push_back(line);
    }
    file.close();

    qDebug() << "[OnnxYolo] Loaded" << m_classNames.size() << "class names";
    return !m_classNames.empty();
}

void OnnxYoloDetector::setClassNames(const std::vector<std::string> &names) {
    m_classNames = names;
}

// ══════════════════════════════════════════════════════════════════════════════
// 预处理 — BGR 原图 → NCHW float blob
// ══════════════════════════════════════════════════════════════════════════════

// [保留] 旧版 blobFromImage 预处理（直接 resize，不保持宽高比）
// static cv::Mat preprocessFrame(const cv::Mat &frame, int w, int h) {
//     // cv::dnn::blobFromImage 一步完成：
//     //   resize → BGR→RGB(swapRB=true) → scale 1/255 → NCHW
//     return cv::dnn::blobFromImage(
//         frame,
//         1.0 / 255.0,                    // scale: 归一化到 [0,1]
//         cv::Size(w, h),                 // 缩放至网络输入尺寸
//         cv::Scalar(0, 0, 0),            // mean: YOLOv3-tiny 不做均值减法
//         true,                            // swapRB: BGR → RGB
//         false                            // crop: 不裁剪
//     );
// }

// ══════════════════════════════════════════════════════════════════════════════
// 预处理 — LetterBox + 手动转 NCHW blob（不依赖 opencv_dnn 模块）
// ══════════════════════════════════════════════════════════════════════════════
//
// LetterBox 流程：
//   ① 计算等比缩放比例 → resize 到 (newW, newH)
//   ② 居中放置到灰色画布 (w, h)，两侧/上下填充 114
//   ③ BGR→RGB → 归一化 [0,1] → HWC→NCHW 四维 blob
//
// 输出参数 outScale / outPad* 供后处理还原坐标：
//   orig_x = (lb_x - padX) / scale
//   orig_y = (lb_y - padY) / scale
// ══════════════════════════════════════════════════════════════════════════════

static cv::Mat preprocessFrame(const cv::Mat &frame, int w, int h,
                                float &outScale, int &outPadX, int &outPadY) {
    // ── ① 计算等比缩放，保持宽高比 ──
    const float scale = std::min(static_cast<float>(w) / frame.cols,
                                 static_cast<float>(h) / frame.rows);
    const int newW = static_cast<int>(std::round(frame.cols * scale));
    const int newH = static_cast<int>(std::round(frame.rows * scale));

    outScale = scale;
    outPadX  = (w - newW) / 2;
    outPadY  = (h - newH) / 2;

    // ── ② 等比缩放 ──
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    // ── ③ 放置到灰色画布（YOLO 约定填充值 114）──
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(outPadX, outPadY, newW, newH)));

    // ── ④ BGR → RGB, uint8 → float, 归一化 [0,1] ──
    // cv::Mat rgb;
    // cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    // cv::Mat floatMat;
    // rgb.convertTo(floatMat, CV_32F, 1.0 / 255.0);

    // ── ⑤ HWC → CHW → NCHW（手动构建 4D blob）──
    // std::vector<cv::Mat> planes(3);
    // cv::split(floatMat, planes);          // planes[0]=R, planes[1]=G, planes[2]=B

    // int sz[] = {1, 3, h, w};
    // cv::Mat blob(4, sz, CV_32F);
    // for (int c = 0; c < 3; ++c) {
    //     planes[c].copyTo(cv::Mat(h, w, CV_32F, blob.ptr<float>(0, c)));
    // }

    // ═══════════════════════════════════════════════════════════════════════════
    // [备选] ④+⑤ 融合版：一次遍历完成 BGR→RGB + 归一化 + HWC→NCHW
    // 省掉 cvtColor、convertTo、split 三步中间分配，速度与 blobFromImage 持平
    // 如需极致性能可替换上方 ④⑤ 步骤，当前版本可读性更好
    // ═══════════════════════════════════════════════════════════════════════════
    
    int sz[] = {1, 3, h, w};
    cv::Mat blob(4, sz, CV_32F);
    
    for (int y = 0; y < h; ++y) {
        const uint8_t *srcRow = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const uint8_t *px = srcRow + x * 3;
            blob.ptr<float>(0, 0)[y * w + x] = px[2] / 255.0f;  // R ← BGR[2]
            blob.ptr<float>(0, 1)[y * w + x] = px[1] / 255.0f;  // G ← BGR[1]
            blob.ptr<float>(0, 2)[y * w + x] = px[0] / 255.0f;  // B ← BGR[0]
        }
    }

    return blob;
}

// ══════════════════════════════════════════════════════════════════════════════
// 后处理 — 解析 YOLOv3-tiny 输出 → 检测结果
//
// 每个尺度输出形状: [1, 3*(5+K), GH, GW]  (NCHW 布局)
//   GH×GW 为网格尺寸（13×13 或 26×26）
//   每锚点占 (5+K) 个通道: [tx, ty, tw, th, objectness, class_0..class_K]
//
// 解码公式：
//   bx = (σ(tx) + cx) / GW          — 网格内偏移 + 网格索引
//   by = (σ(ty) + cy) / GH
//   bw = anchor_w × exp(tw) / IW    — 锚点缩放
//   bh = anchor_h × exp(th) / IH
//   confidence = σ(objectness) × max(σ(class_scores))
//
// 注意：所有坐标在解码阶段归一化到 [0,1]，最后才乘原图尺寸
// ══════════════════════════════════════════════════════════════════════════════

static void decodeYoloScale(
    const float *data,              // [1, 3*(5+K), GH, GW] NCHW
    int K,                          // 类别数量
    int GH, int GW,                 // 网格尺寸
    int IW, int IH,                 // 网络输入尺寸
    int FW, int FH,                 // 原始帧尺寸
    int stride,                     // 下采样步长
    const float anchors[3][2],      // 该尺度的 3 个锚点
    float confThreshold,
    float letterBoxScale,           // LetterBox 等比缩放比例
    int letterBoxPadX,              // LetterBox 水平填充量
    int letterBoxPadY,              // LetterBox 垂直填充量
    std::vector<int> &classIds,
    std::vector<float> &confidences,
    std::vector<cv::Rect> &boxes)
{
    const int step = 5 + K;  // 每个锚点的通道数
    const int na = 3;        // 锚点个数

    // 对每个网格单元
    for (int gy = 0; gy < GH; ++gy) {
        for (int gx = 0; gx < GW; ++gx) {
            // 对每个锚点
            for (int a = 0; a < na; ++a) {
                // NCHW 索引: channel_base = a * step
                // 位置 (c, gy, gx) 在 NCHW 布局中的索引 = c*GH*GW + gy*GW + gx
                const int base = gy * GW + gx;
                const int chOffset = a * step * GH * GW;

                // 读取原始 logits
                const float tx = data[chOffset + 0 * GH * GW + base];
                const float ty = data[chOffset + 1 * GH * GW + base];
                const float tw = data[chOffset + 2 * GH * GW + base];
                const float th = data[chOffset + 3 * GH * GW + base];
                const float objLogit = data[chOffset + 4 * GH * GW + base];

                // ── objectness ──
                const float obj = sigmoid(objLogit);
                if (obj < confThreshold * 0.1f) // 快速跳过极低置信度
                    continue;

                // ── 找最佳类别 ──
                float maxClassScore = 0.0f;
                int bestClass = -1;
                for (int c = 0; c < K; ++c) {
                    const float clsLogit = data[chOffset + (5 + c) * GH * GW + base];
                    const float clsScore = sigmoid(clsLogit);
                    if (clsScore > maxClassScore) {
                        maxClassScore = clsScore;
                        bestClass = c;
                    }
                }

                const float confidence = obj * maxClassScore;
                if (confidence < confThreshold)
                    continue;

                // ── 解码边界框（归一化坐标 0~1，相对于 LetterBox 画布）──
                const float bx = (sigmoid(tx) + static_cast<float>(gx)) / static_cast<float>(GW);
                const float by = (sigmoid(ty) + static_cast<float>(gy)) / static_cast<float>(GH);
                const float bw = anchors[a][0] * std::exp(tw) / static_cast<float>(IW);
                const float bh = anchors[a][1] * std::exp(th) / static_cast<float>(IH);

                // ── 从 LetterBox 画布坐标还原为原始图像坐标 ──
                // lb* = 归一化坐标 × 画布尺寸 → 画布像素
                // 减去 padding offset → 除以 scale → 原图像素
                const float lbX = bx * static_cast<float>(IW);
                const float lbY = by * static_cast<float>(IH);
                const float lbW = bw * static_cast<float>(IW);
                const float lbH = bh * static_cast<float>(IH);

                const int left   = static_cast<int>((lbX - lbW * 0.5f - static_cast<float>(letterBoxPadX)) / letterBoxScale);
                const int top    = static_cast<int>((lbY - lbH * 0.5f - static_cast<float>(letterBoxPadY)) / letterBoxScale);
                const int width  = static_cast<int>(lbW / letterBoxScale);
                const int height = static_cast<int>(lbH / letterBoxScale);

                // ── 裁剪到图像范围内 ──
                const int x1 = std::max(0, left);
                const int y1 = std::max(0, top);
                const int x2 = std::min(FW, left + width);
                const int y2 = std::min(FH, top + height);

                if (x2 <= x1 || y2 <= y1)
                    continue;

                classIds.push_back(bestClass);
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 检测入口 — 完整的 预处理 → 推理 → 后处理 流水线
// ══════════════════════════════════════════════════════════════════════════════

bool OnnxYoloDetector::detect(const cv::Mat &frame, std::vector<OnnxDetection> &detections) {
    detections.clear();

    if (!m_loaded || !m_impl->session) {
        std::cerr << "[OnnxYolo] Model not loaded." << std::endl;
        return false;
    }
    if (frame.empty()) {
        std::cerr << "[OnnxYolo] Input frame is empty." << std::endl;
        return false;
    }

    try {
        // ══ 步骤① 预处理（LetterBox + NCHW blob）══
        float letterBoxScale;
        int letterBoxPadX, letterBoxPadY;
        cv::Mat blob = preprocessFrame(frame, m_inputWidth, m_inputHeight,
                                        letterBoxScale, letterBoxPadX, letterBoxPadY);
        // 保存 LetterBox 参数，供后处理坐标还原使用
        m_letterBoxScale = letterBoxScale;
        m_letterBoxPadX  = letterBoxPadX;
        m_letterBoxPadY  = letterBoxPadY;

        // ══ 步骤② 构造输入张量 ══
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> inputShape = {
            1, 3, m_inputHeight, m_inputWidth
        };
        size_t inputElementCount = 1 * 3 * m_inputHeight * m_inputWidth;

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            reinterpret_cast<float *>(blob.data),
            inputElementCount,
            inputShape.data(),
            inputShape.size()
        );

        // ══ 步骤③ ONNX 推理 ══
        std::vector<const char *> inputNames  = {m_impl->inputName.c_str()};
        std::vector<const char *> outputNames;
        outputNames.reserve(m_impl->outputNames.size());
        for (auto &name : m_impl->outputNames)
            outputNames.push_back(name.c_str());

        auto outputTensors = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(), &inputTensor, 1,
            outputNames.data(), outputNames.size()
        );

        // ══ 步骤④ 提取输出数据 ══
        // outputTensors 在 postProcess 期间必须存活（持有数据所有权）
        std::vector<const float *> outputData;
        std::vector<int> outputGH, outputGW;
        outputData.reserve(outputTensors.size());

        for (auto &tensor : outputTensors) {
            outputData.push_back(tensor.GetTensorMutableData<float>());
            auto shapeInfo = tensor.GetTensorTypeAndShapeInfo();
            auto shape = shapeInfo.GetShape();
            // shape = [1, 3*(5+K), GH, GW]
            if (shape.size() >= 4) {
                outputGH.push_back(static_cast<int>(shape[2]));
                outputGW.push_back(static_cast<int>(shape[3]));
            }
        }

        // ══ 步骤⑤ 后处理 ══
        const int K = m_numClasses;
        const int frameW = frame.cols;
        const int frameH = frame.rows;

        std::vector<int>    allClassIds;
        std::vector<float>  allConfidences;
        std::vector<cv::Rect> allBoxes;

        // 遍历每个输出尺度
        for (size_t s = 0; s < outputData.size() && s < 2; ++s) {
            if (s >= outputGH.size() || s >= outputGW.size())
                break;

            decodeYoloScale(
                outputData[s],
                K,
                outputGH[s], outputGW[s],
                m_inputWidth, m_inputHeight,
                frameW, frameH,
                kStrides[s],
                kAnchors[s],
                m_confThreshold,
                m_letterBoxScale,
                m_letterBoxPadX,
                m_letterBoxPadY,
                allClassIds,
                allConfidences,
                allBoxes
            );
        }

        // ══ 步骤⑥ 非极大值抑制 (NMS) ══
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(allBoxes, allConfidences,
                          m_confThreshold, m_nmsThreshold,
                          nmsIndices);

        // ══ 步骤⑦ 组装最终结果 ══
        detections.reserve(nmsIndices.size());
        for (int idx : nmsIndices) {
            OnnxDetection det;
            det.classId    = allClassIds[idx];
            det.confidence = allConfidences[idx];
            det.bbox       = allBoxes[idx];

            // 查找类别名称
            if (det.classId >= 0 && det.classId < static_cast<int>(m_classNames.size()))
                det.className = m_classNames[det.classId];
            else
                det.className = "cls_" + std::to_string(det.classId);

            detections.push_back(det);
        }

        return !detections.empty();

    } catch (const Ort::Exception &e) {
        std::cerr << "[OnnxYolo] Inference error: " << e.what() << std::endl;
        return false;
    } catch (const cv::Exception &e) {
        std::cerr << "[OnnxYolo] OpenCV error: " << e.what() << std::endl;
        return false;
    }
}
