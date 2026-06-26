#include "ObjectDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <QDebug>
#include <QtConcurrent/QtConcurrentRun>   // 线程池支持
#include <QFuture>                        // 异步结果
#include <QFutureWatcher>                 // 可选，用于进度监控

// ========== 辅助结构：单个模板的匹配结果 ==========
struct TemplateMatchResult {
    int templateIndex = -1;
    int goodCount = 0;
    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
};

// ========== 构造 / 析构 ==========
ObjectDetector::ObjectDetector()
    : matchThreshold(0.6f)
    , ransacReprojThreshold(2.0f)
    , nFeatures(1000)
{
    orb = cv::ORB::create(nFeatures);
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

ObjectDetector::~ObjectDetector() {}

// ========== 参数设置 ==========
void ObjectDetector::setMatchThreshold(float thresh) { matchThreshold = thresh; }
void ObjectDetector::setRansacThreshold(float thresh) { ransacReprojThreshold = thresh; }
void ObjectDetector::setOrbFeatures(int nFeatures) {
    this->nFeatures = nFeatures;
    orb = cv::ORB::create(nFeatures);
}

// ========== 模板管理 ==========
bool ObjectDetector::addTemplate(const cv::Mat& templateImg, const std::string &name) {
    if (templateImg.empty()) {
        std::cerr << "[ObjectDetector] Template image is empty." << std::endl;
        return false;
    }
    TemplateData data;
    data.image = templateImg.clone();
    data.name = name;
    computeTemplateFeatures(data);
    data.templateReady = !data.descriptors.empty();
    if (data.templateReady) {
        m_templates.push_back(data);
        qDebug() << "[ObjectDetector] Template" << m_templates.size()
                 << "added:" << QString::fromStdString(name)
                 << "| keypoints:" << data.keypoints.size();
    }
    return data.templateReady;
}

void ObjectDetector::computeTemplateFeatures(TemplateData &tpl) {
    cv::Mat gray;
    if (tpl.image.channels() == 3)
        cv::cvtColor(tpl.image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = tpl.image.clone();
    orb->detectAndCompute(gray, cv::noArray(), tpl.keypoints, tpl.descriptors);
    if (tpl.descriptors.empty())
        std::cerr << "[ObjectDetector] No features found in template." << std::endl;
}

// ========== 检测入口 ==========
bool ObjectDetector::detect(const cv::Mat& frame, cv::Rect& bbox) {
    if (m_templates.empty()) {
        std::cerr << "[ObjectDetector] No templates loaded." << std::endl;
        return false;
    }
    if (frame.empty()) {
        std::cerr << "[ObjectDetector] Frame is empty." << std::endl;
        return false;
    }
    std::vector<cv::Point2f> dummy;
    return findObject(frame, bbox, dummy);
}

// ========== 核心检测（线程池并行版本） ==========
bool ObjectDetector::findObject(const cv::Mat& frame, cv::Rect& bbox,
                                std::vector<cv::Point2f>& /*matchedPoints*/)
{
    // ----- 1. 帧特征提取（单线程，只做一次） -----
    cv::Mat grayFrame;
    if (frame.channels() == 3)
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    else
        grayFrame = frame.clone();

    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(grayFrame, cv::noArray(), frameKeypoints, frameDescriptors);

    if (frameDescriptors.empty())
        return false;

    // ----- 2. 并行处理所有模板的匹配（线程池） -----
    // 每个模板的匹配任务独立，互不干扰，可并行
    QList<QFuture<TemplateMatchResult>> futures;
    futures.reserve(m_templates.size());

    for (int t = 0; t < (int)m_templates.size(); ++t) {
        const TemplateData &tpl = m_templates[t];
        if (!tpl.templateReady || tpl.descriptors.empty())
            continue;

        // 提交任务给 Qt 全局线程池
        QFuture<TemplateMatchResult> future = QtConcurrent::run(
            // lambda 捕获必要的参数（值传递或const引用）
            [this, &tpl, &frameDescriptors, &frameKeypoints, t]() -> TemplateMatchResult {
                TemplateMatchResult result;
                result.templateIndex = t;

                // 2a. knn 匹配 (k=2)
                std::vector<std::vector<cv::DMatch>> knnMatches;
                matcher->knnMatch(tpl.descriptors, frameDescriptors, knnMatches, 2);

                // 2b. Lowe's ratio test
                std::vector<cv::DMatch> goodMatches;
                goodMatches.reserve(knnMatches.size());
                for (const auto& km : knnMatches) {
                    if (km.size() >= 2) {
                        float dist1 = km[0].distance;
                        float dist2 = km[1].distance;
                        if (dist1 < matchThreshold * dist2)
                            goodMatches.push_back(km[0]);
                    }
                }

                // 2c. 如果足够，收集点对
                if (goodMatches.size() >= 4) {
                    result.goodCount = (int)goodMatches.size();
                    result.srcPts.reserve(goodMatches.size());
                    result.dstPts.reserve(goodMatches.size());
                    for (const auto& m : goodMatches) {
                        result.srcPts.push_back(tpl.keypoints[m.queryIdx].pt);
                        result.dstPts.push_back(frameKeypoints[m.trainIdx].pt);
                    }
                }
                return result;
            }
        );
        futures.append(future);
    }

    // ----- 3. 收集所有结果，选出最佳模板 -----
    TemplateMatchResult best;
    for (auto& f : futures) {
        TemplateMatchResult res = f.result();   // 阻塞等待该任务完成
        if (res.goodCount > best.goodCount) {
            best = std::move(res);
        }
    }

    // 没有足够匹配的模板
    if (best.templateIndex < 0 || best.goodCount < 4)
        return false;

    // ----- 4. 仅对最佳模板执行 RANSAC 单应性（主线程串行）-----
    cv::Mat H = cv::findHomography(best.srcPts, best.dstPts,
                                   cv::RANSAC, ransacReprojThreshold,
                                   cv::noArray(), 2000, 0.995);
    if (H.empty())
        return false;

    // ----- 5. 用最佳模板计算边界框 -----
    const TemplateData &bestTpl = m_templates[best.templateIndex];
    std::vector<cv::Point2f> tplCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(bestTpl.image.cols - 1, 0),
        cv::Point2f(bestTpl.image.cols - 1, bestTpl.image.rows - 1),
        cv::Point2f(0, bestTpl.image.rows - 1)
    };
    std::vector<cv::Point2f> sceneCorners(4);
    cv::perspectiveTransform(tplCorners, sceneCorners, H);

    bbox = cv::boundingRect(sceneCorners);
    // 裁剪到帧范围
    if (bbox.x < 0) bbox.x = 0;
    if (bbox.y < 0) bbox.y = 0;
    if (bbox.x + bbox.width > frame.cols)
        bbox.width = frame.cols - bbox.x;
    if (bbox.y + bbox.height > frame.rows)
        bbox.height = frame.rows - bbox.y;

    return bbox.width > 0 && bbox.height > 0;
}