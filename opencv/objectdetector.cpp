#include "ObjectDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <QDebug>

ObjectDetector::ObjectDetector()
    : matchThreshold(0.6f)
    , ransacReprojThreshold(2.0f)
    , nFeatures(1000)
{
    orb = cv::ORB::create(nFeatures);
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
}

ObjectDetector::~ObjectDetector() {}

void ObjectDetector::setMatchThreshold(float thresh)
{
    matchThreshold = thresh;
}

void ObjectDetector::setRansacThreshold(float thresh)
{
    ransacReprojThreshold = thresh;
}

void ObjectDetector::setOrbFeatures(int nFeatures)
{
    this->nFeatures = nFeatures;
    orb = cv::ORB::create(nFeatures);
}

// ── 模板管理 ──────────────────────────────────────────────

bool ObjectDetector::addTemplate(const cv::Mat& templateImg, const std::string &name)
{
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

// bool ObjectDetector::addTemplates(const std::vector<cv::Mat>& images)
// {
//     bool anyOk = false;
//     for (size_t i = 0; i < images.size(); ++i) {
//         std::string name = "template_" + std::to_string(i);
//         if (addTemplate(images[i], name))
//             anyOk = true;
//     }
//     return anyOk;
// }

void ObjectDetector::computeTemplateFeatures(TemplateData &tpl)
{
    cv::Mat gray;
    if (tpl.image.channels() == 3)
        cv::cvtColor(tpl.image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = tpl.image.clone();

    orb->detectAndCompute(gray, cv::noArray(), tpl.keypoints, tpl.descriptors);
    if (tpl.descriptors.empty())
        std::cerr << "[ObjectDetector] No features found in template." << std::endl;
}

// ── 检测（多模板） ────────────────────────────────────────

bool ObjectDetector::detect(const cv::Mat& frame, cv::Rect& bbox)
{
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

bool ObjectDetector::findObject(const cv::Mat& frame, cv::Rect& bbox,
                                std::vector<cv::Point2f>& /*matchedPoints*/)
{
    // ── 1. 帧转灰度 + 提取特征（只做一次） ──
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

    // ── 2. 遍历所有模板，仅用 Lowe's ratio test 筛选最佳 ──
    int bestIndex = -1;
    int bestGoodMatches = 0;
    // 暂存最佳模板的匹配点对（只存最优的，避免每个模板都算）
    std::vector<cv::Point2f> bestSrcPts, bestDstPts;

    for (size_t t = 0; t < m_templates.size(); ++t) {
        const TemplateData &tpl = m_templates[t];
        if (!tpl.templateReady || tpl.descriptors.empty())
            continue;

        // 2a. knn 匹配（k=2）
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(tpl.descriptors, frameDescriptors, knnMatches, 2);

        // 2b. Lowe's ratio test 筛选
        std::vector<cv::DMatch> goodMatches;
        goodMatches.reserve(knnMatches.size());
        for (size_t i = 0; i < knnMatches.size(); ++i) {
            if (knnMatches[i].size() >= 2) {
                float dist1 = knnMatches[i][0].distance;
                float dist2 = knnMatches[i][1].distance;
                if (dist1 < matchThreshold * dist2)
                    goodMatches.push_back(knnMatches[i][0]);
            }
        }

        // goodMatches 数量不够，跳过
        if (goodMatches.size() < 4)
            continue;

        // 比当前最优更好？更新记录 + 暂存匹配点对
        if ((int)goodMatches.size() > bestGoodMatches) {
            bestGoodMatches = (int)goodMatches.size();
            bestIndex = (int)t;

            // 暂存匹配点对（仅此模板，避免后面重复计算）
            bestSrcPts.clear(); bestSrcPts.reserve(goodMatches.size());
            bestDstPts.clear(); bestDstPts.reserve(goodMatches.size());
            for (const auto& m : goodMatches) {
                bestSrcPts.push_back(tpl.keypoints[m.queryIdx].pt);
                bestDstPts.push_back(frameKeypoints[m.trainIdx].pt);
            }
        }
    }

    // ── 3. 没有模板通过 Lowe's test ──
    if (bestIndex < 0)
        return false;

    // ── 4. 只对最佳模板做一次 RANSAC 单应性计算 ──
    cv::Mat H = cv::findHomography(bestSrcPts, bestDstPts, cv::RANSAC, ransacReprojThreshold,cv::noArray(), 2000, 0.995);
    if (H.empty())
        return false;

    // ── 5. 用最佳模板 + 单应性矩阵计算边界框 ──
    const TemplateData &bestTpl = m_templates[bestIndex];
    std::vector<cv::Point2f> tplCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(bestTpl.image.cols - 1, 0),
        cv::Point2f(bestTpl.image.cols - 1, bestTpl.image.rows - 1),
        cv::Point2f(0, bestTpl.image.rows - 1)
    };
    std::vector<cv::Point2f> sceneCorners(4);
    cv::perspectiveTransform(tplCorners, sceneCorners, H);

    bbox = cv::boundingRect(sceneCorners);
    if (bbox.x < 0) bbox.x = 0;
    if (bbox.y < 0) bbox.y = 0;
    if (bbox.x + bbox.width > frame.cols)
        bbox.width = frame.cols - bbox.x;
    if (bbox.y + bbox.height > frame.rows)
        bbox.height = frame.rows - bbox.y;

    return bbox.width > 0 && bbox.height > 0;
}
