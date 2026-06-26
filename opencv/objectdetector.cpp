#include "ObjectDetector.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <QDebug>

ObjectDetector::ObjectDetector()
    : matchThreshold(0.75f)
    , ransacReprojThreshold(3.0f)
    , nFeatures(1000)
    , templateReady(false)
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

bool ObjectDetector::setTemplate(const cv::Mat& templateImg)
{
    if (templateImg.empty())
    {
        std::cerr << "[ObjectDetector] Template image is empty." << std::endl;
        return false;
    }
    this->templateImg = templateImg.clone();
    computeTemplateFeatures();
    templateReady = !tplDescriptors.empty();
    return templateReady;
}

void ObjectDetector::computeTemplateFeatures()
{
    // 转为灰度图
    cv::Mat gray;
    if (templateImg.channels() == 3)
        cv::cvtColor(templateImg, gray, cv::COLOR_BGR2GRAY);
    else
        gray = templateImg.clone();

    orb->detectAndCompute(gray, cv::noArray(), tplKeypoints, tplDescriptors);
    if (tplDescriptors.empty())
        std::cerr << "[ObjectDetector] No features found in template." << std::endl;
}

void ObjectDetector::getTemplateFeatures(std::vector<cv::KeyPoint>& keypoints,
                                         cv::Mat& descriptors) const
{
    keypoints = tplKeypoints;
    descriptors = tplDescriptors.clone();
}

bool ObjectDetector::detect(const cv::Mat& frame, cv::Rect& bbox)
{
    if (!templateReady)
    {
        std::cerr << "[ObjectDetector] Template not set or has no features." << std::endl;
        return false;
    }
    if (frame.empty())
    {
        std::cerr << "[ObjectDetector] Frame is empty." << std::endl;
        return false;
    }

    std::vector<cv::Point2f> dummy;
    return findObject(frame, bbox, dummy);
}

bool ObjectDetector::findObject(const cv::Mat& frame, cv::Rect& bbox,
                                std::vector<cv::Point2f>& /*matchedPoints*/)
{
    // 1. 帧转灰度
    cv::Mat grayFrame;
    if (frame.channels() == 3)
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    else
        grayFrame = frame.clone();

    // 2. 提取帧特征
    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    orb->detectAndCompute(grayFrame, cv::noArray(), frameKeypoints, frameDescriptors);

    if (frameDescriptors.empty() || tplDescriptors.empty())   //没有检测到特征点退出函数
        return false;

    // 3. knn 匹配（k=2）
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(tplDescriptors, frameDescriptors, knnMatches, 2);

    // 4. Lowe's ratio test 筛选
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); ++i)
    {
        if (knnMatches[i].size() >= 2)
        {
            float dist1 = knnMatches[i][0].distance;
            float dist2 = knnMatches[i][1].distance;
            if (dist1 < matchThreshold * dist2)
                goodMatches.push_back(knnMatches[i][0]);
        }
    }

    if (goodMatches.size() < 4) // 至少需要4对点计算单应性
        return false;

    // 5. 整理匹配点对
    std::vector<cv::Point2f> srcPts, dstPts;
    srcPts.reserve(goodMatches.size());
    dstPts.reserve(goodMatches.size());
    for (const auto& m : goodMatches)
    {
        srcPts.push_back(tplKeypoints[m.queryIdx].pt);
        dstPts.push_back(frameKeypoints[m.trainIdx].pt);
    }

    // 6. RANSAC 求单应性矩阵
    cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, ransacReprojThreshold);
    if (H.empty())
        return false;

    // 7. 将模板四个角点映射到当前帧
    std::vector<cv::Point2f> tplCorners = {
        cv::Point2f(0, 0),
        cv::Point2f(templateImg.cols - 1, 0),
        cv::Point2f(templateImg.cols - 1, templateImg.rows - 1),
        cv::Point2f(0, templateImg.rows - 1)
    };
    std::vector<cv::Point2f> sceneCorners(4);
    cv::perspectiveTransform(tplCorners, sceneCorners, H);

    // 8. 计算最小包围矩形并裁剪到图像范围
    bbox = cv::boundingRect(sceneCorners);
    if (bbox.x < 0) bbox.x = 0;
    if (bbox.y < 0) bbox.y = 0;
    if (bbox.x + bbox.width > frame.cols)
        bbox.width = frame.cols - bbox.x;
    if (bbox.y + bbox.height > frame.rows)
        bbox.height = frame.rows - bbox.y;

    return bbox.width > 0 && bbox.height > 0;
}

// void processVideo()
// {
//     ObjectDetector detector;
//     detector.setMatchThreshold(0.7);   // 可选调参
//     detector.setRansacThreshold(4.0);

//     // 1. 加载模板图片
//     cv::Mat templ = cv::imread("E:/tanke.png");
//     if (!detector.setTemplate(templ))
//     {
//         qDebug() << "Failed to set template";
//         return;
//     }

//     // 2. 打开视频
//     cv::VideoCapture cap("E:/tanke.mp4");
//     if (!cap.isOpened())
//     {
//         qDebug() << "Cannot open video";
//         return;
//     }

//     cv::Mat frame;
//     while (cap.read(frame))
//     {
//         cv::Rect bbox;
//         if (detector.detect(frame, bbox))
//         {
//             // 在帧上绘制矩形框（仅作演示）
//             cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
//             qDebug() << "Detected at" << bbox.x << bbox.y
//                      << bbox.width << bbox.height;
//         }
//         else
//         {
//             qDebug() << "Not detected";
//         }

//     }
// }