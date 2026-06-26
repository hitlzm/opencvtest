#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QObject>
/**
 * @brief 基于 ORB 特征匹配的目标检测器
 * 
 * 使用模板图像在视频帧中搜索相似目标，返回其矩形边界框。
 * 内部采用 ORB 特征提取 + BFMatcher (汉明距离) + Lowe's ratio test
 * + RANSAC 单应性矩阵求取目标位置。
 */
// void processVideo();
//多模板改造，使用多个模板进行特征点匹配
// struct TemplateData {
//     cv::Mat image;
//     std::vector<cv::KeyPoint> keypoints;
//     cv::Mat descriptors;
//     std::string name; // 可选：用于调试输出（如 "正面", "侧面"）
// };

class ObjectDetector : public QObject
{
    Q_OBJECT
public:
    ObjectDetector();
    ~ObjectDetector();

    /**
     * @brief 设置模板图像（必须调用此方法后方可进行检测）
     * @param templateImg 模板图像（彩色或灰度均可，内部会自动转灰度）
     * @return true 成功提取特征，false 图像为空或无特征点
     */
    bool setTemplate(const cv::Mat& templateImg);

    /**
     * @brief 在单帧图像中检测目标
     * @param frame 输入帧（彩色或灰度）
     * @param bbox 输出目标矩形框（若检测失败则为空矩形）
     * @return true 检测到目标，false 未检测到
     */
    bool detect(const cv::Mat& frame, cv::Rect& bbox);

    // ---------- 参数调节接口 ----------
    void setMatchThreshold(float thresh);           // 默认 0.75 (Lowe's ratio)
    void setRansacThreshold(float thresh);          // 默认 3.0 (重投影误差)
    void setOrbFeatures(int nFeatures);             // 默认 1000

    // 调试用：获取模板特征点与描述子
    void getTemplateFeatures(std::vector<cv::KeyPoint>& keypoints,
                             cv::Mat& descriptors) const;
 
private:
    // std::vector<TemplateData> m_templates;
    cv::Mat templateImg;                // 模板图像（彩色）
    std::vector<cv::KeyPoint> tplKeypoints;  //模板特征点
    cv::Mat tplDescriptors;              //特征描述子
    cv::Ptr<cv::ORB> orb;               //特征提取类
    cv::Ptr<cv::BFMatcher> matcher;     //匹配类

    float matchThreshold;               // Lowe's ratio 阈值
    float ransacReprojThreshold;        // RANSAC 重投影误差阈值
    int nFeatures;                      // ORB 特征点数量
    bool templateReady;

    void computeTemplateFeatures();
    bool findObject(const cv::Mat& frame, cv::Rect& bbox,
                    std::vector<cv::Point2f>& matchedPoints);
};

#endif // OBJECTDETECTOR_H