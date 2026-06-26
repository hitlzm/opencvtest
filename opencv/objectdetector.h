#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <QObject>
/**
 * @brief 基于 ORB 特征匹配的目标检测器（多模板支持）
 *
 * 可加载多个模板图像（正面、侧面等），在视频帧中分别
 * 匹配每个模板，最终选用匹配度（goodMatches 数量）最高
 * 的模板来计算目标矩形边界框。
 *
 * 内部采用 ORB 特征提取 + BFMatcher (汉明距离) +
 * Lowe's ratio test + RANSAC 单应性矩阵求取目标位置。
 */

struct TemplateData {
    cv::Mat image;                          // 模板图像
    std::vector<cv::KeyPoint> keypoints;    // 特征点
    cv::Mat descriptors;                    // 特征描述子
    bool templateReady = false;
    std::string name;                       // 可选：调试用（如 "front", "side"）
};

class ObjectDetector : public QObject
{
    Q_OBJECT
public:
    ObjectDetector();
    ~ObjectDetector();

    /**
     * @brief 添加一个模板图像（可多次调用，加载多个模板）
     * @param templateImg 模板图像（彩色或灰度均可）
     * @return true 成功提取特征，false 图像为空或无特征点
     */
    bool addTemplate(const cv::Mat& templateImg, const std::string &name = "");

    /**
     * @brief 便捷：一次性加载多个模板
     */
    // bool addTemplates(const std::vector<cv::Mat>& images);

    /**
     * @brief 在单帧图像中检测目标（遍历所有模板，选匹配度最高的）
     * @param frame 输入帧
     * @param bbox 输出目标矩形框
     * @return true 检测到目标
     */
    bool detect(const cv::Mat& frame, cv::Rect& bbox);

    // ---------- 参数调节接口 ----------
    void setMatchThreshold(float thresh);           // 默认 0.75 (Lowe's ratio)
    void setRansacThreshold(float thresh);          // 默认 3.0 (重投影误差)
    void setOrbFeatures(int nFeatures);             // 默认 1000

    /// 已加载的模板数量
    int templateCount() const { return (int)m_templates.size(); }

private:
    std::vector<TemplateData> m_templates;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::BFMatcher> matcher;

    float matchThreshold;               // Lowe's ratio 阈值
    float ransacReprojThreshold;        // RANSAC 重投影误差阈值
    int nFeatures;                      // ORB 特征点数量

    void computeTemplateFeatures(TemplateData &tpl);
    bool findObject(const cv::Mat& frame, cv::Rect& bbox,
                    std::vector<cv::Point2f>& matchedPoints);
};

#endif // OBJECTDETECTOR_H
