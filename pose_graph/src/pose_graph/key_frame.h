#pragma once
#include "commons.h"
#include "../ThirdParty/DBoW/DBoW2.h"
#include "../ThirdParty/DVision/DVision.h"
#include "../ThirdParty/DBoW/TemplatedDatabase.h"
#include "../ThirdParty/DBoW/TemplatedVocabulary.h"
#include "camera_model/pinhole_camera.h"
#include <opencv2/core/eigen.hpp>


using namespace DBoW2;
using namespace DVision;

template <typename Derived>
static void reduceVector(std::vector<Derived> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

class BriefExtractor
{
public:
    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
    BriefExtractor(const std::string &pattern_file);
    DVision::BRIEF m_brief;
};

class KeyFrame
{
public:
    KeyFrame(double _timestamp, Mat3d &rotation, Vec3d &translation, std::vector<int> &_points_id, std::vector<cv::Point3f> &_points_3d, std::vector<cv::Point2f> &_points_2d_uv, std::vector<cv::Point2f> &_points_2d_norm, cv::Mat &_image);

    void computeWindowBRIEFPoint();

    void computeBRIEFPoint();

    bool findConnection(std::shared_ptr<KeyFrame> &old_kf);

    void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);

    bool searchInAera(const BRIEF::bitset window_descriptor,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      const std::vector<cv::KeyPoint> &keypoints_old_norm,
                      cv::Point2f &best_match,
                      cv::Point2f &best_match_norm);

    int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

    static void initializeExtractor(std::string &pattern_file);

    static void initializeCamera(const PinholeParams &camera_config);

    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const std::vector<cv::Point3f> &matched_3d,
                   std::vector<uchar> &status,
                   Eigen::Vector3d &pnp_t_old, Eigen::Matrix3d &pnp_r_old);

    double time_stamp;
    bool has_loop;
    int index;
    int loop_index;
    int local_index;
    Mat3d loop_rotation;
    Vec3d loop_translation;
    double loop_yaw_deg;
    Mat3d local_rotation;
    Vec3d local_translation;
    Mat3d global_rotation;
    Vec3d global_translation;

    std::vector<int> points_id;
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d_uv;
    std::vector<cv::Point2f> points_2d_norm;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints_norm;

    std::vector<cv::KeyPoint> window_keypoints;
    std::vector<BRIEF::bitset> brief_descriptors;
    std::vector<BRIEF::bitset> window_brief_descriptors;
    cv::Mat image;
    static std::shared_ptr<BriefExtractor> extractor_ptr;
    static std::shared_ptr<PinholeCamera> camera_ptr;
    static Mat3d r_ic;
    static Vec3d t_ic;
    static int min_loop_num;
    static bool save_image;
};