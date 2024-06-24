#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "camera_model/pinhole_camera.h"

bool inBorder(const cv::Point2f &pt, int width, int height, int border);

void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);

void reduceVector(std::vector<int> &v, std::vector<uchar> status);

struct TrackerConfig
{
    bool equalize = true;
    double focal_length = 460;
    double f_threshold = 1.0;
    int min_dist = 30;
    int max_count = 150;
    int window_size = 20;
};
class FeatureTracker
{
public:
    FeatureTracker(const TrackerConfig &config, std::shared_ptr<PinholeCamera> camera);

    

    void rejectWithF();

    void setMask();

    void addPoints();

    void undistortedPoints();

    void updateID();

    void track(const cv::Mat &img, double timestamp, bool publish);

    TrackerConfig& config() { return m_config; }
    std::vector<cv::Point2f> &curPtsUV() { return m_cur_pts; }
    std::vector<cv::Point2f> &curPtsXY() { return m_cur_un_pts; }
    std::vector<cv::Point2f> &ptsVelocity() { return m_pts_velocity; }
    std::vector<int> &ids() { return m_ids; }
    std::vector<int> &trackCnt() { return m_track_cnt; }

private:
    TrackerConfig m_config;
    std::shared_ptr<PinholeCamera> m_camera;
    cv::Mat m_mask, m_prev_img, m_cur_img;
    double m_prev_time, m_cur_time;
    std::vector<int> m_ids, m_track_cnt;
    std::vector<cv::Point2f> m_prev_pts, m_cur_pts, m_cur_un_pts, m_cache_pts, m_pts_velocity;
    std::map<int, cv::Point2f> m_prev_un_pts_map, m_cur_un_pts_map;
    static int n_id;
};