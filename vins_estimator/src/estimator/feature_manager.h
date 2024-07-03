#pragma once
#include "commons.h"
#include <list>

struct FeatureInFrame
{
    FeatureInFrame(const Vec7d &_feat, double td) : cur_td(td)
    {
        point.x() = _feat[0];
        point.y() = _feat[1];
        point.z() = _feat[2];
        uv.x() = _feat[3];
        uv.y() = _feat[4];
        velocity.x() = _feat[5];
        velocity.y() = _feat[6];
    }
    double cur_td;
    Vec3d point;
    Vec2d uv;
    Vec2d velocity;
};

struct FeaturePoint
{
    FeaturePoint(int _feature_id, int _start_frame) : feature_id(_feature_id), start_frame(_start_frame), is_depth_valid(false), solve_flag(0), estimated_depth(-1.0)
    {
        observations.clear();
    }
    int endFrame() { return start_frame + static_cast<int>(observations.size()) - 1; }
    const int feature_id;
    int start_frame;
    bool is_depth_valid;
    int solve_flag; // 0: not optimized 1: optimized 2: failed
    double estimated_depth;
    std::vector<FeatureInFrame> observations;
};

class FeatureManager
{
    FeatureManager();

    double compensatedParallax2(const FeaturePoint &it_per_id, int frame_count);

    bool addFeatureCheckParallax(int frame_count, const TrackedFeatures &feats, double td);

    void getCorresponding(int frame_count_l, int frame_count_r, std::vector<std::pair<Vec3d, Vec3d>> &corres);

    void triangulate(Mat3d rs[], Vec3d ps[], const Mat3d &ric, const Vec3d &tic);

    void removeBack();

    void removeBackShiftDepth(const Mat3d &marg_r, const Vec3d &marg_p, const Mat3d &new_r, const Vec3d &new_p);

    void removeFront(int frame_count);

    void removeFailures();

    std::list<FeaturePoint> features;

    int &minTrackCount() { return m_min_track_count; }
    double &parallaxThreshold() { return m_parallax_threshold; }

    double &minDepthThreshold() { return m_min_depth_threshold; }

private:
    int m_min_track_count = 20;
    double m_min_depth_threshold = 0.1;
    double m_parallax_threshold = 10.0 / 460.0;
};