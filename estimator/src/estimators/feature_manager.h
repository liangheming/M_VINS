#pragma once
#include "commons.h"
#include <list>

class FeaturePerFrame
{
public:
    FeaturePerFrame(const Vec7d &_feat, double td) : cur_td(td)
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
class FeaturePerID
{
public:
    FeaturePerID(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame), used_num(0), estimated_depth(-1.0), solve_flag(0) { feature_per_frame.clear(); }
    int endFrame() { return start_frame + static_cast<int>(feature_per_frame.size()) - 1; }
    const int feature_id;
    int start_frame;
    int used_num;
    double estimated_depth;
    int solve_flag;
    std::vector<FeaturePerFrame> feature_per_frame;
};

class FeatureManager
{
public:
    FeatureManager(Mat3d _rs[]);

    void setRic(const Mat3d &_ric) { r_ic = _ric; }

    bool addFeatureCheckParallax(int frame_count, const Feats &feats, double td);

    double compensatedParallax2(const FeaturePerID &it_per_id, int frame_count);

    void getCorresponding(int frame_count_l, int frame_count_r, std::vector<std::pair<Vec3d, Vec3d>> &corres);

    int getFeatureCount();

    Eigen::VectorXd getDepthVector();

    void clearDepth(const Eigen::VectorXd &x);

    void setDepth(const Eigen::VectorXd &x);

    void triangulate(Mat3d rs[], Vec3d ps[], const Mat3d &ric, const Vec3d &tic);

    void removeBackShiftDepth(const Mat3d& marg_R, const Vec3d& marg_P, const Mat3d& new_R, const Vec3d& new_P);

    void removeBack();

    void removeFront(int frame_count);
    void removeFailures();
    double parallax_threshold = 10.0 / 460.0;
    std::list<FeaturePerID> features;

private:
    const Mat3d *rs;
    Mat3d r_ic;
};