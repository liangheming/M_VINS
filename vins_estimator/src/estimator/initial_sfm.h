#pragma once
#include "commons.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>

struct SFMFeature
{
    bool state;
    int id;
    std::vector<std::pair<int, Vec2d>> observation;
    double position[3];
    double depth;
};
struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v)
    {
    }

    template <typename T>
    bool operator()(const T *const camera_r, const T *const camera_t, const T *point, T *residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_r, point, p);
        p[0] += camera_t[0];
        p[1] += camera_t[1];
        p[2] += camera_t[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionError3D, 2, 4, 3, 3>(
            new ReprojectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};
class GlobalSFM
{
public:
    GlobalSFM() = default;
    bool construct(int frame_num, Mat3d *rs, Vec3d *ts, int l,
                   const Mat3d relative_r, const Vec3d relative_t,
                   std::vector<SFMFeature> &sfm_f, std::map<int, Vec3d> &sfm_tracked_points);

private:
    int feature_num;
    bool solveFrameByPnP(Mat3d &r_initial, Vec3d &p_initial, int i, std::vector<SFMFeature> &sfm_f);
    void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &pose0,
                              int frame1, Eigen::Matrix<double, 3, 4> &pose1,
                              std::vector<SFMFeature> &sfm_f);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                          Vec2d &point0, Vec2d &point1, Vec3d &point_3d);
};