#pragma once
#include "commons.h"

class Integration
{
public:
    Integration() = delete;
    Integration(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);
    void setNoise(const Vec4d &noise);
    void push_back(double _dt, const Vec3d &_acc, const Vec3d &_gyr);
    void propagate(double _dt, const Vec3d &_acc_1, const Vec3d &_gyr_1);
    void midPointIntegration();
    void repropagate(const Vec3d &_linearized_ba, const Vec3d &_linearized_bg);

    double dt;
    Vec3d acc_0, gyr_0, acc_1, gyr_1, linearized_ba, linearized_bg;
    const Vec3d linearized_acc, linearized_gyr;
    Mat15d jacobian, covariance;
    Mat18d noise;
    double sum_dt;
    Vec3d delta_p;
    Mat3d delta_q;
    Vec3d delta_v;
    std::vector<double> dt_buf;
    std::vector<Vec3d> acc_buf;
    std::vector<Vec3d> gyr_buf;
};
