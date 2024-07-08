#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;
using Quatd = Eigen::Quaterniond;

Vec3d rot2ypr(const Eigen::Matrix3d &R);

Mat3d ypr2rot(const Vec3d &ypr);

template <typename T>
static T normalizeAngle(const T &angle_degrees)
{
    T two_pi(2.0 * 180);
    if (angle_degrees > 0)
        return angle_degrees - two_pi * std::floor((angle_degrees + T(180)) / two_pi);
    else
        return angle_degrees + two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
};
