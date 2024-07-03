#pragma once
#include <map>
#include <memory>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>

const int WINDOW_SIZE = 10;
const int THREADS_NUM = 2;

using Vec2d = Eigen::Vector2d;
using Mat2d = Eigen::Matrix2d;
using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;
using Vec4d = Eigen::Vector4d;
using Mat4d = Eigen::Matrix4d;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using TrackedFeatures = std::map<int, Vec7d>;
using Mat15d = Eigen::Matrix<double, 15, 15>;
using Mat18d = Eigen::Matrix<double, 18, 18>;
using Mat15x18d = Eigen::Matrix<double, 15, 18>;
using Quatd = Eigen::Quaterniond;

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

Mat3d Jr(const Vec3d &val);

Mat3d Jr_inv(const Vec3d &val);