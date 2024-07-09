#pragma once
#include <map>
#include <mutex>
#include <memory>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>

const int WINDOW_SIZE = 10;
const int THREADS_NUM = 2;
const int POSE_SIZE = 7;
const int SPEEDBIAS_SIZE = 9;
const int MAX_FEATURE_SIZE = 1000;

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

bool solveRelativeRT(const std::vector<std::pair<Vec3d, Vec3d>> &corres, Mat3d &rotation, Vec3d &translation, const double &ransac_threshold);

Vec3d rot2ypr(const Eigen::Matrix3d &R);

Mat3d ypr2rot(const Vec3d &ypr);

Mat3d rotFromG(const Eigen::Vector3d &g);
