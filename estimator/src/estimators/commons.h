#pragma once
#include <map>
#include <memory>
#include <iostream>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
#include <opencv2/opencv.hpp>

const int WINDOW_SIZE = 10;

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;
using Vec4d = Eigen::Vector4d;
using Mat4d = Eigen::Matrix4d;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Feats = std::map<int, Vec7d>;
using Mat15d = Eigen::Matrix<double, 15, 15>;
using Mat18d = Eigen::Matrix<double, 18, 18>;
using Mat15x18d = Eigen::Matrix<double, 15, 18>;
using Quatd = Eigen::Quaterniond;

Mat3d Jr(const Vec3d &val);

bool solveRelativeRT(const std::vector<std::pair<Vec3d, Vec3d>> &corres, Mat3d &rotation, Vec3d &translation);