#pragma once
#include <map>
#include <memory>
#include <iostream>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>
const int WINDOW_SIZE = 20;

using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;
using Vec4d = Eigen::Vector4d;
using Mat4d = Eigen::Matrix4d;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Feats = std::map<int, Vec7d>;
using Mat15d = Eigen::Matrix<double, 15, 15>;
using Mat18d = Eigen::Matrix<double, 18, 18>;
using Mat15x18d = Eigen::Matrix<double, 15, 18>;

Mat3d Jr(const Vec3d& val);