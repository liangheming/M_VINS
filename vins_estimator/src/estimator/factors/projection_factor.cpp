#include "projection_factor.h"

ProjectionFactor::ProjectionFactor(const Vec3d &_pts_i, const Vec3d &_pts_j, const Mat2d &_sqrt_info)
    : pts_i(_pts_i), pts_j(_pts_j), sqrt_info(_sqrt_info)
{
}
bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // 格式转换
    Vec3d pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    // 计算残差
    Vec3d pts_camera_i = pts_i / inv_dep_i;
    Vec3d pts_imu_i = qic * pts_camera_i + tic;

    Vec3d pts_w = qi * pts_imu_i + pi;
    Vec3d pts_imu_j = qj.inverse() * (pts_w - pj);
    Vec3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);

    Eigen::Map<Vec2d> residual(residuals);

    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();

    residual = sqrt_info * residual;

    // 计算雅可比
    if (jacobians)
    {
        Mat3d ri = qi.toRotationMatrix();
        Mat3d rj = qj.toRotationMatrix();
        Mat3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            jacobian_pose_i.block<2, 3>(0, 0) = reduce * ric.transpose() * rj.transpose();
            jacobian_pose_i.block<2, 3>(0, 3) = -1.0 * reduce * ric.transpose() * rj.transpose() * ri * Sophus::SO3d::hat(pts_imu_i);
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
            jacobian_pose_j.setZero();
            jacobian_pose_j.block<2, 3>(0, 0) = -1.0 * reduce * ric.transpose() * rj.transpose();
            jacobian_pose_j.block<2, 3>(0, 3) = reduce * ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
        }
        Mat3d temp_r = ric.transpose() * rj.transpose() * ri * ric;
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            jacobian_ex_pose.setZero();
            jacobian_ex_pose.block<2, 3>(0, 0) = reduce * ric.transpose() * (rj.transpose() * ri - Mat3d::Identity());
            jacobian_ex_pose.block<2, 3>(0, 3) = reduce * (Sophus::SO3d::hat(temp_r * pts_camera_i) - temp_r * Sophus::SO3d::hat(pts_camera_i) + Sophus::SO3d::hat(ric.transpose() * (rj.transpose() * (ri * tic + pi - pj) - tic)));
        }
        if (jacobians[3])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[3]);
            jacobian_feature = -1.0 * reduce * temp_r * pts_i / (inv_dep_i * inv_dep_i);
        }
    }
    return true;
}