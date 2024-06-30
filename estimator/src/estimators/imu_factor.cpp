#include "imu_factor.h"

IMUFactor::IMUFactor(std::shared_ptr<Integration> _integration, Vec3d _g_vec) : integration(_integration), g_vec(_g_vec)
{
}

bool IMUFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Vec3d bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Vec3d bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

    Vec3d pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vec3d vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Vec3d baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Vec3d bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = integration->evaluate(pi, qi, vi, bai, bgi, pj, qj, vj, baj, bgj, g_vec);
    Mat15d sqrt_info = Eigen::LLT<Mat15d>(integration->covariance.inverse()).matrixL().transpose();
    residual = sqrt_info * residual;

    if (jacobians)
    {
        double sum_dt = integration->sum_dt;
        Mat3d dp_dba = integration->jacobian.template block<3, 3>(O_P, O_BA);
        Mat3d dp_dbg = integration->jacobian.template block<3, 3>(O_P, O_BG);
        Mat3d dq_dbg = integration->jacobian.template block<3, 3>(O_R, O_BG);
        Mat3d dv_dba = integration->jacobian.template block<3, 3>(O_V, O_BA);
        Mat3d dv_dbg = integration->jacobian.template block<3, 3>(O_V, O_BG);
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();
            jacobian_pose_i.block<3, 3>(O_P, O_P) = -qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(O_P, O_R) = Sophus::SO3d::hat(qi.inverse() * (-0.5 * g_vec * sum_dt * sum_dt + pj - pi - vi * sum_dt));

            Quatd corrected_delta_q = Quatd(integration->delta_q) * deltaQ(dq_dbg * (bgi - integration->linearized_bg));
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qleft(qj.inverse() * qi) * Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

            jacobian_pose_i.block<3, 3>(O_V, O_R) = Sophus::SO3d::hat(qi.inverse() * (-g_vec * sum_dt + vj - vi));

            jacobian_pose_i = sqrt_info * jacobian_pose_i;
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();

            jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -qi.inverse().toRotationMatrix() * sum_dt;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Qleft(qj.inverse() * qi * Quatd(integration->delta_q)).bottomRightCorner<3, 3>() * dq_dbg;

            jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -qi.inverse().toRotationMatrix();
            jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

            jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Mat3d::Identity();
            jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Mat3d::Identity();
            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        }

        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();
            jacobian_pose_j.block<3, 3>(O_P, O_P) = qi.inverse().toRotationMatrix();

            Quatd corrected_delta_q = Quatd(integration->delta_q) * deltaQ(dq_dbg * (bgi - integration->linearized_bg));
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Qleft(corrected_delta_q.inverse() * qi.inverse() * qj).bottomRightCorner<3, 3>();
            jacobian_pose_j = sqrt_info * jacobian_pose_j;
        }

        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();

            jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = qi.inverse().toRotationMatrix();

            jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Mat3d::Identity();

            jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Mat3d::Identity();

            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        }
    }
    return true;
}