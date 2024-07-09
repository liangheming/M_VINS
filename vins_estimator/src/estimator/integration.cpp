#include "integration.h"

Integration::Integration(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                         const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    : acc_0(_acc_0), gyr_0(_gyr_0), linearized_ba(_linearized_ba), linearized_bg(_linearized_bg), linearized_acc(_acc_0), linearized_gyr(_gyr_0),
      jacobian(Mat15d::Identity()), covariance(Mat15d::Zero()),
      sum_dt(0.0), delta_p(Vec3d::Zero()), delta_q(Quatd::Identity()), delta_v(Vec3d::Zero())

{
    noise = Mat18d::Zero();
    noise.block<3, 3>(0, 0) = (0.08 * 0.08) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (0.004 * 0.004) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (0.08 * 0.08) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (0.004 * 0.004) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (0.00004 * 0.00004) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (2.0e-6 * 2.0e-6) * Eigen::Matrix3d::Identity();
    dt_buf.clear();
    acc_buf.clear();
    gyr_buf.clear();
}
void Integration::setNoise(const Vec4d &_noise)
{
    noise = Mat18d::Zero();
    noise.block<3, 3>(0, 0) = (_noise(0) * _noise(0)) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) = (_noise(1) * _noise(1)) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) = (_noise(0) * _noise(0)) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) = (_noise(1) * _noise(1)) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) = (_noise(2) * _noise(2)) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) = (_noise(3) * _noise(3)) * Eigen::Matrix3d::Identity();
}
void Integration::push_back(double _dt, const Vec3d &_acc, const Vec3d &_gyr)
{
    dt_buf.push_back(_dt);
    acc_buf.push_back(_acc);
    gyr_buf.push_back(_gyr);
    propagate(_dt, _acc, _gyr);
}
void Integration::propagate(double _dt, const Vec3d &_acc_1, const Vec3d &_gyr_1)
{
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    midPointIntegration();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
}
void Integration::midPointIntegration()
{
    Vec3d un_acc_0 = delta_q * (acc_0 - linearized_ba);
    Vec3d un_gyr = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
    Quatd new_delta_q = delta_q * deltaQ(un_gyr * dt);
    Vec3d un_acc_1 = new_delta_q * (acc_1 - linearized_ba);
    Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Vec3d new_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
    Vec3d new_delta_v = delta_v + un_acc * dt;

    Mat15d F = Mat15d::Identity();
    Mat3d r_a_0_x = delta_q.toRotationMatrix() * skewSymmetric(acc_0 - linearized_ba);
    Mat3d r_a_1_x = new_delta_q.toRotationMatrix() * skewSymmetric(acc_1 - linearized_ba);

    double dt_2 = dt * dt, dt_3 = dt * dt * dt;
    F.block<3, 3>(0, 3) = -0.25 * dt_2 * (r_a_1_x * (Mat3d::Identity() - skewSymmetric(un_gyr * dt)) + r_a_0_x);
    F.block<3, 3>(0, 6) = Mat3d::Identity() * dt;
    F.block<3, 3>(0, 9) = -0.25 * dt_2 * (delta_q.toRotationMatrix() + new_delta_q.toRotationMatrix());
    F.block<3, 3>(0, 12) = 0.25 * dt_3 * r_a_1_x;
    F.block<3, 3>(3, 3) = Mat3d::Identity() - skewSymmetric(un_gyr * dt);
    F.block<3, 3>(3, 12) = -dt * Mat3d::Identity();
    F.block<3, 3>(6, 3) = -0.5 * dt * (r_a_1_x * (Mat3d::Identity() - skewSymmetric(un_gyr * dt)) + r_a_0_x);
    F.block<3, 3>(6, 9) = -0.5 * dt * (delta_q.toRotationMatrix() + new_delta_q.toRotationMatrix());
    F.block<3, 3>(6, 12) = 0.5 * dt_2 * r_a_1_x;

    Mat15x18d V = Mat15x18d::Zero();
    V.block<3, 3>(0, 0) = 0.25 * dt_2 * delta_q.toRotationMatrix();
    V.block<3, 3>(0, 3) = -0.125 * dt_3 * r_a_1_x;
    V.block<3, 3>(0, 6) = 0.25 * dt_2 * new_delta_q.toRotationMatrix();
    V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
    V.block<3, 3>(3, 3) = 0.5 * dt * Mat3d::Identity();
    V.block<3, 3>(3, 9) = V.block<3, 3>(3, 3);
    V.block<3, 3>(6, 0) = 0.5 * dt * delta_q.toRotationMatrix();
    V.block<3, 3>(6, 3) = -0.25 * dt_2 * r_a_1_x;
    V.block<3, 3>(6, 6) = 0.5 * dt * new_delta_q.toRotationMatrix();
    V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
    V.block<3, 3>(9, 12) = Mat3d::Identity() * dt;
    V.block<3, 3>(12, 15) = Mat3d::Identity() * dt;

    jacobian = F * jacobian;
    covariance = F * covariance * F.transpose() + V * noise * V.transpose();

    delta_p = new_delta_p;
    delta_q = new_delta_q;
    delta_v = new_delta_v;
}
void Integration::repropagate(const Vec3d &_linearized_ba, const Vec3d &_linearized_bg)
{
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setIdentity();
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (size_t i = 0; i < dt_buf.size(); i++)
        propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

Eigen::Matrix<double, 15, 1> Integration::evaluate(const Vec3d &pi, const Quatd &qi, const Vec3d &vi, const Vec3d &bai, const Vec3d &bgi,
                                                   const Vec3d &pj, const Quatd &qj, const Vec3d &vj, const Vec3d &baj, const Vec3d &bgj, const Vec3d &g_vec)
{
    Eigen::Matrix<double, 15, 1> residuals;
    Mat3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Mat3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);
    Mat3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Mat3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Mat3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Vec3d dba = bai - linearized_ba;
    Vec3d dbg = bgi - linearized_bg;

    Quatd corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg);
    Vec3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Vec3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(O_P, 0) = qi.inverse() * (0.5 * g_vec * sum_dt * sum_dt + pj - pi - vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = 2.0 * (corrected_delta_q.inverse() * (qi.inverse() * qj)).vec();
    residuals.block<3, 1>(O_V, 0) = qi.inverse() * (g_vec * sum_dt + vj - vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = baj - bai;
    residuals.block<3, 1>(O_BG, 0) = bgj - bgi;

    return residuals;
}