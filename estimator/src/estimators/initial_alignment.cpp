#include "initial_alignment.h"

Eigen::MatrixXd TangentBasis(Vec3d &g0)
{
    Vec3d b, c;
    Vec3d a = g0.normalized();
    Vec3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Vec3d *Bgs)
{
    Mat3d A;
    Vec3d b;
    Vec3d delta_bg;
    A.setZero();
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        Eigen::MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(3);
        tmp_b.setZero();
        // Eigen::Quaterniond q_ij(frame_i->second.r.transpose() * frame_j->second.r);
        Mat3d r_ij = frame_i->second.r.transpose() * frame_j->second.r;
        tmp_A = frame_j->second.integration->jacobian.template block<3, 3>(O_R, O_BG);
        // tmp_b = 2 * (Quatd(frame_j->second.integration->delta_q).inverse() * q_ij).vec(); // 这里需要验证一下是否等于 Log()
        tmp_b = Sophus::SO3d(frame_j->second.integration->delta_q.transpose() * r_ij).log();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.integration->repropagate(Vec3d::Zero(), Bgs[0]);
    }
}

bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d &g, Vec3d &t_ic, Eigen::VectorXd &x)
{

    double g_norm = g.norm();
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    Eigen::MatrixXd A(n_state, n_state);
    A.setZero();
    Eigen::VectorXd b(n_state);
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Mat3d::Identity();
        tmp_A.block<3, 3>(0, 6) = -frame_i->second.r.transpose() * dt * dt / 2 * Mat3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.r.transpose() * (frame_j->second.t - frame_i->second.t) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.integration->delta_p + frame_i->second.r.transpose() * frame_j->second.r * t_ic - t_ic;
        // cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Mat3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.r.transpose() * frame_j->second.r;
        tmp_A.block<3, 3>(3, 6) = -frame_i->second.r.transpose() * dt * Mat3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.integration->delta_v;

        Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    g = x.segment<3>(n_state - 4);
    if (fabs(g.norm() - g_norm) > 1.0 || s < 0)
        return false;
    RefineGravity(all_image_frame, g, t_ic, g_norm, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    if (s < 0.0)
        return false;
    else
        return true;
}

void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Vec3d &g, Vec3d &t_ic, double g_norm, Eigen::VectorXd &x)
{
    Vec3d g0 = g.normalized() * g_norm;
    Vec3d lx, ly;

    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;
    Eigen::MatrixXd A(n_state, n_state);
    A.setZero();
    Eigen::VectorXd b(n_state);
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    for (int k = 0; k < 4; k++)
    {
        Eigen::MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Mat3d::Identity();
            tmp_A.block<3, 2>(0, 6) = -frame_i->second.r.transpose() * dt * dt / 2 * Mat3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.r.transpose() * (frame_j->second.t - frame_i->second.t) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.integration->delta_p + frame_i->second.r.transpose() * frame_j->second.r * t_ic - t_ic + frame_i->second.r.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Mat3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.r.transpose() * frame_j->second.r;
            tmp_A.block<3, 2>(3, 6) = -frame_i->second.r.transpose() * dt * Mat3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.integration->delta_v + frame_i->second.r.transpose() * dt * Mat3d::Identity() * g0;

            Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * g_norm;
    }
    g = g0;
}

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d *Bgs, Vec3d &g, Vec3d &t_ic, Eigen::VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs);
    if (LinearAlignment(all_image_frame, g, t_ic, x))
        return true;
    else
        return false;
}