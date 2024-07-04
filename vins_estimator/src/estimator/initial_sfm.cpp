#include "initial_sfm.h"

bool GlobalSFM::solveFrameByPnP(Mat3d &r_initial, Vec3d &t_initial, int i, std::vector<SFMFeature> &sfm_f)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state != true)
            continue;
        Vec2d point2d;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == i) // 这里有不必要的计算
            {
                Vec2d img_pts = sfm_f[j].observation[k].second;
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3);
                break;
            }
        }
    }

    if (int(pts_2_vector.size()) < 10)
        return false;

    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(r_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(t_initial, t);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if (!pnp_succ)
    {
        return false;
    }
    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd r_pnp;
    cv::cv2eigen(r, r_pnp);
    Eigen::MatrixXd t_pnp;
    cv::cv2eigen(t, t_pnp);
    r_initial = r_pnp;
    t_initial = t_pnp;
    return true;
}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &pose0, Eigen::Matrix<double, 3, 4> &pose1,
                                 Vec2d &point0, Vec2d &point1, Vec3d &point_3d)
{
    Mat4d design_matrix = Mat4d::Zero();
    design_matrix.row(0) = point0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = point0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = point1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = point1[1] * pose1.row(2) - pose1.row(1);

    Vec4d triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &pose0,
                                     int frame1, Eigen::Matrix<double, 3, 4> &pose1,
                                     std::vector<SFMFeature> &sfm_f)
{
    assert(frame0 != frame1);
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        bool has_0 = false, has_1 = false;
        Vec2d point0;
        Vec2d point1;
        for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
        {
            if (sfm_f[j].observation[k].first == frame0)
            {
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if (sfm_f[j].observation[k].first == frame1)
            {
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }

        if (has_0 && has_1)
        {
            Vec3d point_3d;
            triangulatePoint(pose0, pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }
}
bool GlobalSFM::construct(int frame_num, Mat3d *rs, Vec3d *ts, int l,
                          const Mat3d relative_r, const Vec3d relative_t,
                          std::vector<SFMFeature> &sfm_f, std::map<int, Vec3d> &sfm_tracked_points)
{
    feature_num = sfm_f.size();
    // 设置枢纽帧
    rs[l].setIdentity();
    ts[l].setZero();
    rs[frame_num - 1] = relative_r;
    ts[frame_num - 1] = relative_t;

    Mat3d c_rotations[frame_num];
    Vec3d c_translations[frame_num];
    Quatd c_quats[frame_num];
    double c_rotations_arr[frame_num][4];
    double c_translations_arr[frame_num][3];
    Eigen::Matrix<double, 3, 4> poses[frame_num];

    c_quats[l] = Quatd(rs[l].transpose());
    c_rotations[l] = rs[l].transpose();
    c_translations[l] = -1.0 * c_rotations[l] * ts[l];
    poses[l].block<3, 3>(0, 0) = c_rotations[l];
    poses[l].block<3, 1>(0, 3) = c_translations[l];

    c_quats[frame_num - 1] = Quatd(rs[frame_num - 1].transpose());
    c_rotations[frame_num - 1] = rs[frame_num - 1].transpose();
    c_translations[frame_num - 1] = -1.0 * c_rotations[frame_num - 1] * ts[frame_num - 1];
    poses[frame_num - 1].block<3, 3>(0, 0) = c_rotations[frame_num - 1];
    poses[frame_num - 1].block<3, 1>(0, 3) = c_translations[frame_num - 1];
    // 从枢纽帧向右边，进行三角化 然后3d-2d 进行pnp
    for (int i = l; i < frame_num - 1; i++)
    {
        if (i > l)
        {
            Mat3d r_initial = c_rotations[i - 1];
            Vec3d t_initial = c_translations[i - 1];
            if (!solveFrameByPnP(r_initial, t_initial, i, sfm_f))
                return false;
            c_rotations[i] = r_initial;
            c_translations[i] = t_initial;
            c_quats[i] = c_rotations[i];
            poses[i].block<3, 3>(0, 0) = c_rotations[i];
            poses[i].block<3, 1>(0, 3) = c_translations[i];
        }
        triangulateTwoFrames(i, poses[i], frame_num - 1, poses[frame_num - 1], sfm_f);
    }
    // 尽量将能被l帧看到的特征点进行三角化，因为接下来要反向的进行三角化和pnp
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, poses[l], i, poses[i], sfm_f);

    for (int i = l - 1; i >= 0; i--)
    {
        Mat3d r_initial = c_rotations[i + 1];
        Vec3d t_initial = c_translations[i + 1];
        if (!solveFrameByPnP(r_initial, t_initial, i, sfm_f))
            return false;
        c_rotations[i] = r_initial;
        c_translations[i] = t_initial;
        c_quats[i] = c_rotations[i];
        poses[i].block<3, 3>(0, 0) = c_rotations[i];
        poses[i].block<3, 1>(0, 3) = c_translations[i];
        // triangulate
        triangulateTwoFrames(i, poses[i], l, poses[l], sfm_f);
    }

    // 将所有还未被三角化的点进行三角化
    for (int j = 0; j < feature_num; j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)
        {
            Vec2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vec3d point_3d;
            triangulatePoint(poses[frame_0], poses[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
        }
    }

    // 进行BA
    ceres::Problem problem;

    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    for (int i = 0; i < frame_num; i++)
    {
        c_translations_arr[i][0] = c_translations[i].x();
        c_translations_arr[i][1] = c_translations[i].y();
        c_translations_arr[i][2] = c_translations[i].z();

        c_rotations_arr[i][0] = c_quats[i].w();
        c_rotations_arr[i][1] = c_quats[i].x();
        c_rotations_arr[i][2] = c_quats[i].y();
        c_rotations_arr[i][3] = c_quats[i].z();
        problem.AddParameterBlock(c_rotations_arr[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translations_arr[i], 3);
        if (i == l)
        {
            problem.SetParameterBlockConstant(c_rotations_arr[i]);
        }
        // 这里为什么要设置最后一帧的旋转也为常量？
        if (i == l || i == frame_num - 1)
        {
            problem.SetParameterBlockConstant(c_translations_arr[i]);
        }
    }
    for (int i = 0; i < feature_num; i++)
    {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction *cost_function = ReprojectionError3D::Create(
                sfm_f[i].observation[j].second.x(),
                sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotations_arr[l], c_translations_arr[l], sfm_f[i].position);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost >= 5e-03)
        return false;

    for (int i = 0; i < frame_num; i++)
    {
        rs[i] = Quatd(c_rotations_arr[i][0], c_rotations_arr[i][1], c_rotations_arr[i][2], c_rotations_arr[i][3]).toRotationMatrix().transpose();
        ts[i] = -1.0 * (rs[i] * Vec3d(c_translations_arr[i][0], c_translations_arr[i][1], c_translations_arr[i][2]));
    }

    for (int i = 0; i < (int)sfm_f.size(); i++)
    {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vec3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    return true;
}