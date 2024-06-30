#include "sw_estimator.h"

SlideWindowEstimator::SlideWindowEstimator(const SWConfig &config) : feature_manager(rs), m_config(config)
{
    g = Vec3d(0.0, 0.0, -9.81007);
    clearState();
}
void SlideWindowEstimator::clearState()
{
    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ps[i].setZero();
        vs[i].setZero();
        rs[i].setIdentity();
        bas[i].setZero();
        bgs[i].setZero();
        timestamps[i] = 0.0;
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
    }
    time_delay = 0.0;
    integrations.clear();
    integrations.resize(WINDOW_SIZE + 1, nullptr);
    temp_integration = nullptr;

    m_state.frame_count = 0;
    m_state.first_imu = true;
    m_state.acc_0.setZero();
    m_state.gyro_0.setZero();
    m_state.initial_timestamp = 0.0;

    marginalization_flag = SWMarginFlag::MARGIN_OLD;
    solve_flag = SolveFlag::INITIAL;
    all_image_frame.clear();
    last_r = Mat3d::Identity();
    last_r0 = Mat3d::Identity();
    r_ic = Mat3d::Identity();
    last_p = Vec3d::Zero();
    last_p0 = Vec3d::Zero();
    t_ic = Vec3d::Zero();
}
void SlideWindowEstimator::processImu(const double &dt, const Vec3d &acc, const Vec3d &gyro)
{
    if (m_state.first_imu)
    {
        m_state.first_imu = false;
        m_state.acc_0 = acc;
        m_state.gyro_0 = gyro;
    }
    int j = m_state.frame_count;

    if (!integrations[j])
    {
        integrations[j].reset(new Integration(m_state.acc_0, m_state.gyro_0, bas[j], bgs[j]));
    }

    if (j != 0)
    {
        integrations[j]->push_back(dt, acc, gyro);
        temp_integration->push_back(dt, acc, gyro);

        dt_buf[j].push_back(dt);
        linear_acceleration_buf[j].push_back(acc);
        angular_velocity_buf[j].push_back(gyro);

        Vec3d un_acc_0 = rs[j] * (m_state.acc_0 - bas[j]) + g;
        Vec3d un_gyr = 0.5 * (m_state.gyro_0 + gyro) - bgs[j];
        rs[j] *= Sophus::SO3d::exp(un_gyr * dt).matrix();
        Vec3d un_acc_1 = rs[j] * (acc - bas[j]) + g;
        Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        ps[j] += dt * vs[j] + 0.5 * dt * dt * un_acc;
        vs[j] += dt * un_acc;
    }
    m_state.acc_0 = acc;
    m_state.gyro_0 = gyro;
}
void SlideWindowEstimator::processFeature(const Feats &feats, double timestamp)
{ // 关键帧判断
    if (feature_manager.addFeatureCheckParallax(m_state.frame_count, feats, time_delay))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    timestamps[m_state.frame_count] = timestamp;
    ImageFrame image_frame(feats, timestamp);
    image_frame.integration = temp_integration;
    all_image_frame.insert(std::make_pair(timestamp, image_frame));
    temp_integration = std::make_shared<Integration>(m_state.acc_0, m_state.gyro_0, bas[m_state.frame_count], bgs[m_state.frame_count]);

    if (solve_flag == INITIAL)
    {
        if (m_state.frame_count == WINDOW_SIZE)
        {
            bool initial_success = false;
            if (timestamp - m_state.initial_timestamp > 0.1)
            {
                initial_success = initialStructure();
                m_state.initial_timestamp = timestamp;
            }
            if (initial_success)
            {
                solve_flag = SolveFlag::NON_LINEAR;
                solveOdometry();
                slideWindow();
                // 删除一些丢失的特征点
                feature_manager.removeFailures();
                last_r = rs[WINDOW_SIZE];
                last_p = ps[WINDOW_SIZE];
                last_r0 = rs[0];
                last_p0 = ps[0];
            }
            else
            {
                slideWindow();
            }
        }
        else
        {
            m_state.frame_count++;
        }
    }
    else
    {

        solveOdometry();
        // 失败检查
        slideWindow();
        feature_manager.removeFailures();
        last_r = rs[WINDOW_SIZE];
        last_p = ps[WINDOW_SIZE];
        last_r0 = rs[0];
        last_p0 = ps[0];
    }

    if (solve_flag == NON_LINEAR)
    {
        std::cout << "==============" << temp_count++ << "=================" << std::endl;
        std::cout << "ps:" << ps[WINDOW_SIZE].transpose() << std::endl;
        std::cout << "vs:" << vs[WINDOW_SIZE].transpose() << std::endl;
        std::cout << "bas:" << bas[WINDOW_SIZE].transpose() << std::endl;
        std::cout << "bgs:" << bgs[WINDOW_SIZE].transpose() << std::endl;
    }
}
bool SlideWindowEstimator::relativePose(Mat3d &relative_r, Vec3d &relative_t, int &l)
{
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        std::vector<std::pair<Vec3d, Vec3d>> corres;
        feature_manager.getCorresponding(i, WINDOW_SIZE, corres);
        if (corres.size() <= 20)
            continue;
        double sum_parallax = 0;

        for (int j = 0; j < int(corres.size()); j++)
        {
            Vec2d pts_0(corres[j].first(0), corres[j].first(1));
            Vec2d pts_1(corres[j].second(0), corres[j].second(1));
            double parallax = (pts_0 - pts_1).norm();
            sum_parallax += parallax;
        }
        double average_parallax = 1.0 * sum_parallax / double(corres.size());
        if (average_parallax * 460 > 30 && solveRelativeRT(corres, relative_r, relative_t))
        {
            {
                l = i;
                return true;
            }
        }
    }
    return false;
}
bool SlideWindowEstimator::visualInitialAlign()
{
    Eigen::VectorXd xs;
    bool result = VisualIMUAlignment(all_image_frame, bgs, g, t_ic, xs);
    if (!result)
        return result;

    for (int i = 0; i <= m_state.frame_count; i++)
    {
        Mat3d ri = all_image_frame[timestamps[i]].r;
        Vec3d pi = all_image_frame[timestamps[i]].t;
        ps[i] = pi;
        rs[i] = ri;
        all_image_frame[timestamps[i]].is_keyframe = true;
    }

    Eigen::VectorXd dep = feature_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    feature_manager.clearDepth(dep);
    feature_manager.setRic(r_ic);
    feature_manager.triangulate(rs, ps, r_ic, t_ic);

    double s = (xs.tail<1>())(0);

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        integrations[i]->repropagate(Vec3d::Zero(), bgs[i]);
    }
    // 相对第一帧的位姿，坐标系是轴帧
    for (int i = m_state.frame_count; i >= 0; i--)
        ps[i] = s * ps[i] - rs[i] * t_ic - (s * ps[0] - rs[0] * t_ic);

    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_keyframe)
        {
            kv++;
            vs[kv] = frame_i->second.r * xs.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : feature_manager.features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Mat3d r0 = Quatd::FromTwoVectors(g.normalized(), Vec3d(0, 0, -1.0)).toRotationMatrix();
    double yaw = rot2ypr(r0 * rs[0]).x();
    r0 = ypr2rot(Eigen::Vector3d{-yaw, 0, 0}) * r0;
    g = Vec3d(0, 0, -1.0) * g.norm();

    for (int i = 0; i <= m_state.frame_count; i++)
    {
        ps[i] = r0 * ps[i];
        rs[i] = r0 * rs[i];
        vs[i] = r0 * vs[i];

        std::cout << "=================" << i << "================" << std::endl;
        std::cout << ps[i].transpose() << std::endl;
        std::cout << Eigen::Quaterniond(rs[i]).coeffs().transpose() << std::endl;
        std::cout << vs[i].transpose() << std::endl;
    }
    return true;
}

bool SlideWindowEstimator::initialStructure()
{
    std::vector<SFMFeature> sfm_f;
    for (auto &it_per_id : feature_manager.features)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vec3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Vec2d(pts_j.x(), pts_j.y()));
        }
        sfm_f.push_back(tmp_feature);
    }

    Mat3d relative_r;
    Vec3d relative_t;
    int l;

    // 确定枢纽帧
    if (!relativePose(relative_r, relative_t, l))
        return false;
    // 纯视觉初始化
    GlobalSFM sfm;
    Quatd qs[m_state.frame_count + 1];
    Vec3d ts[m_state.frame_count + 1];
    std::map<int, Vec3d> sfm_tracked_points;
    if (!sfm.construct(m_state.frame_count + 1, qs, ts, l, relative_r, relative_t, sfm_f, sfm_tracked_points))
    {
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // 对所有临时帧进行pnp求解
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Vec3d>::iterator it;
    frame_it = all_image_frame.begin();

    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        cv::Mat r, rvec, t, D, tmp_r;
        // 这里为视觉惯性对齐做准备
        if ((frame_it->first) == timestamps[i])
        {
            frame_it->second.is_keyframe = true;
            frame_it->second.r = qs[i].toRotationMatrix() * r_ic.transpose();
            frame_it->second.t = ts[i];
            i++;
            continue;
        }
        if ((frame_it->first) > timestamps[i])
        {
            i++;
        }

        Mat3d r_inital = (qs[i].inverse()).toRotationMatrix();
        Vec3d t_inital = -r_inital * ts[i];
        cv::eigen2cv(r_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(t_inital, t);

        frame_it->second.is_keyframe = false;

        std::vector<cv::Point3f> pts_3_vector;
        std::vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.feats)
        {
            int feature_id = id_pts.first;
            it = sfm_tracked_points.find(feature_id);
            if (it != sfm_tracked_points.end())
            {
                Vec3d world_pts = it->second;
                cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                pts_3_vector.push_back(pts_3);
                Vec2d img_pts = id_pts.second.head<2>();
                cv::Point2f pts_2(img_pts(0), img_pts(1));
                pts_2_vector.push_back(pts_2);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
            return false;
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
            return false;
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd r_pnp, tmp_r_pnp;
        cv::cv2eigen(r, tmp_r_pnp);
        r_pnp = tmp_r_pnp.transpose();
        Eigen::MatrixXd t_pnp;
        cv::cv2eigen(t, t_pnp);
        t_pnp = r_pnp * (-t_pnp);
        frame_it->second.r = r_pnp * r_ic.transpose();
        frame_it->second.t = t_pnp;
    }

    // 视觉惯性对齐
    if (visualInitialAlign())
        return true;
    else
        return false;
}

void SlideWindowEstimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_pose[i][0] = ps[i].x();
        para_pose[i][1] = ps[i].y();
        para_pose[i][2] = ps[i].z();
        Quatd q(rs[i]);
        para_pose[i][3] = q.x();
        para_pose[i][4] = q.y();
        para_pose[i][5] = q.z();
        para_pose[i][6] = q.w();

        para_speed_bias[i][0] = vs[i].x();
        para_speed_bias[i][1] = vs[i].y();
        para_speed_bias[i][2] = vs[i].z();

        para_speed_bias[i][3] = bas[i].x();
        para_speed_bias[i][4] = bas[i].y();
        para_speed_bias[i][5] = bas[i].z();

        para_speed_bias[i][6] = bgs[i].x();
        para_speed_bias[i][7] = bgs[i].y();
        para_speed_bias[i][8] = bgs[i].z();
    }

    para_ex_pose[0] = t_ic.x();
    para_ex_pose[1] = t_ic.y();
    para_ex_pose[2] = t_ic.z();
    Quatd q(r_ic);
    para_ex_pose[3] = q.x();
    para_ex_pose[4] = q.y();
    para_ex_pose[5] = q.z();
    para_ex_pose[6] = q.w();

    Eigen::VectorXd dep = feature_manager.getDepthVector();
    int max_size = std::min(int(dep.size()), m_config.max_feature);
    for (int i = 0; i < max_size; i++)
        para_features[i][0] = dep(i);
}
void SlideWindowEstimator::double2vector()
{
    Vec3d origin_r0 = rot2ypr(rs[0]);
    Vec3d origin_p0 = ps[0];

    Vec3d optimized_r0 = rot2ypr(Quatd(para_pose[0][6], para_pose[0][3], para_pose[0][4], para_pose[0][5]).toRotationMatrix());
    double y_diff = origin_r0.x() - optimized_r0.x();
    Mat3d rot_diff = ypr2rot(Vec3d(y_diff, 0, 0));
    if (abs(abs(origin_r0.y()) - 90) < 1.0 || abs(abs(optimized_r0.y()) - 90) < 1.0)
    {
        rot_diff = rs[0] * Quatd(para_pose[0][6], para_pose[0][3], para_pose[0][4], para_pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        rs[i] = rot_diff * Quatd(para_pose[i][6], para_pose[i][3], para_pose[i][4], para_pose[i][5]).normalized().toRotationMatrix();

        ps[i] = rot_diff * Vec3d(para_pose[i][0] - para_pose[0][0], para_pose[i][1] - para_pose[0][1], para_pose[i][2] - para_pose[0][2]) + origin_p0;

        vs[i] = rot_diff * Vec3d(para_speed_bias[i][0], para_speed_bias[i][1], para_speed_bias[i][2]);

        bas[i] = Vec3d(para_speed_bias[i][3], para_speed_bias[i][4], para_speed_bias[i][5]);

        bgs[i] = Vec3d(para_speed_bias[i][6], para_speed_bias[i][7], para_speed_bias[i][8]);
    }

    t_ic = Vec3d(para_ex_pose[0],
                 para_ex_pose[1],
                 para_ex_pose[2]);
    r_ic = Quatd(para_ex_pose[6], para_ex_pose[3], para_ex_pose[4], para_ex_pose[5]).toRotationMatrix();

    Eigen::VectorXd dep = feature_manager.getDepthVector();
    int max_size = std::min(int(dep.size()), m_config.max_feature);
    for (int i = 0; i < max_size; i++)
        dep(i) = para_features[i][0];
    feature_manager.setDepth(dep);
}

void SlideWindowEstimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseParameterization();
        problem.AddParameterBlock(para_pose[i], 7, local_parameterization);
        problem.AddParameterBlock(para_speed_bias[i], 9);
    }

    ceres::LocalParameterization *local_parameterization = new PoseParameterization();
    problem.AddParameterBlock(para_ex_pose, 7, local_parameterization);
    problem.SetParameterBlockConstant(para_ex_pose);

    vector2double();

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(integrations[j], g);
        // std::cout << "add imu factor " << i << " " << j << std::endl;
        // std::cout << imu_factor->integration->delta_p.transpose() << j << std::endl;
        // std::cout << imu_factor->integration->delta_v.transpose() << j << std::endl;
        problem.AddResidualBlock(imu_factor, NULL, para_pose[i], para_speed_bias[i], para_pose[j], para_speed_bias[j]);
    }
    Mat2d project_sqrt_info = 460.0 / 1.5 * Mat2d::Identity();

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : feature_manager.features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        if (feature_index >= m_config.max_feature)
            break;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vec3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vec3d pts_j = it_per_frame.point;

            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j, project_sqrt_info);
            // std::cout << "add feature " << feature_index << " " << imu_i << " " << imu_j << " | " << para_features[feature_index][0] << std::endl;
            problem.AddResidualBlock(f, loss_function, para_pose[imu_i], para_pose[imu_j], para_ex_pose, para_features[feature_index]);
            f_m_cnt++;
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 8;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = 0.04 * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = 0.04;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    double2vector();
}

void SlideWindowEstimator::solveOdometry()
{
    if (m_state.frame_count < WINDOW_SIZE)
        return;
    if (solve_flag == NON_LINEAR)
    {
        feature_manager.triangulate(rs, ps, r_ic, t_ic);
        optimization();
    }
}

void SlideWindowEstimator::slideWindowNew()
{
    feature_manager.removeFront(m_state.frame_count);
}

void SlideWindowEstimator::slideWindowOld()
{
    if (solve_flag == NON_LINEAR)
    {
        Mat3d r0, r1;
        Vec3d p0, p1;
        r0 = back_r0 * r_ic;
        r1 = rs[0] * r_ic;
        p0 = back_p0 + back_r0 * t_ic;
        p1 = ps[0] + rs[0] * t_ic;
        feature_manager.removeBackShiftDepth(r0, p0, r1, p1);
    }
    else
        feature_manager.removeBack();
}

void SlideWindowEstimator::slideWindow()
{
    if (marginalization_flag == MARGIN_OLD)
    {
        back_r0 = rs[0];
        back_p0 = ps[0];
        if (m_state.frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                rs[i].swap(rs[i + 1]);
                integrations[i].swap(integrations[i + 1]);
                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                timestamps[i] = timestamps[i + 1];
                ps[i].swap(ps[i + 1]);
                vs[i].swap(vs[i + 1]);
                bas[i].swap(bas[i + 1]);
                bgs[i].swap(bgs[i + 1]);
            }

            timestamps[WINDOW_SIZE] = timestamps[WINDOW_SIZE - 1];
            ps[WINDOW_SIZE] = ps[WINDOW_SIZE - 1];
            vs[WINDOW_SIZE] = vs[WINDOW_SIZE - 1];
            rs[WINDOW_SIZE] = rs[WINDOW_SIZE - 1];
            bas[WINDOW_SIZE] = bas[WINDOW_SIZE - 1];
            bgs[WINDOW_SIZE] = bgs[WINDOW_SIZE - 1];

            integrations[WINDOW_SIZE].reset(new Integration(m_state.acc_0, m_state.gyro_0, bas[WINDOW_SIZE], bgs[WINDOW_SIZE]));
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            if (!all_image_frame.empty())
                std::map<double, ImageFrame>().swap(all_image_frame);
            slideWindowOld();
        }
    }
    else
    {
        if (m_state.frame_count == WINDOW_SIZE)
        {
            for (size_t i = 0; i < dt_buf[m_state.frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[m_state.frame_count][i];
                Vec3d tmp_linear_acceleration = linear_acceleration_buf[m_state.frame_count][i];
                Vec3d tmp_angular_velocity = angular_velocity_buf[m_state.frame_count][i];

                integrations[m_state.frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[m_state.frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[m_state.frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[m_state.frame_count - 1].push_back(tmp_angular_velocity);
            }

            timestamps[m_state.frame_count - 1] = timestamps[m_state.frame_count];
            ps[m_state.frame_count - 1] = ps[m_state.frame_count];
            vs[m_state.frame_count - 1] = vs[m_state.frame_count];
            rs[m_state.frame_count - 1] = rs[m_state.frame_count];
            bas[m_state.frame_count - 1] = bas[m_state.frame_count];
            bgs[m_state.frame_count - 1] = bgs[m_state.frame_count];

            integrations[WINDOW_SIZE].reset(new Integration(m_state.acc_0, m_state.gyro_0, bas[WINDOW_SIZE], bgs[WINDOW_SIZE]));
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            slideWindowNew();
        }
    }
}