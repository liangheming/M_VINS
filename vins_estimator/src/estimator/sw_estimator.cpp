#include "sw_estimator.h"

SlidingWindowEstimator::SlidingWindowEstimator(const EstimatorConfig &config) : m_config(config)
{
    reset();
}
void SlidingWindowEstimator::reset()
{
    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        m_state.ps[i].setZero();
        m_state.vs[i].setZero();
        m_state.rs[i].setIdentity();
        m_state.bas[i].setZero();
        m_state.bgs[i].setZero();
        m_state.timestamps[i] = 0.0;
        m_state.dt_buf[i].clear();
        m_state.linear_acceleration_buf[i].clear();
        m_state.angular_velocity_buf[i].clear();
    }
    m_state.td = 0.0;
    m_state.frame_count = 0;
    m_state.first_imu = true;
    m_state.acc_0.setZero();
    m_state.gyro_0.setZero();
    m_state.initial_timestamp = 0.0;
    m_state.last_r.setIdentity();
    m_state.last_p.setZero();
    m_state.last_r0.setIdentity();
    m_state.last_p0.setZero();
    m_state.back_r0.setIdentity();
    m_state.back_p0.setZero();

    m_state.integrations.clear();
    m_state.integrations.resize(WINDOW_SIZE + 1, nullptr);
    m_state.temp_integration = nullptr;

    m_state.ric = m_config.ric;
    m_state.tic = m_config.tic;
    m_state.gravity = Vec3d(0.0, 0.0, m_config.g_norm);

    m_state.last_marginalization_info = nullptr;
    m_state.last_marginalization_parameter_blocks.clear();

    marginalization_flag = MarginFlag::MARGIN_OLD;
    solve_flag = SolveFlag::INITIAL;
    feature_manager.features.clear();
    all_image_frame.clear();
}

void SlidingWindowEstimator::processImu(const double &dt, const Vec3d &acc, const Vec3d &gyro)
{
    if (m_state.first_imu)
    {
        m_state.first_imu = false;
        m_state.acc_0 = acc;
        m_state.gyro_0 = gyro;
    }
    int j = m_state.frame_count;

    if (!m_state.integrations[j])
    {
        m_state.integrations[j].reset(new Integration(m_state.acc_0, m_state.gyro_0, m_state.bas[j], m_state.bgs[j]));
    }

    if (j != 0)
    {
        m_state.integrations[j]->push_back(dt, acc, gyro);
        if (solve_flag == SolveFlag::INITIAL)
        {
            m_state.temp_integration->push_back(dt, acc, gyro);
        }

        m_state.dt_buf[j].push_back(dt);
        m_state.linear_acceleration_buf[j].push_back(acc);
        m_state.angular_velocity_buf[j].push_back(gyro);

        Vec3d un_acc_0 = m_state.rs[j] * (m_state.acc_0 - m_state.bas[j]) - m_state.gravity;
        Vec3d un_gyr = 0.5 * (m_state.gyro_0 + gyro) - m_state.bgs[j];
        m_state.rs[j] *= deltaQ(un_gyr * dt).toRotationMatrix();
        Vec3d un_acc_1 = m_state.rs[j] * (acc - m_state.bas[j]) - m_state.gravity;
        Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        m_state.ps[j] += dt * m_state.vs[j] + 0.5 * dt * dt * un_acc;
        m_state.vs[j] += dt * un_acc;
    }
    m_state.acc_0 = acc;
    m_state.gyro_0 = gyro;
}
void SlidingWindowEstimator::processFeature(const TrackedFeatures &feats, double timestamp)
{
    if (feature_manager.addFeatureCheckParallax(m_state.frame_count, feats, m_state.td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    m_state.timestamps[m_state.frame_count] = timestamp;

    if (solve_flag == SolveFlag::INITIAL)
    {
        ImageFrame image_frame(feats, timestamp);
        image_frame.integration = m_state.temp_integration;
        all_image_frame.insert(std::make_pair(timestamp, image_frame));
        m_state.temp_integration.reset(new Integration(m_state.acc_0, m_state.gyro_0, m_state.bas[m_state.frame_count], m_state.bgs[m_state.frame_count]));
    }

    if (solve_flag == SolveFlag::INITIAL)
    {
        if (m_state.frame_count < WINDOW_SIZE)
        {
            m_state.frame_count++;
            return;
        }
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
            feature_manager.removeFailures();
        }
        slideWindow();
    }
    else
    {
        solveOdometry();
        slideWindow();
        feature_manager.removeFailures();
    }

    if (m_state.frame_count == WINDOW_SIZE)
    {
        m_state.last_r = m_state.rs[WINDOW_SIZE];
        m_state.last_p = m_state.ps[WINDOW_SIZE];
        m_state.last_r0 = m_state.rs[0];
        m_state.last_p0 = m_state.ps[0];
    }

    // if (solve_flag == NON_LINEAR)
    // {
    //     std::cout << "***********************" << temp_count++ << "***********************" << std::endl;
    //     std::cout << "ps:" << m_state.ps[WINDOW_SIZE].transpose() << std::endl;
    //     std::cout << "vs:" << m_state.vs[WINDOW_SIZE].transpose() << std::endl;
    //     std::cout << "bas:" << m_state.bas[WINDOW_SIZE].transpose() << std::endl;
    //     std::cout << "bgs:" << m_state.bgs[WINDOW_SIZE].transpose() << std::endl;
    //     std::cout << "qs:" << Quatd(m_state.rs[WINDOW_SIZE]).coeffs().transpose() << std::endl;
    //     Vec3d temp_p = m_state.ps[WINDOW_SIZE];
    //     Quatd temp_q = Quatd(m_state.rs[WINDOW_SIZE]);
    //     (*out_file) << std::fixed << std::setprecision(9) << timestamp << " " << temp_p.x() << " " << temp_p.y() << " " << temp_p.z() << " " << temp_q.x() << " " << temp_q.y() << " " << temp_q.z() << " " << temp_q.w() << std::endl;
    // }
}
void SlidingWindowEstimator::slideWindowNew()
{
    feature_manager.removeFront(m_state.frame_count);
}
void SlidingWindowEstimator::slideWindowOld()
{
    if (solve_flag == NON_LINEAR)
    {
        Mat3d r0, r1;
        Vec3d p0, p1;
        r0 = m_state.back_r0 * m_state.ric;
        r1 = m_state.rs[0] * m_state.ric;
        p0 = m_state.back_p0 + m_state.back_r0 * m_state.tic;
        p1 = m_state.ps[0] + m_state.rs[0] * m_state.tic;
        feature_manager.removeBackShiftDepth(r0, p0, r1, p1);
    }
    else
        feature_manager.removeBack();
}
void SlidingWindowEstimator::slideWindow()
{
    assert(m_state.frame_count == WINDOW_SIZE);
    if (marginalization_flag == MarginFlag::MARGIN_OLD)
    {
        double original_t0 = m_state.timestamps[0];
        m_state.back_r0 = m_state.rs[0];
        m_state.back_p0 = m_state.ps[0];

        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            m_state.rs[i].swap(m_state.rs[i + 1]);
            m_state.ps[i].swap(m_state.ps[i + 1]);
            m_state.vs[i].swap(m_state.vs[i + 1]);
            m_state.bas[i].swap(m_state.bas[i + 1]);
            m_state.bgs[i].swap(m_state.bgs[i + 1]);

            m_state.integrations[i].swap(m_state.integrations[i + 1]);
            m_state.dt_buf[i].swap(m_state.dt_buf[i + 1]);
            m_state.linear_acceleration_buf[i].swap(m_state.linear_acceleration_buf[i + 1]);
            m_state.angular_velocity_buf[i].swap(m_state.angular_velocity_buf[i + 1]);
            m_state.timestamps[i] = m_state.timestamps[i + 1];
        }

        m_state.rs[WINDOW_SIZE] = m_state.rs[WINDOW_SIZE - 1];
        m_state.ps[WINDOW_SIZE] = m_state.ps[WINDOW_SIZE - 1];
        m_state.vs[WINDOW_SIZE] = m_state.vs[WINDOW_SIZE - 1];
        m_state.bas[WINDOW_SIZE] = m_state.bas[WINDOW_SIZE - 1];
        m_state.bgs[WINDOW_SIZE] = m_state.bgs[WINDOW_SIZE - 1];

        m_state.integrations[WINDOW_SIZE].reset(new Integration(m_state.acc_0, m_state.gyro_0, m_state.bas[WINDOW_SIZE], m_state.bgs[WINDOW_SIZE]));
        m_state.timestamps[WINDOW_SIZE] = m_state.timestamps[WINDOW_SIZE - 1];

        m_state.dt_buf[WINDOW_SIZE].clear();
        m_state.linear_acceleration_buf[WINDOW_SIZE].clear();
        m_state.angular_velocity_buf[WINDOW_SIZE].clear();

        if (solve_flag == SolveFlag::INITIAL)
        {
            auto iter = all_image_frame.find(original_t0);
            all_image_frame.erase(all_image_frame.begin(), iter);
            all_image_frame.erase(iter);
        }
        slideWindowOld();
    }
    else
    {
        for (size_t i = 0; i < m_state.dt_buf[m_state.frame_count].size(); i++)
        {
            double tmp_dt = m_state.dt_buf[m_state.frame_count][i];
            Vec3d tmp_linear_acceleration = m_state.linear_acceleration_buf[m_state.frame_count][i];
            Vec3d tmp_angular_velocity = m_state.angular_velocity_buf[m_state.frame_count][i];

            m_state.integrations[m_state.frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

            m_state.dt_buf[m_state.frame_count - 1].push_back(tmp_dt);
            m_state.linear_acceleration_buf[m_state.frame_count - 1].push_back(tmp_linear_acceleration);
            m_state.angular_velocity_buf[m_state.frame_count - 1].push_back(tmp_angular_velocity);
        }

        m_state.rs[m_state.frame_count - 1] = m_state.rs[m_state.frame_count];
        m_state.ps[m_state.frame_count - 1] = m_state.ps[m_state.frame_count];
        m_state.vs[m_state.frame_count - 1] = m_state.vs[m_state.frame_count];
        m_state.bas[m_state.frame_count - 1] = m_state.bas[m_state.frame_count];
        m_state.bgs[m_state.frame_count - 1] = m_state.bgs[m_state.frame_count];

        m_state.timestamps[m_state.frame_count - 1] = m_state.timestamps[m_state.frame_count];

        m_state.integrations[WINDOW_SIZE].reset(new Integration(m_state.acc_0, m_state.gyro_0, m_state.bas[WINDOW_SIZE], m_state.bgs[WINDOW_SIZE]));
        m_state.dt_buf[WINDOW_SIZE].clear();
        m_state.linear_acceleration_buf[WINDOW_SIZE].clear();
        m_state.angular_velocity_buf[WINDOW_SIZE].clear();
        slideWindowNew();
    }

    if (solve_flag == SolveFlag::NON_LINEAR && !all_image_frame.empty())
    {
        std::map<double, ImageFrame>().swap(all_image_frame);
        m_state.temp_integration = nullptr;
    }
}

bool SlidingWindowEstimator::initialStructure()
{
    std::vector<SFMFeature> sfm_f;
    for (auto &it_per_id : feature_manager.features)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.observations)
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

    if (!relativePose(relative_r, relative_t, l))
        return false;
    GlobalSFM sfm;
    Mat3d rs[m_state.frame_count + 1];
    Vec3d ts[m_state.frame_count + 1];
    std::map<int, Vec3d> sfm_tracked_points;
    if (!sfm.construct(m_state.frame_count + 1, rs, ts, l, relative_r, relative_t, sfm_f, sfm_tracked_points))
    {
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    if (!beforeVisualInitialAlign(sfm_tracked_points, rs, ts))
        return false;

    Eigen::VectorXd xs;
    if (!visualInitialAlign(xs))
        return false;
    afterVisualInitialAlign(xs);

    // for (int i = 0; i <= m_state.frame_count; i++)
    // {
    //     std::cout << "=================" << i << "================" << std::endl;
    //     std::cout << Eigen::Quaterniond(m_state.rs[i]).coeffs().transpose() << std::endl;
    //     std::cout << "*****************" << std::endl;
    //     std::cout << m_state.ps[i].transpose() << std::endl;
    //     std::cout << m_state.vs[i].transpose() << std::endl;
    //     std::cout << m_state.bgs[i].transpose() << std::endl;
    // }
    // exit(0);
    return true;
}

void SlidingWindowEstimator::solveOdometry()
{
    if (solve_flag == NON_LINEAR)
    {
        feature_manager.triangulate(m_state.rs, m_state.ps, m_state.ric, m_state.tic);
        optimization();
    }
}

bool SlidingWindowEstimator::relativePose(Mat3d &relative_r, Vec3d &relative_t, int &l)
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
        if (average_parallax > m_config.axis_min_parallax && solveRelativeRT(corres, relative_r, relative_t, m_config.ransac_threshold))
        {
            l = i;
            return true;
        }
    }
    return false;
}
bool SlidingWindowEstimator::beforeVisualInitialAlign(std::map<int, Vec3d> &sfm_tracked_points, Mat3d *rs, Vec3d *ts)
{
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Vec3d>::iterator it;
    frame_it = all_image_frame.begin();

    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        cv::Mat r, rvec, t, D, tmp_r;
        // 这里为视觉惯性对齐做准备
        if ((frame_it->first) == m_state.timestamps[i])
        {
            frame_it->second.is_keyframe = true;
            frame_it->second.r = rs[i] * m_state.ric.transpose();
            frame_it->second.t = ts[i];
            i++;
            continue;
        }
        if ((frame_it->first) > m_state.timestamps[i])
            i++;
        Mat3d r_inital = rs[i].transpose();
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
        frame_it->second.r = r_pnp * m_state.ric.transpose();
        frame_it->second.t = t_pnp;
    }
    return true;
}

bool SlidingWindowEstimator::visualInitialAlign(Eigen::VectorXd &xs)
{
    return VisualIMUAlignment(all_image_frame, m_state.bgs, m_state.gravity, m_state.tic, xs);
}

void SlidingWindowEstimator::afterVisualInitialAlign(Eigen::VectorXd &xs)
{
    for (int i = 0; i <= m_state.frame_count; i++)
    {
        Mat3d ri = all_image_frame[m_state.timestamps[i]].r;
        Vec3d pi = all_image_frame[m_state.timestamps[i]].t;
        m_state.ps[i] = pi;
        m_state.rs[i] = ri;
        all_image_frame[m_state.timestamps[i]].is_keyframe = true;
    }

    feature_manager.triangulate(m_state.rs, m_state.ps, m_state.ric, m_state.tic);
    double s = (xs.tail<1>())(0);

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        m_state.integrations[i]->repropagate(Vec3d::Zero(), m_state.bgs[i]);
    }

    for (int i = m_state.frame_count; i >= 0; i--)
        m_state.ps[i] = s * m_state.ps[i] - m_state.rs[i] * m_state.tic - (s * m_state.ps[0] - m_state.rs[0] * m_state.tic);

    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_keyframe)
        {
            kv++;
            m_state.vs[kv] = frame_i->second.r * xs.segment<3>(kv * 3);
        }
    }

    for (auto &it_per_id : feature_manager.features)
    {
        int used_num = static_cast<int>(it_per_id.observations.size());
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Mat3d r0 = rotFromG(m_state.gravity);
    double yaw = rot2ypr(r0 * m_state.rs[0]).x();
    r0 = ypr2rot(Vec3d{-yaw, 0, 0}) * r0;
    m_state.gravity = r0 * m_state.gravity;

    for (int i = 0; i <= m_state.frame_count; i++)
    {
        m_state.ps[i] = r0 * m_state.ps[i];
        m_state.rs[i] = r0 * m_state.rs[i];
        m_state.vs[i] = r0 * m_state.vs[i];
    }
}

void SlidingWindowEstimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);

    feature_manager.updateCachedFeatures(MAX_FEATURE_SIZE);

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseParameterization();
        problem.AddParameterBlock(m_params.pose[i], POSE_SIZE, local_parameterization);
        problem.AddParameterBlock(m_params.speed_bias[i], SPEEDBIAS_SIZE);
    }

    ceres::LocalParameterization *local_parameterization = new PoseParameterization();
    problem.AddParameterBlock(m_params.ext_pose, POSE_SIZE, local_parameterization);

    if (!m_config.estimate_ext)
        problem.SetParameterBlockConstant(m_params.ext_pose);

    vector2double();

    if (m_state.last_marginalization_info != nullptr)
    {
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(m_state.last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, m_state.last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (m_state.integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(m_state.integrations[j], m_state.gravity);
        problem.AddResidualBlock(imu_factor, NULL, m_params.pose[i], m_params.speed_bias[i], m_params.pose[j], m_params.speed_bias[j]);
    }

    for (size_t i = 0; i < feature_manager.cached_features.size(); i++)
    {
        auto &iter = feature_manager.cached_features[i];
        int imu_i = iter->start_frame, imu_j = imu_i - 1;
        Vec3d pts_i = iter->observations[0].point;
        for (auto &observation : iter->observations)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vec3d pts_j = observation.point;
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j, m_config.proj_sqrt_info);
            problem.AddResidualBlock(f, loss_function, m_params.pose[imu_i], m_params.pose[imu_j], m_params.ext_pose, m_params.features[i]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 8;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = 0.04 * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = 0.04;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    double2vector();

    prepareMarginInfo();
}

void SlidingWindowEstimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        m_params.pose[i][0] = m_state.ps[i].x();
        m_params.pose[i][1] = m_state.ps[i].y();
        m_params.pose[i][2] = m_state.ps[i].z();
        Quatd q(m_state.rs[i]);
        m_params.pose[i][3] = q.x();
        m_params.pose[i][4] = q.y();
        m_params.pose[i][5] = q.z();
        m_params.pose[i][6] = q.w();

        m_params.speed_bias[i][0] = m_state.vs[i].x();
        m_params.speed_bias[i][1] = m_state.vs[i].y();
        m_params.speed_bias[i][2] = m_state.vs[i].z();

        m_params.speed_bias[i][3] = m_state.bas[i].x();
        m_params.speed_bias[i][4] = m_state.bas[i].y();
        m_params.speed_bias[i][5] = m_state.bas[i].z();

        m_params.speed_bias[i][6] = m_state.bgs[i].x();
        m_params.speed_bias[i][7] = m_state.bgs[i].y();
        m_params.speed_bias[i][8] = m_state.bgs[i].z();
    }
    m_params.ext_pose[0] = m_state.tic.x();
    m_params.ext_pose[1] = m_state.tic.y();
    m_params.ext_pose[2] = m_state.tic.z();
    Quatd q(m_state.ric);
    m_params.ext_pose[3] = q.x();
    m_params.ext_pose[4] = q.y();
    m_params.ext_pose[5] = q.z();
    m_params.ext_pose[6] = q.w();

    for (size_t i = 0; i < feature_manager.cached_features.size(); i++)
    {
        m_params.features[i][0] = 1.0 / feature_manager.cached_features[i]->estimated_depth;
    }
}

void SlidingWindowEstimator::double2vector()
{
    Vec3d origin_r0 = rot2ypr(m_state.rs[0]);
    Vec3d origin_p0 = m_state.ps[0];
    Vec3d optimized_r0 = rot2ypr(Quatd(m_params.pose[0][6], m_params.pose[0][3], m_params.pose[0][4], m_params.pose[0][5]).normalized().toRotationMatrix());
    double y_diff = origin_r0.x() - optimized_r0.x();
    Mat3d rot_diff = ypr2rot(Vec3d(y_diff, 0, 0));
    if (abs(abs(origin_r0.y()) - 90) < 1.0 || abs(abs(optimized_r0.y()) - 90) < 1.0)
    {
        rot_diff = m_state.rs[0] * Quatd(m_params.pose[0][6], m_params.pose[0][3], m_params.pose[0][4], m_params.pose[0][5]).normalized().toRotationMatrix().transpose();
    }
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        m_state.rs[i] = rot_diff * Quatd(m_params.pose[i][6], m_params.pose[i][3], m_params.pose[i][4], m_params.pose[i][5]).normalized().toRotationMatrix();

        m_state.ps[i] = rot_diff * Vec3d(m_params.pose[i][0] - m_params.pose[0][0], m_params.pose[i][1] - m_params.pose[0][1], m_params.pose[i][2] - m_params.pose[0][2]) + origin_p0;

        m_state.vs[i] = rot_diff * Vec3d(m_params.speed_bias[i][0], m_params.speed_bias[i][1], m_params.speed_bias[i][2]);

        m_state.bas[i] = Vec3d(m_params.speed_bias[i][3], m_params.speed_bias[i][4], m_params.speed_bias[i][5]);

        m_state.bgs[i] = Vec3d(m_params.speed_bias[i][6], m_params.speed_bias[i][7], m_params.speed_bias[i][8]);
    }
    m_state.tic = Vec3d(m_params.ext_pose[0],
                        m_params.ext_pose[1],
                        m_params.ext_pose[2]);
    m_state.ric = Quatd(m_params.ext_pose[6], m_params.ext_pose[3], m_params.ext_pose[4], m_params.ext_pose[5]).normalized().toRotationMatrix();

    for (size_t i = 0; i < feature_manager.cached_features.size(); i++)
    {
        if (m_params.features[i][0] <= 0.001)
        {
            feature_manager.cached_features[i]->solve_flag = 2;
            continue;
        }
        feature_manager.cached_features[i]->solve_flag = 1;
        double estimate_depth = 1.0 / m_params.features[i][0];
        feature_manager.cached_features[i]->estimated_depth = estimate_depth;
    }
}

void SlidingWindowEstimator::prepareMarginInfo()
{
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    if (marginalization_flag == MARGIN_OLD)
    {
        std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();
        vector2double();
        if (m_state.last_marginalization_info != nullptr)
        {
            std::vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(m_state.last_marginalization_parameter_blocks.size()); i++)
            {
                if (m_state.last_marginalization_parameter_blocks[i] == m_params.pose[0] ||
                    m_state.last_marginalization_parameter_blocks[i] == m_params.speed_bias[0])
                    drop_set.push_back(i);
            }
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(m_state.last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           m_state.last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (m_state.integrations[1]->sum_dt < 10.0)
        {
            IMUFactor *imu_factor = new IMUFactor(m_state.integrations[1], m_state.gravity);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           std::vector<double *>{m_params.pose[0], m_params.speed_bias[0], m_params.pose[1], m_params.speed_bias[1]},
                                                                           std::vector<int>{0, 1});
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        for (size_t i = 0; i < feature_manager.cached_features.size(); i++)
        {
            auto &it_per_id = feature_manager.cached_features[i];
            int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;
            Vec3d pts_i = it_per_id->observations[0].point;
            for (auto &obser : it_per_id->observations)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;
                Vec3d pts_j = obser.point;

                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j, m_config.proj_sqrt_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                               std::vector<double *>{m_params.pose[imu_i], m_params.pose[imu_j], m_params.ext_pose, m_params.features[i]},
                                                                               std::vector<int>{0, 3});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        marginalization_info->preMarginalize();
        marginalization_info->marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(m_params.pose[i])] = m_params.pose[i - 1];
            addr_shift[reinterpret_cast<long>(m_params.speed_bias[i])] = m_params.speed_bias[i - 1];
        }
        addr_shift[reinterpret_cast<long>(m_params.ext_pose)] = m_params.ext_pose;
        std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        m_state.last_marginalization_info = marginalization_info;
        m_state.last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (m_state.last_marginalization_info != nullptr &&
            std::count(std::begin(m_state.last_marginalization_parameter_blocks), std::end(m_state.last_marginalization_parameter_blocks), m_params.pose[WINDOW_SIZE - 1]))
        {
            std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();
            vector2double();

            std::vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(m_state.last_marginalization_parameter_blocks.size()); i++)
            {
                if (m_state.last_marginalization_parameter_blocks[i] == m_params.pose[WINDOW_SIZE - 1])
                    drop_set.push_back(i);
            }

            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(m_state.last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           m_state.last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);

            marginalization_info->preMarginalize();

            marginalization_info->marginalize();

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(m_params.pose[i])] = m_params.pose[i - 1];
                    addr_shift[reinterpret_cast<long>(m_params.speed_bias[i])] = m_params.speed_bias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(m_params.pose[i])] = m_params.pose[i];
                    addr_shift[reinterpret_cast<long>(m_params.speed_bias[i])] = m_params.speed_bias[i];
                }
            }

            addr_shift[reinterpret_cast<long>(m_params.ext_pose)] = m_params.ext_pose;
            std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            m_state.last_marginalization_info = marginalization_info;
            m_state.last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
}