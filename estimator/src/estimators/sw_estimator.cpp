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
                last_r = rs[WINDOW_SIZE];
                last_r0 = rs[0];
                last_p = ps[0];
                last_p0 = ps[0];
                exit(0);
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
        // 删除一些丢失的特征点
        last_r = rs[WINDOW_SIZE];
        last_r0 = rs[0];
        last_p = ps[0];
        last_p0 = ps[0];
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

void SlideWindowEstimator::solveOdometry()
{
}

void SlideWindowEstimator::slideWindow()
{
}