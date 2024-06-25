#include "sw_estimator.h"

SlideWindowEstimator::SlideWindowEstimator(const SWConfig &config) : m_config(config)
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
    last_p = Vec3d::Zero();
    last_p0 = Vec3d::Zero();
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
{    // 关键帧判断
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

bool SlideWindowEstimator::initialStructure()
{
    return true;
}

void SlideWindowEstimator::solveOdometry()
{
}

void SlideWindowEstimator::slideWindow()
{
}