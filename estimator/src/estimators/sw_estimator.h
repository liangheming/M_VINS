#pragma once
#include "commons.h"
#include "integration.h"
#include "initial_alignment.h"
#include "feature_manager.h"
#include "initial_sfm.h"
#include "pose_parameterization.h"
#include "imu_factor.h"
#include "projection_factor.h"

enum SWMarginFlag
{
    MARGIN_OLD,
    MARGIN_SECOND_NEW
};
enum SolveFlag
{
    INITIAL,
    NON_LINEAR
};

struct SWConfig
{
    int max_feature = 1000;
    bool estimate_td = false;
    bool estimate_ext = false;
};
struct SWState
{
    bool first_imu = true;
    Vec3d acc_0 = Vec3d::Zero();
    Vec3d gyro_0 = Vec3d::Zero();
    int frame_count = 0;
    double initial_timestamp = 0.0;
};

class SlideWindowEstimator
{
public:
    SlideWindowEstimator(const SWConfig &config);

    void clearState();
    void setConstGravity(const double &g_const) { g = Vec3d(0, 0, -g_const); }

    void processImu(const double &dt, const Vec3d &acc, const Vec3d &gyro);

    void processFeature(const Feats &feats, double timestamp);

    bool relativePose(Mat3d &relative_r, Vec3d &relative_t, int &l);

    bool visualInitialAlign();

    bool initialStructure();

    void vector2double();

    void double2vector();

    void optimization();

    void solveOdometry();

    void slideWindowNew();

    void slideWindowOld();

    void slideWindow();

    SWMarginFlag marginalization_flag;
    SolveFlag solve_flag;

    Vec3d g;
    double time_delay;
    Vec3d ps[WINDOW_SIZE + 1];
    Vec3d vs[WINDOW_SIZE + 1];
    Mat3d rs[WINDOW_SIZE + 1];
    Vec3d bas[WINDOW_SIZE + 1];
    Vec3d bgs[WINDOW_SIZE + 1];

    Mat3d back_r0, last_r, last_r0, r_ic;
    Vec3d back_p0, last_p, last_p0, t_ic;

    double timestamps[WINDOW_SIZE + 1];

    double para_pose[WINDOW_SIZE + 1][7];
    double para_speed_bias[WINDOW_SIZE + 1][9];
    double para_features[1000][1];
    double para_ex_pose[7];
    std::vector<std::shared_ptr<Integration>> integrations;
    std::shared_ptr<Integration> temp_integration;

    std::vector<double> dt_buf[WINDOW_SIZE + 1];
    std::vector<Vec3d> linear_acceleration_buf[WINDOW_SIZE + 1];
    std::vector<Vec3d> angular_velocity_buf[WINDOW_SIZE + 1];
    std::map<double, ImageFrame> all_image_frame;
    FeatureManager feature_manager;

private:
    SWConfig m_config;
    SWState m_state;
};