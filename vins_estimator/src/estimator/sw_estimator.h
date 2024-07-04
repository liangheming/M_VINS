#pragma once
#include "commons.h"
#include "initial_sfm.h"
#include "integration.h"
#include "feature_manager.h"
#include "initial_alignment.h"

#include "factors/imu_factor.h"
#include "factors/projection_factor.h"
#include "factors/pose_parameterization.h"
#include "factors/marginalization_factor.h"

enum MarginFlag
{
    MARGIN_OLD,
    MARGIN_SECOND_NEW
};
enum SolveFlag
{
    INITIAL,
    NON_LINEAR
};
struct EstimatorConfig
{
    double g_norm = 9.81;
    bool estimate_td = false;
    bool estimate_ext = false;
    Mat3d ric;
    Vec3d tic;
    double axis_min_parallax = 30.0 / 460.0;
    double ransac_threshold = 0.3 / 460.0;
    Mat2d proj_sqrt_info = Mat2d::Identity() * 460.0 / 1.5;
};
struct EstimatorState
{
    Mat3d rs[WINDOW_SIZE + 1];
    Vec3d ps[WINDOW_SIZE + 1];
    Vec3d vs[WINDOW_SIZE + 1];
    Vec3d bas[WINDOW_SIZE + 1];
    Vec3d bgs[WINDOW_SIZE + 1];
    Mat3d ric;
    Vec3d tic;
    double td;
    Vec3d gravity;

    int frame_count;
    double initial_timestamp;
    bool first_imu;
    Vec3d acc_0;
    Vec3d gyro_0;

    std::shared_ptr<Integration> temp_integration;
    std::vector<std::shared_ptr<Integration>> integrations;

    double timestamps[WINDOW_SIZE + 1];
    std::vector<double> dt_buf[WINDOW_SIZE + 1];
    std::vector<Vec3d> linear_acceleration_buf[WINDOW_SIZE + 1];
    std::vector<Vec3d> angular_velocity_buf[WINDOW_SIZE + 1];

    Mat3d last_r, last_r0, back_r0;
    Vec3d last_p, last_p0, back_p0;

    std::shared_ptr<MarginalizationInfo> last_marginalization_info;
    std::vector<double *> last_marginalization_parameter_blocks;
};
struct EstimatorParams
{
    double pose[WINDOW_SIZE + 1][POSE_SIZE];
    double speed_bias[WINDOW_SIZE + 1][SPEEDBIAS_SIZE];
    double ext_pose[POSE_SIZE];
    double features[MAX_FEATURE_SIZE][1];
};

class SlidingWindowEstimator
{
public:
    SlidingWindowEstimator(const EstimatorConfig &m_config);

    ~SlidingWindowEstimator() { out_file->close(); }

    void reset();

    void processImu(const double &dt, const Vec3d &acc, const Vec3d &gyro);

    void processFeature(const TrackedFeatures &feats, double timestamp);

    bool initialStructure();

    void solveOdometry();

    void slideWindowNew();

    void slideWindowOld();

    void slideWindow();

    bool relativePose(Mat3d &relative_r, Vec3d &relative_t, int &l);

    bool beforeVisualInitialAlign(std::map<int, Vec3d> &sfm_tracked_points, Mat3d *rs, Vec3d *ts);

    bool visualInitialAlign(Eigen::VectorXd &xs);

    void afterVisualInitialAlign(Eigen::VectorXd &xs);

    void optimization();

    void vector2double();

    void double2vector();

    void prepareMarginInfo();

    EstimatorState &state() { return m_state; }

    MarginFlag marginalization_flag;
    SolveFlag solve_flag;
    FeatureManager feature_manager;
    std::map<double, ImageFrame> all_image_frame;

    size_t temp_count = 0;
    std::shared_ptr<std::ofstream> out_file;

private:
    EstimatorConfig m_config;
    EstimatorState m_state;
    EstimatorParams m_params;
};