#pragma once
#include "commons.h"
#include "integration.h"

class ImageFrame
{
public:
    ImageFrame() = default;
    ImageFrame(const TrackedFeatures &_feats, double _t) : timestamp(_t), is_keyframe(false) { feats = _feats; }

    TrackedFeatures feats;
    double timestamp;
    Mat3d r;
    Vec3d t;
    std::shared_ptr<Integration> integration;
    bool is_keyframe;
};

Eigen::MatrixXd TangentBasis(Vec3d &g0);

void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Vec3d *bgs);

bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d &gravity, Vec3d &tic, Eigen::VectorXd &xs);

void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Vec3d &gravity, Vec3d &tic, double g_norm, Eigen::VectorXd &xs);

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d *bgs, Vec3d &gravity, Vec3d &tic, Eigen::VectorXd &xs);
