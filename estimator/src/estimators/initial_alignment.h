#pragma once
#include "commons.h"
#include "integration.h"

class ImageFrame
{
public:
    ImageFrame() = default;
    ImageFrame(const Feats &_feats, double _t) : timestamp(_t), is_keyframe(false) { feats = _feats; }

    Feats feats;
    double timestamp;
    Mat3d r;
    Vec3d t;
    std::shared_ptr<Integration> integration;
    bool is_keyframe;
};
Eigen::MatrixXd TangentBasis(Vec3d &g0);

void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Vec3d *Bgs);

bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d &g, Vec3d &t_ic, Eigen::VectorXd &x);

void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Vec3d &g, Vec3d &t_ic, double g_norm, Eigen::VectorXd &x);

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d *Bgs, Vec3d &g, Vec3d &t_ic, Eigen::VectorXd &x);
