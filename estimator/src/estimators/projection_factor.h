#pragma once
#include "commons.h"
#include <ceres/ceres.h>

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{

public:
    ProjectionFactor(const Vec3d &_pts_i, const Vec3d &_pts_j, const Mat2d &_sqrt_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    Vec3d pts_i, pts_j;
    Mat2d sqrt_info;
};