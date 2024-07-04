#pragma once
#include "../commons.h"
#include "../integration.h"
#include <ceres/ceres.h>


class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
public:
    IMUFactor() = delete;

    IMUFactor(std::shared_ptr<Integration> _integration, Vec3d _g_vec);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    std::shared_ptr<Integration> integration;
    Vec3d g_vec;
};