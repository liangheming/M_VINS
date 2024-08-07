#pragma once
#include "../commons.h"
#include <ceres/ceres.h>
#include <sophus/so3.hpp>

class PoseParameterization : public ceres::LocalParameterization
{

    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};