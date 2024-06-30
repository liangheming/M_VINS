#include "pose_parameterization.h"

bool PoseParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Vec3d> _p(x);

    Eigen::Map<const Quatd> _q(x + 3);

    Eigen::Map<const Vec3d> dp(delta);

    Quatd dq = deltaQ(Eigen::Map<const Vec3d>(delta + 3));

    Eigen::Map<Vec3d> p(x_plus_delta);
    Eigen::Map<Quatd> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();
    return true;
}

bool PoseParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}
