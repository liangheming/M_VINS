#include "commons.h"
Mat3d Jr(const Vec3d &val)
{
    return Sophus::SO3d::leftJacobian(val).transpose();
}

Mat3d Jr_inv(const Vec3d &val)
{
    return Sophus::SO3d::leftJacobianInverse(val).transpose();
}