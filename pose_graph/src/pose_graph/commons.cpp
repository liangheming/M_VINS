#include "commons.h"

Vec3d rot2ypr(const Eigen::Matrix3d &R)
{
    Vec3d n = R.col(0);
    Vec3d o = R.col(1);
    Vec3d a = R.col(2);

    Vec3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}
Mat3d ypr2rot(const Vec3d &ypr)
{

    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Mat3d Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Mat3d Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Mat3d Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}