#include "commons.h"

bool solveRelativeRT(const std::vector<std::pair<Vec3d, Vec3d>> &corres, Mat3d &rotation, Vec3d &translation, const double &ransac_threshold)
{
    if (corres.size() < 15)
        return false;
    std::vector<cv::Point2f> ll, rr;
    for (int i = 0; i < int(corres.size()); i++)
    {
        ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
        rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
    }

    cv::Mat mask;
    cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, ransac_threshold, 0.99, mask);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++)
    {
        T(i) = trans.at<double>(i, 0);
        for (int j = 0; j < 3; j++)
            R(i, j) = rot.at<double>(i, j);
    }
    rotation = R.transpose();
    translation = -R.transpose() * T;
    if (inlier_cnt > 12)
        return true;
    else
        return false;
}

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

Mat3d rotFromG(const Eigen::Vector3d &g)
{
    Mat3d R0;
    Vec3d ng1 = g.normalized();
    Vec3d ng2{0, 0, 1.0};
    R0 = Quatd::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = rot2ypr(R0).x();
    R0 = ypr2rot(Vec3d{-yaw, 0, 0}) * R0;
    return R0;
}
