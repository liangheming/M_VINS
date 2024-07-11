#pragma once
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

struct PinholeParams
{
    int width = 752;
    int height = 480;
    double fx = 4.616e+02;
    double fy = 4.603e+02;
    double cx = 3.630e+02;
    double cy = 2.481e+02;
    double k1 = -2.917e-01;
    double k2 = 8.228e-02;
    double p1 = 5.333e-05;
    double p2 = -1.578e-04;
};

class PinholeCamera
{
public:
    PinholeCamera(const PinholeParams &params);
    PinholeParams &params() { return m_params; }

    void distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u) const;
    void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const;

private:
    PinholeParams m_params;
    bool m_distortion;
    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
};