#include "pinhole_camera.h"

PinholeCamera::PinholeCamera(const PinholeParams &params) : m_params(params),
                                                            m_distortion(params.k1 != 0.0),
                                                            m_inv_K11(1.0 / params.fx),
                                                            m_inv_K13(-params.cx / params.fx),
                                                            m_inv_K22(1.0 / params.fy),
                                                            m_inv_K23(-params.cy / params.fy)
{
}

void PinholeCamera::distortion(const Eigen::Vector2d &p_u, Eigen::Vector2d &d_u) const
{
    double k1 = m_params.k1;
    double k2 = m_params.k2;
    double p1 = m_params.p1;
    double p2 = m_params.p2;

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
        p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

void PinholeCamera::liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const
{
    double mx_d, my_d, mx_u, my_u;
    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;
    if (m_distortion)
    {
        int n = 8;
        Eigen::Vector2d d_u;
        distortion(Eigen::Vector2d(mx_d, my_d), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
        for (int i = 1; i < n; ++i)
        {
            distortion(Eigen::Vector2d(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }
    else
    {
        mx_u = mx_d;
        my_u = my_d;
    }

    P << mx_u, my_u, 1.0;
}
