#include "feature_manager.h"

FeatureManager::FeatureManager(Mat3d _rs[]) : rs(_rs)
{
    r_ic.setIdentity();
    features.clear();
}
double FeatureManager::compensatedParallax2(const FeaturePerID &it_per_id, int frame_count)
{
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];
    Vec3d p_j = frame_j.point;
    Vec3d p_i = frame_i.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    double u_i = p_i(0);
    double v_i = p_i(1);

    double du = u_i - u_j, dv = v_i - v_j;

    return std::sqrt(du * du + dv * dv);
}
bool FeatureManager::addFeatureCheckParallax(int frame_count, const Feats &feats, double td)
{
    int last_track_num = 0, parallax_num = 0;
    double parallax_sum = 0.0;
    // std::cout << "feat_size:  " << feats.size() << std::endl;
    for (auto &feat : feats)
    {
        // std::cout << frame_count << " " << feat.first << " " << feat.second.transpose() << std::endl;
        FeaturePerFrame f_per_frame(feat.second, td);
        int feature_id = feat.first;
        auto it = std::find_if(features.begin(), features.end(), [feature_id](const FeaturePerID &it)
                               { return it.feature_id == feature_id; });
        if (it == features.end())
        {
            features.push_back(FeaturePerID(feature_id, frame_count));
            features.back().feature_per_frame.push_back(f_per_frame);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_frame);
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : features)
    {
        if (it_per_id.start_frame <= frame_count - 2 && it_per_id.endFrame() >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    if (parallax_num == 0)
        return true;
    else
        return (parallax_sum / parallax_num) >= parallax_threshold;
}

void FeatureManager::getCorresponding(int frame_count_l, int frame_count_r, std::vector<std::pair<Vec3d, Vec3d>> &corres)
{
    corres.clear();
    for (auto &it : features)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vec3d a = Vec3d::Zero(), b = Vec3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            corres.emplace_back(a, b);
        }
    }
}
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : features)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}
Eigen::VectorXd FeatureManager::getDepthVector()
{
    Eigen::VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::setDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}
void FeatureManager::clearDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

void FeatureManager::triangulate(Mat3d rs[], Vec3d ps[], const Mat3d &ric, const Vec3d &tic)
{
    for (auto &it_per_id : features)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;
        Eigen::Matrix<double, 3, 4> p0;
        Eigen::Vector3d t0 = ps[imu_i] + rs[imu_i] * tic;
        Eigen::Matrix3d r0 = rs[imu_i] * ric;
        p0.leftCols<3>() = Eigen::Matrix3d::Identity();
        p0.rightCols<1>() = Eigen::Vector3d::Zero();
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = ps[imu_j] + rs[imu_j] * tic;
            Eigen::Matrix3d r1 = rs[imu_j] * ric;
            // 在第一帧的坐标系下算
            Eigen::Vector3d t = r0.transpose() * (t1 - t0);
            Eigen::Matrix3d r = r0.transpose() * r1;
            Eigen::Matrix<double, 3, 4> p;
            p.leftCols<3>() = r.transpose();
            p.rightCols<1>() = -r.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * p.row(2) - f[2] * p.row(0);
            svd_A.row(svd_idx++) = f[1] * p.row(2) - f[2] * p.row(1);
        }
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        it_per_id.estimated_depth = svd_method;
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = 5.0;
        }
    }
}

void FeatureManager::removeBackShiftDepth(const Mat3d &marg_R, const Vec3d &marg_P, const Mat3d &new_R, const Vec3d &new_P)
{
    for (auto it = features.begin(); it != features.end(); it++)
    {
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Vec3d uv_i = it->feature_per_frame[0].point;
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                features.erase(it);
                continue;
            }
            else
            {
                Vec3d pts_i = uv_i * it->estimated_depth;
                Vec3d w_pts_i = marg_R * pts_i + marg_P;
                Vec3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = 5.0;
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = features.begin(); it != features.end(); it++)
    {
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0) // 这边似乎也可以判定 < 2
                features.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = features.begin(); it != features.end(); it++)
    {

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                features.erase(it);
        }
    }
}