#include "feature_manager.h"

double FeatureManager::compensatedParallax2(const FeaturePoint &it_per_id, int frame_count)
{
    const FeatureInFrame &frame_i = it_per_id.observations[frame_count - 2 - it_per_id.start_frame];
    const FeatureInFrame &frame_j = it_per_id.observations[frame_count - 1 - it_per_id.start_frame];
    const Vec3d &p_j = frame_j.point;
    const Vec3d &p_i = frame_i.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    double u_i = p_i(0);
    double v_i = p_i(1);

    double du = u_i - u_j, dv = v_i - v_j;

    return std::sqrt(du * du + dv * dv);
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const TrackedFeatures &feats, double td)
{
    int last_track_num = 0, parallax_num = 0;
    double parallax_sum = 0.0;

    for (auto &feat : feats)
    {
        FeatureInFrame f_per_frame(feat.second, td);
        int feature_id = feat.first;
        auto it = std::find_if(features.begin(), features.end(), [feature_id](const FeaturePoint &it)
                               { return it.feature_id == feature_id; });
        if (it == features.end())
        {
            features.push_back(FeaturePoint(feature_id, frame_count));
            features.back().observations.push_back(f_per_frame);
        }
        else if (it->feature_id == feature_id)
        {
            it->observations.push_back(f_per_frame);
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < m_min_track_count)
        return true;

    for (auto &it_per_id : features)
    {
        // 这里主要判断次新帧是不是关键帧
        if (it_per_id.start_frame <= frame_count - 2 && it_per_id.endFrame() >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }
    if (parallax_num == 0)
        return true;
    else
        return (parallax_sum / parallax_num) >= m_parallax_threshold;
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

            a = it.observations[idx_l].point;

            b = it.observations[idx_r].point;
            corres.emplace_back(a, b);
        }
    }
}

/**
 * 将没有三角化的特侦点进行三角化
 */
void FeatureManager::triangulate(Mat3d rs[], Vec3d ps[], const Mat3d &ric, const Vec3d &tic)
{
    for (auto &it_per_id : features)
    {
        int obervation_count = static_cast<int>(it_per_id.observations.size());
        if (!(obervation_count >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // 已经三角化过的不再进行三角化
        if (it_per_id.estimated_depth > 0.0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Eigen::MatrixXd svd_A(2 * obervation_count, 4);
        int svd_idx = 0;
        Eigen::Matrix<double, 3, 4> p0;
        Eigen::Vector3d t0 = ps[imu_i] + rs[imu_i] * tic;
        Eigen::Matrix3d r0 = rs[imu_i] * ric;
        p0.leftCols<3>() = Eigen::Matrix3d::Identity();
        p0.rightCols<1>() = Eigen::Vector3d::Zero();
        for (auto &it_per_frame : it_per_id.observations)
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
        it_per_id.is_depth_valid = true;
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = 5.0;
            it_per_id.is_depth_valid = false;
        }
    }
}

/**
 * @brief 移除最老的关键帧
 */
void FeatureManager::removeBack()
{

    for (auto it = features.begin(), it_next = features.begin(); it != features.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->observations.erase(it->observations.begin());
            if (it->observations.size() < 2)
                features.erase(it);
        }
    }
}

/**
 * @brief 移除滑窗中最老帧的关键帧，同时将最老帧的深度关联到次老帧上
 */
void FeatureManager::removeBackShiftDepth(const Mat3d &marg_r, const Vec3d &marg_p, const Mat3d &new_r, const Vec3d &new_p)
{
    for (auto it = features.begin(), it_next = features.begin(); it != features.end(); it = it_next)
    {
        it_next++;
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Vec3d uv_i = it->observations[0].point;
            it->observations.erase(it->observations.begin());
            if (it->observations.size() < 2)
            {
                features.erase(it);
                continue;
            }
            else
            {
                Vec3d pts_i = uv_i * it->estimated_depth;
                Vec3d w_pts_i = marg_r * pts_i + marg_p;
                Vec3d pts_j = new_r.transpose() * (w_pts_i - new_p);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                {
                    it->estimated_depth = dep_j;
                    it->is_depth_valid = true;
                }
                else
                {
                    it->estimated_depth = 5.0;
                    it->is_depth_valid = false;
                }
            }
        }
    }
}
/**
 * @brief 移除次新帧
 * @param frame_count 最新帧所在的滑窗索引
 */
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = features.begin(), it_next = features.begin(); it != features.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
            it->start_frame--;
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->observations.erase(it->observations.begin() + j);
            if (it->observations.size() < 2)
                features.erase(it);
        }
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = features.begin(), it_next = features.begin();
         it != features.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            features.erase(it);
    }
}