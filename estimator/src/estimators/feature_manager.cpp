#include "feature_manager.h"

FeatureManager::FeatureManager(Mat3d _rs[]) : rs(_rs)
{
    ric.setIdentity();
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
    for (auto &feat : feats)
    {
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