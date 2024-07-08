#include "pose_graph_4dof.h"

PoseGraph4DOF::PoseGraph4DOF(const PoseGraphConfig &_config) : config(_config)
{
    drift_rotation = Mat3d::Identity();
    drift_translation = Vec3d::Zero();
    global_index = 0;
    earliest_loop_index = -1;
    key_frames.clear();
    std::queue<int>().swap(optimize_buf);
}

int PoseGraph4DOF::detectLoop(std::shared_ptr<KeyFrame> keyframe, int frame_index)
{
    QueryResults ret;

    db.query(keyframe->brief_descriptors, ret, config.loop_frame_search_cnt, frame_index - config.min_frame_gap_cnt);
    db.add(keyframe->brief_descriptors);
    bool find_loop = false;

    if (ret.size() >= 1 && ret[0].Score > config.score_threshold)
    {
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            if (ret[i].Score > config.second_score_threshold)
            {
                find_loop = true;
            }
        }
    }

    if (find_loop && frame_index > config.min_frame_gap_cnt)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > config.second_score_threshold))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;
}

std::shared_ptr<KeyFrame> PoseGraph4DOF::getKeyFrame(int index)
{
    std::list<std::shared_ptr<KeyFrame>>::iterator it = key_frames.begin();
    for (; it != key_frames.end(); it++)
    {
        if ((*it)->index == index)
            break;
    }
    if (it != key_frames.end())
        return *it;
    else
        return nullptr;
}
void PoseGraph4DOF::addKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop)
{
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop)
        loop_index = detectLoop(cur_kf, cur_kf->index);
    if (loop_index != -1)
    {
        std::shared_ptr<KeyFrame> old_kf = getKeyFrame(loop_index);
        if (cur_kf->findConnection(old_kf))
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;
            m_loop_buf_mutex.lock();
            optimize_buf.push(cur_kf->index);
            m_loop_buf_mutex.unlock();
        }
    }
    m_key_frames_mutex.lock();
    cur_kf->global_rotation = drift_rotation * cur_kf->local_rotation;
    cur_kf->global_translation = drift_rotation * cur_kf->local_translation + drift_translation;
    key_frames.push_back(cur_kf);
    m_key_frames_mutex.unlock();
}

bool PoseGraph4DOF::optimize4DoF()
{
    int cur_index = -1;
    int first_looped_index = -1;
    m_loop_buf_mutex.lock();
    while (!optimize_buf.empty())
    {
        cur_index = optimize_buf.front();
        first_looped_index = earliest_loop_index;
        optimize_buf.pop();
    }
    m_loop_buf_mutex.unlock();

    if (cur_index == -1)
        return false;

    int max_length = cur_index + 1;
    double t_array[max_length][3];
    Quatd q_array[max_length];
    double euler_array[max_length][3];

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();
    list<std::shared_ptr<KeyFrame>>::iterator it;

    m_key_frames_mutex.lock();
    std::shared_ptr<KeyFrame> cur_kf = getKeyFrame(cur_index);
    int i = 0;

    for (it = key_frames.begin(); it != key_frames.end(); it++)
    {
        if ((*it)->index < first_looped_index)
            continue;
        (*it)->local_index = i;
        Mat3d tmp_r = (*it)->local_rotation;
        Vec3d tmp_t = (*it)->local_translation;
        Quatd tmp_q(tmp_r);
        t_array[i][0] = tmp_t(0);
        t_array[i][1] = tmp_t(1);
        t_array[i][2] = tmp_t(2);
        q_array[i] = tmp_q;
        Vec3d euler_angle = rot2ypr(tmp_q.toRotationMatrix());
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();
        problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
        problem.AddParameterBlock(t_array[i], 3);
        if ((*it)->index == first_looped_index)
        {
            problem.SetParameterBlockConstant(euler_array[i]);
            problem.SetParameterBlockConstant(t_array[i]);
        }

        for (int j = 1; j < 5; j++)
        {
            if (i - j >= 0)
            {
                Vec3d euler_conncected = rot2ypr(q_array[i - j].toRotationMatrix());
                Vec3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
                relative_t = q_array[i - j].inverse() * relative_t;
                double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                ceres::CostFunction *cost_function = FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw, euler_conncected.y(), euler_conncected.z());
                problem.AddResidualBlock(cost_function, NULL, euler_array[i - j], t_array[i - j], euler_array[i], t_array[i]);
            }
        }

        if ((*it)->has_loop)
        {
            assert((*it)->loop_index >= first_looped_index);
            int connected_index = getKeyFrame((*it)->loop_index)->local_index;
            Vec3d euler_conncected = rot2ypr(q_array[connected_index].toRotationMatrix());
            Vec3d relative_t = (*it)->loop_translation;
            double relative_yaw = (*it)->loop_yaw_deg;

            ceres::CostFunction *cost_function = FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                            relative_yaw, euler_conncected.y(), euler_conncected.z());
            problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                     t_array[connected_index],
                                     euler_array[i],
                                     t_array[i]);
        }
        if ((*it)->index == cur_index)
            break;
        i++;
    }
    m_key_frames_mutex.unlock();

    ceres::Solve(options, &problem, &summary);

    m_key_frames_mutex.lock();
    i = 0;
    for (it = key_frames.begin(); it != key_frames.end(); it++)
    {
        if ((*it)->index < first_looped_index)
            continue;
        Quatd tmp_q(ypr2rot(Vec3d(euler_array[i][0], euler_array[i][1], euler_array[i][2])));
        (*it)->global_translation = Vec3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        (*it)->global_rotation = tmp_q.toRotationMatrix();
        if ((*it)->index == cur_index)
            break;
        i++;
    }

    double yaw_drift = rot2ypr(cur_kf->global_rotation).x() - rot2ypr(cur_kf->local_rotation).x();
    drift_rotation = ypr2rot(Vec3d(yaw_drift, 0, 0));
    drift_translation = cur_kf->global_translation - drift_rotation * cur_kf->local_translation;
    it++;
    for (; it != key_frames.end(); it++)
    {

        (*it)->global_translation = drift_rotation * (*it)->local_translation + drift_translation;
        (*it)->global_rotation = drift_rotation * (*it)->local_rotation;
    }
    m_key_frames_mutex.unlock();

    return true;
}

void PoseGraph4DOF::loadVocabulary(std::string &voc_path)
{
    voc = std::make_shared<BriefVocabulary>(voc_path);
    db.setVocabulary(*voc, false, 0);
}