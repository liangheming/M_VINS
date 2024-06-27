#include "feature_tracker.h"
bool inBorder(const cv::Point2f &pt, int width, int height, int border)
{
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return border <= img_x && img_x < width - border && border <= img_y && img_y < height - border;
}
void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    size_t j = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

int FeatureTracker::n_id = 0;
void reduceVector(std::vector<int> &v, std::vector<uchar> status)
{
    size_t j = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
FeatureTracker::FeatureTracker(const TrackerConfig &config, std::shared_ptr<PinholeCamera> camera) : m_config(config), m_camera(camera)
{
}
void FeatureTracker::rejectWithF()
{
    if (m_cur_pts.size() < 8)
        return;
    std::vector<cv::Point2f> un_pre_pts(m_prev_pts.size()), un_cur_pts(m_cur_pts.size());
    for (size_t i = 0; i < m_prev_pts.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        m_camera->liftProjective(Eigen::Vector2d(m_prev_pts[i].x, m_prev_pts[i].y), tmp_p);
        tmp_p.x() = m_config.focal_length * tmp_p.x() / tmp_p.z() + m_camera->params().width / 2.0;
        tmp_p.y() = m_config.focal_length * tmp_p.y() / tmp_p.z() + m_camera->params().height / 2.0;
        un_pre_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

        m_camera->liftProjective(Eigen::Vector2d(m_cur_pts[i].x, m_cur_pts[i].y), tmp_p);
        tmp_p.x() = m_config.focal_length * tmp_p.x() / tmp_p.z() + m_camera->params().width / 2.0;
        tmp_p.y() = m_config.focal_length * tmp_p.y() / tmp_p.z() + m_camera->params().height / 2.0;
        un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    std::vector<uchar> status;
    cv::findFundamentalMat(un_pre_pts, un_cur_pts, cv::FM_RANSAC, m_config.f_threshold, 0.99, status);
    reduceVector(m_prev_pts, status);
    reduceVector(m_cur_pts, status);
    reduceVector(m_ids, status);
    reduceVector(m_track_cnt, status);
}
void FeatureTracker::setMask()
{
    m_mask = cv::Mat(m_camera->params().height, m_camera->params().width, CV_8UC1, cv::Scalar(255));
    if (m_cur_pts.empty())
        return;
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;
    for (size_t i = 0; i < m_cur_pts.size(); i++)
        cnt_pts_id.push_back(std::make_pair(m_track_cnt[i], std::make_pair(m_cur_pts[i], m_ids[i])));
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
         { return a.first > b.first; });
    m_cur_pts.clear();
    m_ids.clear();
    m_track_cnt.clear();
    for (auto &it : cnt_pts_id)
    {
        if (m_mask.at<uchar>(it.second.first) == 255)
        {
            m_cur_pts.push_back(it.second.first);
            m_ids.push_back(it.second.second);
            m_track_cnt.push_back(it.first);
            cv::circle(m_mask, it.second.first, m_config.min_dist, 0, -1);
        }
    }
}
void FeatureTracker::addPoints()
{
    if (m_cache_pts.empty())
        return;
    for (auto &p : m_cache_pts)
    {
        m_cur_pts.push_back(p);
        m_ids.push_back(-1);
        m_track_cnt.push_back(1);
    }
}
void FeatureTracker::undistortedPoints()
{
    m_cur_un_pts.clear();
    m_cur_un_pts_map.clear();

    for (size_t i = 0; i < m_cur_pts.size(); i++)
    {
        Eigen::Vector2d a(m_cur_pts[i].x, m_cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        m_cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        m_cur_un_pts_map.insert(std::make_pair(m_ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    }

    if (!m_prev_un_pts_map.empty())
    {
        double dt = m_cur_time - m_prev_time;
        m_pts_velocity.clear();
        for (size_t i = 0; i < m_cur_un_pts.size(); i++)
        {
            if (m_ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = m_prev_un_pts_map.find(m_ids[i]);
                if (it != m_prev_un_pts_map.end())
                {
                    double v_x = (m_cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (m_cur_un_pts[i].y - it->second.y) / dt;
                    m_pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                {
                    // 应该不会发生
                    m_pts_velocity.push_back(cv::Point2f(0, 0));
                }
            }
            else
            {
                m_pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (size_t i = 0; i < m_cur_pts.size(); i++)
        {
            m_pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    m_prev_un_pts_map = m_cur_un_pts_map;
}
void FeatureTracker::updateID()
{
    for (size_t i = 0; i < m_ids.size(); i++)
    {
        if (m_ids[i] == -1)
            m_ids[i] = n_id++;
    }
}
void FeatureTracker::track(const cv::Mat &img, double timestamp, bool publish)
{
    m_cur_time = timestamp;
    cv::Mat temp_img;
    if (m_config.equalize)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, temp_img);
        m_cur_img = temp_img;
    }
    else
    {
        m_cur_img = img.clone();
    }

    m_cur_pts.clear();

    if (m_prev_pts.size() > 0)
    {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(m_prev_img, m_cur_img, m_prev_pts, m_cur_pts, status, err, cv::Size(21, 21), 3);

        for (size_t i = 0; i < m_cur_pts.size(); i++)
            if (status[i] && !inBorder(m_cur_pts[i], m_camera->params().width, m_camera->params().height, 1))
                status[i] = 0;
        reduceVector(m_prev_pts, status);
        reduceVector(m_cur_pts, status);
        reduceVector(m_ids, status);
        reduceVector(m_track_cnt, status);
    }

    for (auto &n : m_track_cnt)
        n++;

    if (publish)
    {

        rejectWithF();
        setMask();
        int n_max_cnt = m_config.max_count - static_cast<int>(m_cur_pts.size());
        if (n_max_cnt > 0)
            cv::goodFeaturesToTrack(m_cur_img, m_cache_pts, n_max_cnt, 0.01, m_config.min_dist, m_mask);
        else
            m_cache_pts.clear();
        addPoints();
    }
    m_prev_img = m_cur_img;
    m_prev_pts = m_cur_pts;
    undistortedPoints();
    m_prev_time = m_cur_time;

    if (publish)
        updateID();
}
