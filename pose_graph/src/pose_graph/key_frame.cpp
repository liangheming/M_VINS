#include "key_frame.h"

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
    m_brief.compute(im, keys, descriptors);
}
BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
        throw string("Could not open file ") + pattern_file;

    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}
std::shared_ptr<BriefExtractor> KeyFrame::extractor_ptr = nullptr;
std::shared_ptr<PinholeCamera> KeyFrame::camera_ptr = nullptr;
Mat3d KeyFrame::r_ic = Mat3d::Identity();
Vec3d KeyFrame::t_ic = Vec3d::Zero();
int KeyFrame::min_loop_num = 25;

bool KeyFrame::save_image = false;
void KeyFrame::initializeExtractor(std::string &pattern_file)
{
    extractor_ptr = std::make_shared<BriefExtractor>(pattern_file);
}
void KeyFrame::initializeCamera(const PinholeParams &camera_config)
{
    camera_ptr = std::make_shared<PinholeCamera>(camera_config);
}
KeyFrame::KeyFrame(double _timestamp, Mat3d &rotation, Vec3d &translation, std::vector<int> &_points_id, std::vector<cv::Point3f> &_points_3d, std::vector<cv::Point2f> &_points_2d_uv, std::vector<cv::Point2f> &_points_2d_norm, cv::Mat &_image)
{
    time_stamp = _timestamp;
    local_rotation = rotation;
    local_translation = translation;
    points_id = _points_id;
    points_3d = _points_3d;
    points_2d_uv = _points_2d_uv;
    points_2d_norm = _points_2d_norm;
    image = _image.clone();
    has_loop = false;
    index = -1;
    loop_index = -1;
    local_index = -1;
    loop_rotation.setIdentity();
    loop_translation.setZero();
    loop_yaw_deg = 0.0;

    computeWindowBRIEFPoint();
    computeBRIEFPoint();

    if (!save_image)
        image.release();
}

void KeyFrame::computeWindowBRIEFPoint()
{
    assert(KeyFrame::extractor_ptr != nullptr);
    for (int i = 0; i < (int)points_2d_uv.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = points_2d_uv[i];
        window_keypoints.push_back(key);
    }
    (*extractor_ptr)(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint()
{
    assert(KeyFrame::camera_ptr != nullptr && KeyFrame::extractor_ptr != nullptr);
    cv::FAST(image, keypoints, 20, true);
    (*extractor_ptr)(image, keypoints, brief_descriptors);
    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        camera_ptr->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }
}
int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}
bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for (int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if (dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    if (bestIndex != -1 && bestDist < 80)
    {
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        return true;
    }
    else
        return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for (int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
            status.push_back(1);
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}
void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &pnp_t_old, Eigen::Matrix3d &pnp_r_old)
{
    status.clear();
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Mat3d r_inital;
    Vec3d p_inital;
    Mat3d r_w_c = local_rotation * r_ic;
    Vec3d t_w_c = local_translation + local_rotation * t_ic;

    r_inital = r_w_c.inverse();
    p_inital = -(r_inital * t_w_c);

    cv::eigen2cv(r_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(p_inital, t);

    cv::Mat inliers;
    cv::solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);
    for (int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Mat3d r_pnp, r_w_c_old;
    cv::cv2eigen(r, r_pnp);
    r_w_c_old = r_pnp.transpose();
    Vec3d t_pnp, t_w_c_old;
    cv::cv2eigen(t, t_pnp);
    t_w_c_old = r_w_c_old * (-t_pnp);

    pnp_r_old = r_w_c_old * r_ic.transpose();
    pnp_t_old = t_w_c_old - pnp_r_old * t_ic;
}
bool KeyFrame::findConnection(std::shared_ptr<KeyFrame> &old_kf)
{
    std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    std::vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    std::vector<cv::Point3f> matched_3d;
    std::vector<int> matched_id;
    std::vector<uchar> status;

    matched_3d = points_3d;
    matched_2d_cur = points_2d_uv;
    matched_2d_cur_norm = points_2d_norm;
    matched_id = points_id;

    searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);

    Vec3d pnp_t_old;
    Mat3d pnp_r_old;
    Vec3d relative_t;
    Mat3d relative_r;
    double relative_yaw;
    if ((int)matched_2d_cur.size() > KeyFrame::min_loop_num)
    {
        status.clear();
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, pnp_t_old, pnp_r_old);
        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_2d_cur_norm, status);
        reduceVector(matched_2d_old_norm, status);
        reduceVector(matched_3d, status);
        reduceVector(matched_id, status);
    }

    if ((int)matched_2d_cur.size() > KeyFrame::min_loop_num)
    {
        relative_t = pnp_r_old.transpose() * (local_translation - pnp_t_old);
        relative_r = pnp_r_old.transpose() * local_rotation;
        relative_yaw = normalizeAngle(rot2ypr(local_rotation).x() - rot2ypr(pnp_r_old).x());
        if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
        {
            has_loop = true;
            loop_index = old_kf->index;
            loop_rotation = relative_r;
            loop_translation = relative_t;
            loop_yaw_deg = relative_yaw;
            return true;
        }
    }
    return false;
}
