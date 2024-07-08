#include <mutex>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>

#include <camera_model/pinhole_camera.h>
#include "pose_graph/pose_graph_4dof.h"

struct NodeConfig
{
    std::string img_topic = "/cam0/image_raw";
    std::string key_odom_topic = "/vins_estimator/keyframe_odom";
    std::string key_point_topic = "/vins_estimator/keyframe_points";
    std::string patern_file = "/home/zhouzhou/vs_projects/ws_vins/src/pose_graph/config/brief_pattern.yml";
    std::string vocabulary_file = "/home/zhouzhou/vs_projects/ws_vins/src/pose_graph/config/brief_k10L6.bin";
    int skip_first = 10;
    int skip = 0;
    double skip_dis = 0.0;
};

struct NodeState
{
    std::mutex buf_mutex;
    double last_img_time = 0.0;
    double last_key_odom_time = 0.0;
    double last_key_point_time = 0.0;

    std::queue<sensor_msgs::ImageConstPtr> img_buf;
    std::queue<nav_msgs::OdometryConstPtr> key_odom_buf;
    std::queue<sensor_msgs::PointCloudConstPtr> key_point_buf;

    int skip_first_cnt = 0;
    int skip_cnt = 0;

    Vec3d last_t = Vec3d(-100.0, -100.0, -100.0);
};

class PoseGraphNode
{
public:
    PoseGraphNode() : m_nh("~")
    {
        loadParams();
        initSubsribers();
        initPublishers();
        KeyFrame::r_ic << 0.0148655429818, -0.999880929698, 0.00414029679422, 0.999557249008, 0.0149672133247, 0.025715529948, -0.0257744366974, 0.00375618835797, 0.999660727178;
        KeyFrame::t_ic << -0.0216401454975, -0.064676986768, 0.00981073058949;
        KeyFrame::initializeCamera(m_camera_params);
        KeyFrame::initializeExtractor(m_config.patern_file);
        m_pose_graph = std::make_shared<PoseGraph4DOF>(m_pose_graph_config);
        m_pose_graph->loadVocabulary(m_config.vocabulary_file);
        m_detector_timer = m_nh.createTimer(ros::Duration(0.02), &PoseGraphNode::detectorTimerCallback, this);
        m_optimizer_timer = m_nh.createTimer(ros::Duration(2), &PoseGraphNode::optimizerTimerCallback, this);
    }
    void loadParams()
    {
        ROS_INFO("[PoseGraph] LOAD CONFIG");
    }
    void initSubsribers()
    {
        m_img_sub = m_nh.subscribe(m_config.img_topic, 100, &PoseGraphNode::imgCallback, this);
        m_key_odom_sub = m_nh.subscribe(m_config.key_odom_topic, 100, &PoseGraphNode::keyOdomCallback, this);
        m_key_point_sub = m_nh.subscribe(m_config.key_point_topic, 100, &PoseGraphNode::keyPointCallback, this);
    }
    void initPublishers()
    {
    }

    void imgCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        double cur_time = msg->header.stamp.toSec();
        if (cur_time < m_state.last_img_time)
        {
            ROS_ERROR("imgCallback: img time goes back in time");
            assert(false);
            return;
        }
        m_state.last_img_time = cur_time;
        m_state.buf_mutex.lock();
        m_state.img_buf.push(msg);
        m_state.buf_mutex.unlock();
    }
    void keyOdomCallback(const nav_msgs::OdometryConstPtr &msg)
    {
        double cur_time = msg->header.stamp.toSec();
        if (cur_time < m_state.last_key_odom_time)
        {
            ROS_ERROR("keyOdomCallback: key odom time goes back in time");
            assert(false);
            return;
        }
        m_state.last_key_odom_time = cur_time;
        m_state.buf_mutex.lock();
        m_state.key_odom_buf.push(msg);
        m_state.buf_mutex.unlock();
    }
    void keyPointCallback(const sensor_msgs::PointCloudConstPtr &msg)
    {
        double cur_time = msg->header.stamp.toSec();
        if (cur_time < m_state.last_key_point_time)
        {
            ROS_ERROR("keyPointCallback: key point time goes back in time");
            assert(false);
            return;
        }
        m_state.last_key_point_time = cur_time;
        m_state.buf_mutex.lock();
        m_state.key_point_buf.push(msg);
        m_state.buf_mutex.unlock();
    }

    void detectorTimerCallback(const ros::TimerEvent &event)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;
        {
            std::lock_guard<std::mutex> lock(m_state.buf_mutex);
            if (m_state.img_buf.empty() || m_state.key_point_buf.empty() || m_state.key_odom_buf.empty())
                return;

            if (m_state.img_buf.front()->header.stamp.toSec() > m_state.key_odom_buf.front()->header.stamp.toSec())
            {
                m_state.key_odom_buf.pop();
                ROS_WARN("throw pose at beginning");
                return;
            }
            if (m_state.img_buf.front()->header.stamp.toSec() > m_state.key_point_buf.front()->header.stamp.toSec())
            {
                m_state.key_point_buf.pop();
                ROS_WARN("throw point at beginning");
                return;
            }
            if (m_state.img_buf.back()->header.stamp.toSec() >= m_state.key_odom_buf.front()->header.stamp.toSec() && m_state.key_point_buf.back()->header.stamp.toSec() >= m_state.key_odom_buf.front()->header.stamp.toSec())
            {
                pose_msg = m_state.key_odom_buf.front();
                m_state.key_odom_buf.pop();
                while (!m_state.key_odom_buf.empty())
                    m_state.key_odom_buf.pop();
                while (m_state.img_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    m_state.img_buf.pop();
                image_msg = m_state.img_buf.front();
                m_state.img_buf.pop();

                while (m_state.key_point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    m_state.key_point_buf.pop();
                point_msg = m_state.key_point_buf.front();
                m_state.key_point_buf.pop();
            }
        }

        if (pose_msg == NULL)
            return;
        if (m_state.skip_first_cnt < m_config.skip_first)
        {
            m_state.skip_first_cnt++;
            return;
        }
        if (m_state.skip_cnt < m_config.skip)
        {
            m_state.skip_cnt++;
            return;
        }
        else
        {
            m_state.skip_cnt = 0;
        }

        Vec3d vio_t = Vec3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
        Mat3d vio_r = Quatd(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x, pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z).toRotationMatrix();

        if ((vio_t - m_state.last_t).norm() < m_config.skip_dis)
            return;
        cv::Mat image = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8)->image;

        std::vector<cv::Point3f> point_3d;
        std::vector<cv::Point2f> point_2d_uv;
        std::vector<cv::Point2f> point_2d_normal;
        std::vector<int> point_id;

        for (unsigned int i = 0; i < point_msg->points.size(); i++)
        {
            cv::Point3f p_3d;
            p_3d.x = point_msg->points[i].x;
            p_3d.y = point_msg->points[i].y;
            p_3d.z = point_msg->points[i].z;
            point_3d.push_back(p_3d);

            cv::Point2f p_2d_uv, p_2d_normal;
            int p_id;
            p_2d_normal.x = point_msg->channels[i].values[0];
            p_2d_normal.y = point_msg->channels[i].values[1];
            p_2d_uv.x = point_msg->channels[i].values[2];
            p_2d_uv.y = point_msg->channels[i].values[3];
            p_id = static_cast<int>(point_msg->channels[i].values[4]);
            point_2d_normal.push_back(p_2d_normal);
            point_2d_uv.push_back(p_2d_uv);
            point_id.push_back(p_id);
        }

        std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(pose_msg->header.stamp.toSec(), vio_r, vio_t, point_id, point_3d, point_2d_uv, point_2d_normal, image);

        m_pose_graph->addKeyFrame(keyframe, true);
        m_state.last_t = vio_t;

        ROS_INFO("add keyframe %lu", m_pose_graph->key_frames.size());

        if (m_pose_graph->key_frames.back()->has_loop)
        {
            ROS_WARN("loop detected loop index: %d cur index: %d !", m_pose_graph->key_frames.back()->loop_index, m_pose_graph->key_frames.back()->index);
        }
    }
    void optimizerTimerCallback(const ros::TimerEvent &event)
    {
        if (m_pose_graph->optimize4DoF())
        {
            ROS_WARN("optimize4DoF");
        }
    }

private:
    ros::NodeHandle m_nh;
    NodeConfig m_config;
    NodeState m_state;
    PinholeParams m_camera_params;
    PoseGraphConfig m_pose_graph_config;
    std::shared_ptr<PoseGraph4DOF> m_pose_graph;
    ros::Subscriber m_img_sub;
    ros::Subscriber m_key_odom_sub;
    ros::Subscriber m_key_point_sub;

    ros::Timer m_detector_timer;
    ros::Timer m_optimizer_timer;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_graph_node");
    PoseGraphNode pose_graph_node;
    ros::spin();
    ros::shutdown();
    return 0;
}