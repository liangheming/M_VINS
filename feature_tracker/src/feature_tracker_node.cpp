#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/ChannelFloat32.h>
#include <geometry_msgs/Point32.h>
#include <cv_bridge/cv_bridge.h>
#include "camera_model/pinhole_camera.h"
#include "feature_tracker.h"

struct NodeParams
{
    std::string img_topic;
    std::string config_file;
    double freq;
};
struct NodeState
{
    bool first_image_flag = true;
    bool publish_frame = false;
    bool first_pub_flag = true;
    int pub_count = 1;
    double first_image_time = 0.0;
    double last_image_time = 0.0;
};

class FeatureTrackerNode
{
public:
    FeatureTrackerNode() : m_nh("~")
    {
        ROS_INFO("Feature tacker node start");
        loadParamsAndInit();
        initSubscribers();
        initPublishers();
    }

    void loadParamsAndInit()
    {
        m_nh.param<std::string>("img_topic", m_params.img_topic, "/camera");
        m_nh.param<std::string>("config_file", m_params.config_file, "nothing");
        m_nh.param<double>("freq", m_params.freq, 10.0);
        PinholeParams pinhole_params;
        TrackerConfig tracker_config;
        m_tracker = std::make_shared<FeatureTracker>(tracker_config, std::make_shared<PinholeCamera>(pinhole_params));
    }

    void initSubscribers()
    {
        m_img_sub = m_nh.subscribe(m_params.img_topic, 100, &FeatureTrackerNode::imgCB, this);
    }

    void initPublishers()
    {
        m_feature_pts_pub = m_nh.advertise<sensor_msgs::PointCloud>("feature", 100);
        m_feature_img_pub = m_nh.advertise<sensor_msgs::Image>("feature_img", 100);
    }

    void imgCB(const sensor_msgs::ImageConstPtr &img_msg)
    {
        if (m_state.first_image_flag)
        {
            m_state.first_image_flag = false;
            m_state.first_image_time = img_msg->header.stamp.toSec();
            m_state.last_image_time = img_msg->header.stamp.toSec();
            return;
        }
        if (img_msg->header.stamp.toSec() - m_state.last_image_time > 1.0 || img_msg->header.stamp.toSec() < m_state.last_image_time)
        {
            ROS_WARN("Image discarded, time difference is too large");
            m_state.first_image_flag = true;
            m_state.first_image_time = 0.0;
            m_state.last_image_time = 0.0;
            m_state.pub_count = 1;
            return;
        }
        m_state.last_image_time = img_msg->header.stamp.toSec();

        if (round(1.0 * m_state.pub_count / (img_msg->header.stamp.toSec() - m_state.first_image_time)) <= m_params.freq)
        {
            m_state.publish_frame = true;
            if (abs(1.0 * m_state.pub_count / (img_msg->header.stamp.toSec() - m_state.first_image_time) - m_params.freq) < 0.01 * m_params.freq)
            {
                m_state.first_image_time = img_msg->header.stamp.toSec();
                m_state.pub_count = 0;
            }
        }
        else
        {
            m_state.publish_frame = false;
        }
        cv_bridge::CvImageConstPtr ptr;

        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

        cv::Mat img = ptr->image;

        m_tracker->track(img, img_msg->header.stamp.toSec(), m_state.publish_frame);

        if (m_state.publish_frame)
        {

            m_state.pub_count++;
            publishFeaturePoints(img_msg);
            publishFeatureImage(img_msg->header, img);
        }
    }

    void publishFeaturePoints(const sensor_msgs::ImageConstPtr &img_msg)
    {
        if (m_state.first_pub_flag)
            m_state.first_pub_flag = false;
        else
        {
            if (m_feature_pts_pub.getNumSubscribers() == 0)
                return;
            sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
            sensor_msgs::ChannelFloat32 id_of_point;
            sensor_msgs::ChannelFloat32 u_of_point;
            sensor_msgs::ChannelFloat32 v_of_point;
            sensor_msgs::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::ChannelFloat32 velocity_y_of_point;
            feature_points->header = img_msg->header;
            feature_points->header.frame_id = "world";
            for (size_t j = 0; j < m_tracker->ids().size(); j++)
            {
                const int &id = m_tracker->ids()[j];
                const cv::Point2f &xy = m_tracker->curPtsXY()[j];
                const cv::Point2f &uv = m_tracker->curPtsUV()[j];
                const cv::Point2f &velocity = m_tracker->ptsVelocity()[j];
                geometry_msgs::Point32 p;
                p.x = xy.x;
                p.y = xy.y;
                p.z = 1.0;
                feature_points->points.push_back(p);
                id_of_point.values.push_back(id);
                u_of_point.values.push_back(uv.x);
                v_of_point.values.push_back(uv.y);
                velocity_x_of_point.values.push_back(velocity.x);
                velocity_y_of_point.values.push_back(velocity.y);
            }
            feature_points->channels.push_back(id_of_point);
            feature_points->channels.push_back(u_of_point);
            feature_points->channels.push_back(v_of_point);
            feature_points->channels.push_back(velocity_x_of_point);
            feature_points->channels.push_back(velocity_y_of_point);
            m_feature_pts_pub.publish(feature_points);
        }
    }

    void publishFeatureImage(const std_msgs::Header &header, cv::Mat &img)
    {
        if (m_feature_img_pub.getNumSubscribers() == 0)
            return;
        cv::Mat show_img;
        cv::cvtColor(img, show_img, CV_GRAY2RGB);
        for (size_t j = 0; j < m_tracker->curPtsUV().size(); j++)
        {
            double len = std::min(1.0, 1.0 * m_tracker->trackCnt()[j] / m_tracker->config().window_size);
            cv::circle(show_img, m_tracker->curPtsUV()[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

        cv_ptr->image = show_img;
        cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
        sensor_msgs::Image img_msg;
        cv_ptr->toImageMsg(img_msg);
        img_msg.header = header;
        m_feature_img_pub.publish(img_msg);
    }

private:
    NodeParams m_params;
    NodeState m_state;
    ros::NodeHandle m_nh;
    ros::Subscriber m_img_sub;
    ros::Publisher m_feature_pts_pub;
    ros::Publisher m_feature_img_pub;
    std::shared_ptr<FeatureTracker> m_tracker;
};
int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    FeatureTrackerNode node;
    ros::spin();
    return 0;
}