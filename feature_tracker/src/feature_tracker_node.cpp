#include <mutex>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <filesystem>

#include "kl_tracker/feature_tracker.h"

double stamp2double(builtin_interfaces::msg::Time stamp)
{
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}
builtin_interfaces::msg::Time double2stamp(double sec)
{
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(sec);
    stamp.nanosec = static_cast<uint32_t>((sec - stamp.sec) * 1e9);
    return stamp;
}

struct NodeConfig
{
    std::string img_topic = "/cam0/image_raw";
    std::string feature_points_topic = "feature";
    std::string feature_image_topic = "tracked_img";
    std::string config_file = "nothing.yaml";
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

class FeatureTrackerNode : public rclcpp::Node
{
public:
    FeatureTrackerNode() : Node("feature_tracker_node")
    {
        loadParams();
        initSubscribers();
        initPublishers();
        m_feature_tracker = std::make_shared<FeatureTracker>(m_tracker_config, std::make_shared<PinholeCamera>(m_camera_params));
    }
    void loadParams()
    {
        this->declare_parameter("img_topic", "/camera/image_raw");
        this->declare_parameter("config_file", "nothing.yaml");
        this->declare_parameter("freq", 10.0);
        this->get_parameter("img_topic", m_config.img_topic);
        this->get_parameter("config_file", m_config.config_file);
        this->get_parameter("freq", m_config.freq);
        if (std::filesystem::exists(m_config.config_file))
        {
            YAML::Node config = YAML::LoadFile(m_config.config_file);
            RCLCPP_INFO(this->get_logger(), "[FeatureTracker] Load From File: %s", m_config.config_file.c_str());
            const YAML::Node &camera_node = config["camera"];
            m_camera_params.width = camera_node["width"].as<int>();
            m_camera_params.height = camera_node["height"].as<int>();
            m_camera_params.fx = camera_node["fx"].as<double>();
            m_camera_params.fy = camera_node["fy"].as<double>();
            m_camera_params.cx = camera_node["cx"].as<double>();
            m_camera_params.cy = camera_node["cy"].as<double>();
            m_camera_params.k1 = camera_node["k1"].as<double>();
            m_camera_params.k2 = camera_node["k2"].as<double>();
            m_camera_params.p1 = camera_node["p1"].as<double>();
            m_camera_params.p2 = camera_node["p2"].as<double>();
            const YAML::Node &tracker_node = config["tracker"];
            m_tracker_config.equalize = tracker_node["equalize"].as<bool>();
            m_tracker_config.f_threshold = tracker_node["f_threshold"].as<double>();
            m_tracker_config.focal_length = tracker_node["focal_length"].as<double>();
            m_tracker_config.min_dist = tracker_node["min_dist"].as<int>();
            m_tracker_config.max_count = tracker_node["max_count"].as<int>();
            m_tracker_config.window_size = tracker_node["window_size"].as<int>();
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "[FeatureTracker] File: %s Not Exits, Load From Default", m_config.config_file.c_str());
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (m_state.first_image_flag)
        {
            m_state.first_image_flag = false;
            m_state.first_image_time = stamp2double(msg->header.stamp);
            m_state.last_image_time = m_state.first_image_time;
            return;
        }
        double cur_image_time = stamp2double(msg->header.stamp);
        if (cur_image_time - m_state.last_image_time > 1.0 || cur_image_time < m_state.last_image_time)
        {
            RCLCPP_WARN(this->get_logger(), "[FeatureTracker] Image Time Error");
            m_state.first_image_flag = true;
            m_state.first_image_time = 0.0;
            m_state.last_image_time = 0.0;
            m_state.pub_count = 1;
            return;
        }

        m_state.last_image_time = cur_image_time;

        if (round(1.0 * m_state.pub_count / (cur_image_time - m_state.first_image_time)) <= m_config.freq)
        {
            m_state.publish_frame = true;
            if (abs(1.0 * m_state.pub_count / (cur_image_time - m_state.first_image_time) - m_config.freq) < 0.01 * m_config.freq)
            {
                m_state.first_image_time = cur_image_time;
                m_state.pub_count = 0;
            }
        }
        else
        {
            m_state.publish_frame = false;
        }
        cv::Mat img = cv_bridge::toCvCopy(msg, "mono8")->image;
        m_feature_tracker->track(img, cur_image_time, m_state.publish_frame);

        if (m_state.publish_frame)
        {

            m_state.pub_count++;
            publishFeaturePoints(msg->header);
            publishFeatureImage(msg->header, img);
        }
    }

    void publishFeaturePoints(std_msgs::msg::Header &header)
    {
        if (m_state.first_pub_flag)
            m_state.first_pub_flag = false;
        else
        {
            if (m_feature_pts_pub->get_subscription_count() < 1)
                return;
            sensor_msgs::msg::PointCloud feature_pts_msg;
            sensor_msgs::msg::ChannelFloat32 id_of_point;
            sensor_msgs::msg::ChannelFloat32 u_of_point;
            sensor_msgs::msg::ChannelFloat32 v_of_point;
            sensor_msgs::msg::ChannelFloat32 velocity_x_of_point;
            sensor_msgs::msg::ChannelFloat32 velocity_y_of_point;
            feature_pts_msg.header = header;
            for (size_t j = 0; j < m_feature_tracker->ids().size(); j++)
            {
                if (m_feature_tracker->trackCnt()[j] <= 1)
                    continue;
                const int &id = m_feature_tracker->ids()[j];

                const cv::Point2f &xy = m_feature_tracker->curPtsXY()[j];
                const cv::Point2f &uv = m_feature_tracker->curPtsUV()[j];
                const cv::Point2f &velocity = m_feature_tracker->ptsVelocity()[j];
                geometry_msgs::msg::Point32 p;
                p.x = xy.x;
                p.y = xy.y;
                p.z = 1.0;
                feature_pts_msg.points.push_back(p);
                id_of_point.values.push_back(id);
                u_of_point.values.push_back(uv.x);
                v_of_point.values.push_back(uv.y);
                velocity_x_of_point.values.push_back(velocity.x);
                velocity_y_of_point.values.push_back(velocity.y);
            }
            feature_pts_msg.channels.push_back(id_of_point);
            feature_pts_msg.channels.push_back(u_of_point);
            feature_pts_msg.channels.push_back(v_of_point);
            feature_pts_msg.channels.push_back(velocity_x_of_point);
            feature_pts_msg.channels.push_back(velocity_y_of_point);
            m_feature_pts_pub->publish(feature_pts_msg);
        }
    }

    void publishFeatureImage(std_msgs::msg::Header &header, const cv::Mat &img)
    {
        if (m_feature_img_pub->get_subscription_count() < 1)
        {
            return;
        }
        cv::Mat show_img;
        cv::cvtColor(img, show_img, CV_GRAY2RGB);
        for (size_t j = 0; j < m_feature_tracker->curPtsUV().size(); j++)
        {
            double len = std::min(1.0, 1.0 * m_feature_tracker->trackCnt()[j] / m_feature_tracker->config().window_size);
            cv::circle(show_img, m_feature_tracker->curPtsUV()[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
        cv_bridge::CvImage cv_image;
        cv_image.header = header;
        cv_image.encoding = sensor_msgs::image_encodings::BGR8;
        cv_image.image = show_img;
        sensor_msgs::msg::Image::SharedPtr msg = cv_image.toImageMsg();
        m_feature_img_pub->publish(*msg);
    }

    void initSubscribers()
    {
        m_img_sub = this->create_subscription<sensor_msgs::msg::Image>(m_config.img_topic, 100, std::bind(&FeatureTrackerNode::imageCallback, this, std::placeholders::_1));
    }

    void initPublishers()
    {
        m_feature_pts_pub = this->create_publisher<sensor_msgs::msg::PointCloud>(m_config.feature_points_topic, 100);
        m_feature_img_pub = this->create_publisher<sensor_msgs::msg::Image>(m_config.feature_image_topic, 100);
    }

private:
    NodeConfig m_config;
    PinholeParams m_camera_params;
    NodeState m_state;
    TrackerConfig m_tracker_config;
    std::shared_ptr<FeatureTracker> m_feature_tracker;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_img_sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr m_feature_pts_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_feature_img_pub;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FeatureTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
