#include <mutex>
#include <queue>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "pose_graph/pose_graph_4dof.h"

#include <yaml-cpp/yaml.h>

struct NodeConfig
{
    std::string img_topic = "/cam0/image_raw";
    std::string key_odom_topic = "/vins_estimator/keyframe_odom";
    std::string key_point_topic = "/vins_estimator/keyframe_points";
    std::string patern_file = "brief_pattern.yml";
    std::string vocabulary_file = "brief_k10L6.bin";
    std::string config_file = "nothing.yaml";

    std::string map_frame = "world";
    std::string body_frame = "body";
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

    std::queue<sensor_msgs::msg::Image::SharedPtr> img_buf;
    std::queue<nav_msgs::msg::Odometry::SharedPtr> key_odom_buf;
    std::queue<sensor_msgs::msg::PointCloud::SharedPtr> key_point_buf;

    int skip_first_cnt = 0;
    int skip_cnt = 0;

    Vec3d last_t = Vec3d(-100.0, -100.0, -100.0);
};
class PoseGraphNode : public rclcpp::Node
{
public:
    PoseGraphNode() : rclcpp::Node("pose_graph_node")
    {
        loadParams();
    }
    void loadParams()
    {
        this->declare_parameter("img_topic", "/cam0/image_raw");
        this->declare_parameter("key_odom_topic", "/vins_estimator/keyframe_odom");
        this->declare_parameter("key_point_topic", "/vins_estimator/keyframe_points");
        this->declare_parameter("config_path", "nothing.yaml");

        this->get_parameter("img_topic", m_node_config.img_topic);
        this->get_parameter("key_odom_topic", m_node_config.key_odom_topic);
        this->get_parameter("key_point_topic", m_node_config.key_point_topic);
        this->get_parameter("config_path", m_node_config.config_file);

        if (std::filesystem::exists(m_node_config.config_file))
        {
            RCLCPP_INFO(this->get_logger(), "[PoseGraph] Load From File: %s", m_node_config.config_file.c_str());
            YAML::Node config = YAML::LoadFile(m_node_config.config_file);
        }
    }

private:
    NodeConfig m_node_config;
    NodeState m_node_state;
    PinholeParams m_camera_params;
    PoseGraphConfig m_pose_graph_config;
    std::shared_ptr<PoseGraph4DOF> m_pose_graph;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseGraphNode>());
    rclcpp::shutdown();
}