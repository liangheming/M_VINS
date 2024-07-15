#include <mutex>
#include <queue>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "pose_graph/pose_graph_4dof.h"

struct NodeConfig
{
    std::string img_topic = "/cam0/image_raw";
    std::string key_odom_topic = "/vins_estimator/keyframe_odom";
    std::string key_point_topic = "/vins_estimator/keyframe_points";
    std::string patern_file = "/home/zhouzhou/vs_projects/ws_vins/src/pose_graph/config/brief_pattern.yml";
    std::string vocabulary_file = "/home/zhouzhou/vs_projects/ws_vins/src/pose_graph/config/brief_k10L6.bin";

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
        
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseGraphNode>());
    rclcpp::shutdown();
}