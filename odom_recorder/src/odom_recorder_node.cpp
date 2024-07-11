#include <filesystem>

#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include <fstream>

double stamp2double(builtin_interfaces::msg::Time stamp)
{
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

struct NodeConfig
{
    std::string odom_topic;
    std::string output_file;
};

class OdomRecorderNode : public rclcpp::Node
{
public:
    OdomRecorderNode() : Node("odom_recorder_node")
    {
        this->declare_parameter<std::string>("odom_topic", "/odom");
        this->declare_parameter<std::string>("output_file", "/home/zhouzhou/odom.csv");
        m_config.odom_topic = this->get_parameter("odom_topic").as_string();
        m_config.output_file = this->get_parameter("output_file").as_string();

        std::filesystem::path output_file_path(m_config.output_file);
        if (!std::filesystem::exists(output_file_path.parent_path()))
        {
            RCLCPP_ERROR(this->get_logger(), "output file parent dir not exist: %s", m_config.output_file.c_str());
            exit(1);
        }
        if (std::filesystem::exists(output_file_path))
        {
            RCLCPP_WARN(this->get_logger(), "output file already exist: %s, removed", m_config.output_file.c_str());
            std::filesystem::remove(output_file_path);
        }
        m_odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(m_config.odom_topic, 100, std::bind(&OdomRecorderNode::odomCallBack, this, std::placeholders::_1));
    }

    void odomCallBack(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "odom received, x: %.6f y: %.6f z: %.6f ", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        std::ofstream file_out(m_config.output_file, std::ios::app);
        double timestamp = stamp2double(msg->header.stamp);
        file_out.setf(std::ios::fixed, std::ios::floatfield);
        file_out << timestamp << " ";
        file_out << msg->pose.pose.position.x << " " << msg->pose.pose.position.y << " " << msg->pose.pose.position.z << " ";
        file_out << msg->pose.pose.orientation.x << " " << msg->pose.pose.orientation.y << " " << msg->pose.pose.orientation.z << " " << msg->pose.pose.orientation.w << std::endl;
        file_out.close();
    }

private:
    NodeConfig m_config;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_odom_sub;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomRecorderNode>());
    rclcpp::shutdown();
    return 0;
}