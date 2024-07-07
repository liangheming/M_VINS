#include <memory>
#include <fstream>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>

struct NodeConfig
{
    std::string euroc_pose_topic;
    std::string save_text_path;
    std::string odom_topic;
};

class RecorderNode
{
public:
    RecorderNode() : m_nh("~")
    {
        loadConfig();
        initSubsriber();
    }
    void loadConfig()
    {
        m_nh.param<std::string>("pose_topic", m_config.euroc_pose_topic, "/leica/position");
        m_nh.param<std::string>("save_text_path", m_config.save_text_path, "/home/zhouzhou/temp/MH_01.txt");
        m_nh.param<std::string>("odom_topic", m_config.odom_topic, "/leica/position");
    }
    void initSubsriber()
    {
        // m_euroc_pose_sub = m_nh.subscribe(m_config.euroc_pose_topic, 10, &RecorderNode::eurocPoseCB, this);
        m_odom_sub = m_nh.subscribe(m_config.odom_topic, 10, &RecorderNode::odomCB, this);
    }
    void eurocPoseCB(const geometry_msgs::PointStampedPtr &msg)
    {
        // m_out_file = std::make_shared<std::ofstream>(m_config.save_text_path, std::ios::app);
        // double time_stamp = msg->header.stamp.toSec();
        // m_out_file->precision(9);
        // *m_out_file << time_stamp << " ";
        // m_out_file->precision(5);
        // *m_out_file << msg->point.x << " " << msg->point.y << " " << msg->point.z << " ";
        // m_out_file->precision(6);
        // *m_out_file << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
        // m_out_file->close();
    }

    void odomCB(const nav_msgs::OdometryPtr &msg)
    {
        ROS_INFO("REC ODOM : %.4f %.4f %.4f ;", msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        m_out_file = std::make_shared<std::ofstream>(m_config.save_text_path, std::ios::app);
        double time_stamp = msg->header.stamp.toSec();
        m_out_file->setf(std::ios::fixed, std::ios::floatfield);
        *m_out_file << time_stamp << " ";
        *m_out_file << msg->pose.pose.position.x << " " << msg->pose.pose.position.y << " " << msg->pose.pose.position.z << " ";
        *m_out_file << msg->pose.pose.orientation.x << " " << msg->pose.pose.orientation.y << " " << msg->pose.pose.orientation.z << " " << msg->pose.pose.orientation.w << std::endl;
        m_out_file->close();
    }

private:
    NodeConfig m_config;
    ros::NodeHandle m_nh;
    ros::Subscriber m_odom_sub;
    // ros::Subscriber m_euroc_pose_sub;

    std::shared_ptr<std::ofstream> m_out_file;
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "odom_recorder_node");
    RecorderNode recorder_node;
    ros::spin();
    ros::shutdown();
    return 0;
}
