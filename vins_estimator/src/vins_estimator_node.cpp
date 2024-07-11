#include <mutex>
#include <memory>
#include <queue>
#include <rclcpp/rclcpp.hpp>
#include <filesystem>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>

#include <nav_msgs/msg/odometry.hpp>

#include "estimator/sw_estimator.h"

#include <yaml-cpp/yaml.h>
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
  std::string feature_topic = "/feature_tracker/feature";
  std::string imu_topic = "/imu0";
  std::string world_frame = "camera";
  std::string body_frame = "body";
  std::string config_file = "nothing.yaml";

  std::string odom_topic = "odom";
  std::string key_odom_topic = "key_odom";
  std::string key_pts_topic = "key_pts";
};

struct NodeState
{
  bool first_feature = true;
  double last_imu_time = 0.0;
  double last_feature_time = 0.0;
  double propagate_time = -1.0;

  std::mutex imu_mutex;
  std::mutex feature_mutex;

  std::queue<sensor_msgs::msg::Imu::SharedPtr> imu_buf;
  std::queue<sensor_msgs::msg::PointCloud::SharedPtr> feature_buf;

  std::pair<std::vector<sensor_msgs::msg::Imu::SharedPtr>, sensor_msgs::msg::PointCloud::SharedPtr> package;
};

class VinsEstimatorNode : public rclcpp::Node
{
public:
  VinsEstimatorNode() : Node("vins_estimator_node")
  {
    loadParams();
    initSubscriber();
    initPublisher();
    m_sw_estimator = std::make_shared<SlidingWindowEstimator>(m_estimator_config);
    m_timer = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&VinsEstimatorNode::mainCallback, this));
  }
  void loadParams()
  {
    this->declare_parameter("feature_topic", "feature");
    this->declare_parameter("imu_topic", "imu_raw");
    this->declare_parameter("world_frame", "camera");
    this->declare_parameter("body_frame", "body");
    this->declare_parameter("config_file", "nothing.yaml");
    this->get_parameter("feature_topic", m_node_config.feature_topic);
    this->get_parameter("imu_topic", m_node_config.imu_topic);
    this->get_parameter("world_frame", m_node_config.world_frame);
    this->get_parameter("body_frame", m_node_config.body_frame);
    this->get_parameter("config_file", m_node_config.config_file);

    if (std::filesystem::exists(m_node_config.config_file))
    {
      RCLCPP_INFO(this->get_logger(), "[VinsEstimator] Load From File: %s", m_node_config.config_file.c_str());
      YAML::Node config = YAML::LoadFile(m_node_config.config_file);
      const YAML::Node &estimator = config["estimator"];
      m_estimator_config.axis_min_parallax = estimator["axis_min_parallax"].as<double>() / 460.0;
      m_estimator_config.estimate_ext = estimator["estimate_ext"].as<bool>();
      m_estimator_config.estimate_td = estimator["estimate_td"].as<bool>();
      m_estimator_config.g_norm = estimator["g_norm"].as<double>();
      m_estimator_config.proj_sqrt_info = Mat2d::Identity() * 460.0 / estimator["observe_cov"].as<double>();
      m_estimator_config.ransac_threshold = estimator["ransac_threshold"].as<double>() / 460.0;
      std::vector<double> ric_vec, tic_vec;
      ric_vec = estimator["ric"].as<std::vector<double>>();
      tic_vec = estimator["tic"].as<std::vector<double>>();
      m_estimator_config.ric << ric_vec[0], ric_vec[1], ric_vec[2], ric_vec[3], ric_vec[4], ric_vec[5], ric_vec[6], ric_vec[7], ric_vec[8];
      m_estimator_config.ric = Quatd(m_estimator_config.ric).toRotationMatrix();
      m_estimator_config.tic << tic_vec[0], tic_vec[1], tic_vec[2];
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "[VinsEstimator] File: %s Not Exits, Load From Default", m_node_config.config_file.c_str());
    }
  }
  void initSubscriber()
  {
    m_imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(
        m_node_config.imu_topic, 1000, std::bind(&VinsEstimatorNode::imuCallback, this, std::placeholders::_1));
    m_feature_sub = this->create_subscription<sensor_msgs::msg::PointCloud>(
        m_node_config.feature_topic, 1000, std::bind(&VinsEstimatorNode::featureCallback, this, std::placeholders::_1));
  }
  void initPublisher()
  {
    m_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(m_node_config.odom_topic, 100);
    m_key_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>(m_node_config.key_odom_topic, 100);
    m_key_pts_pub = this->create_publisher<sensor_msgs::msg::PointCloud>(m_node_config.key_pts_topic, 100);
  }

  void featureCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
  {
    if (m_node_state.first_feature)
    {
      m_node_state.first_feature = false;
      return;
    }
    double current_time = stamp2double(msg->header.stamp);
    if (current_time <= m_node_state.last_feature_time)
    {
      RCLCPP_WARN(this->get_logger(), "FEATURE MESSAGE IN DISORDER!");
      return;
    }
    std::lock_guard<std::mutex> lock(m_node_state.feature_mutex);
    m_node_state.last_feature_time = current_time;
    m_node_state.feature_buf.push(msg);
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    double current_time = stamp2double(msg->header.stamp);
    if (current_time <= m_node_state.last_imu_time)
    {
      RCLCPP_WARN(this->get_logger(), "IMU MESSAGE IN DISORDER!");
      return;
    }
    std::lock_guard<std::mutex> lock(m_node_state.imu_mutex);
    m_node_state.last_imu_time = current_time;
    m_node_state.imu_buf.push(msg);
  }

  bool syncPackage()
  {
    if (m_node_state.imu_buf.empty() || m_node_state.feature_buf.empty())
      return false;

    if (stamp2double(m_node_state.imu_buf.back()->header.stamp) <= stamp2double(m_node_state.feature_buf.front()->header.stamp))
      return false;

    if (stamp2double(m_node_state.imu_buf.front()->header.stamp) >= stamp2double(m_node_state.feature_buf.front()->header.stamp))
    {
      m_node_state.feature_buf.pop();
      return false;
    }

    m_node_state.package.second = m_node_state.feature_buf.front();

    m_node_state.feature_mutex.lock();
    m_node_state.feature_buf.pop();
    m_node_state.feature_mutex.unlock();

    m_node_state.package.first.clear();
    {
      std::lock_guard<std::mutex> lock(m_node_state.imu_mutex);
      while (stamp2double(m_node_state.imu_buf.front()->header.stamp) < stamp2double(m_node_state.package.second->header.stamp))
      {
        m_node_state.package.first.push_back(m_node_state.imu_buf.front());
        m_node_state.imu_buf.pop();
      }
      m_node_state.package.first.push_back(m_node_state.imu_buf.front());
    }
    return true;
  }

  void processImus()
  {
    double feature_time = stamp2double(m_node_state.package.second->header.stamp);
    for (auto &imu : m_node_state.package.first)
    {
      double imu_time = stamp2double(imu->header.stamp);
      Vec3d acc1, gyro1;
      if (imu_time <= feature_time)
      {
        if (m_node_state.propagate_time < 0.0)
          m_node_state.propagate_time = imu_time;
        double dt = imu_time - m_node_state.propagate_time;
        m_node_state.propagate_time = imu_time;
        Vec3d acc = Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
        Vec3d gyro = Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
        acc1 = acc;
        gyro1 = gyro;
        m_sw_estimator->processImu(dt, acc, gyro);
      }
      else
      {
        double dt_1 = feature_time - m_node_state.propagate_time;
        double dt_2 = imu_time - feature_time;
        m_node_state.propagate_time = feature_time;
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        Vec3d acc2 = Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
        Vec3d gyro2 = Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
        Vec3d acc = acc1 * w1 + acc2 * w2;
        Vec3d gyro = gyro1 * w1 + gyro2 * w2;
        m_sw_estimator->processImu(dt_1, acc, gyro);
      }
    }
  }

  void processFeatures()
  {
    TrackedFeatures feats;
    sensor_msgs::msg::PointCloud::SharedPtr &feature = m_node_state.package.second;
    for (size_t i = 0; i < feature->points.size(); i++)
    {
      double x = feature->points[i].x;
      double y = feature->points[i].y;
      double z = feature->points[i].z;
      int id = feature->channels[0].values[i];
      double u = feature->channels[1].values[i];
      double v = feature->channels[2].values[i];
      double vx = feature->channels[3].values[i];
      double vy = feature->channels[4].values[i];
      Vec7d val;
      val << x, y, z, u, v, vx, vy;
      feats[id] = val;
    }
    m_sw_estimator->processFeature(feats, stamp2double(feature->header.stamp));
  }

  void publishOdom(const builtin_interfaces::msg::Time &stamp)
  {
    if (m_odom_pub->get_subscription_count() < 1)
      return;
    if (m_sw_estimator->solve_flag == SolveFlag::INITIAL)
      return;
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = m_node_config.world_frame;
    msg.child_frame_id = m_node_config.body_frame;
    Vec3d position = m_sw_estimator->state().ps[WINDOW_SIZE];
    Mat3d rotation = m_sw_estimator->state().rs[WINDOW_SIZE];
    Quatd quat(rotation);
    msg.pose.pose.position.x = position.x();
    msg.pose.pose.position.y = position.y();
    msg.pose.pose.position.z = position.z();
    msg.pose.pose.orientation.x = quat.x();
    msg.pose.pose.orientation.y = quat.y();
    msg.pose.pose.orientation.z = quat.z();
    msg.pose.pose.orientation.w = quat.w();
    msg.twist.twist.linear.x = m_sw_estimator->state().vs[WINDOW_SIZE].x();
    msg.twist.twist.linear.y = m_sw_estimator->state().vs[WINDOW_SIZE].y();
    msg.twist.twist.linear.z = m_sw_estimator->state().vs[WINDOW_SIZE].z();
    m_odom_pub->publish(msg);
  }

  void publishKeyFrameOdom(const builtin_interfaces::msg::Time &stamp)
  {

    if (m_key_odom_pub->get_subscription_count() < 1)
      return;
    if (m_sw_estimator->solve_flag == SolveFlag::INITIAL || m_sw_estimator->marginalization_flag == MarginFlag::MARGIN_SECOND_NEW)
      return;
    int i = WINDOW_SIZE - 2;
    Vec3d position = m_sw_estimator->state().ps[i];
    Mat3d rotation = m_sw_estimator->state().rs[i];
    Quatd quat(rotation);
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = m_node_config.world_frame;
    msg.child_frame_id = m_node_config.body_frame;
    msg.pose.pose.position.x = position.x();
    msg.pose.pose.position.y = position.y();
    msg.pose.pose.position.z = position.z();
    msg.pose.pose.orientation.x = quat.x();
    msg.pose.pose.orientation.y = quat.y();
    msg.pose.pose.orientation.z = quat.z();
    msg.pose.pose.orientation.w = quat.w();
    m_key_odom_pub->publish(msg);
  }

  void publishKeyFramePoints(const builtin_interfaces::msg::Time &stamp)
  {
    if (m_key_pts_pub->get_subscription_count() < 1)
      return;
    if (m_sw_estimator->solve_flag == SolveFlag::INITIAL || m_sw_estimator->marginalization_flag == MarginFlag::MARGIN_SECOND_NEW)
      return;
    sensor_msgs::msg::PointCloud point_cloud;
    point_cloud.header.stamp = stamp;
    point_cloud.header.frame_id = m_node_config.world_frame;
    for (auto &it_per_id : m_sw_estimator->feature_manager.features)
    {
      int frame_size = it_per_id.observations.size();
      if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
      {
        int imu_i = it_per_id.start_frame;
        Vec3d pts_i = it_per_id.observations[0].point * it_per_id.estimated_depth;
        Vec3d w_pts_i = m_sw_estimator->state().rs[imu_i] * (m_sw_estimator->state().ric * pts_i + m_sw_estimator->state().tic) + m_sw_estimator->state().ps[imu_i];
        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
        int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
        sensor_msgs::msg::ChannelFloat32 p_2d;
        p_2d.values.push_back(it_per_id.observations[imu_j].point.x());
        p_2d.values.push_back(it_per_id.observations[imu_j].point.y());
        p_2d.values.push_back(it_per_id.observations[imu_j].uv.x());
        p_2d.values.push_back(it_per_id.observations[imu_j].uv.y());
        p_2d.values.push_back(it_per_id.feature_id);
        point_cloud.channels.push_back(p_2d);
      }
    }
    m_key_pts_pub->publish(point_cloud);
  }

  void mainCallback()
  {
    if (!syncPackage())
      return;

    processImus();
    processFeatures();
    builtin_interfaces::msg::Time cur_time = m_node_state.package.second->header.stamp;
    publishOdom(cur_time);
    publishKeyFrameOdom(cur_time);
    publishKeyFramePoints(cur_time);
  }

private:
  NodeConfig m_node_config;
  NodeState m_node_state;
  EstimatorConfig m_estimator_config;
  std::shared_ptr<SlidingWindowEstimator> m_sw_estimator;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr m_feature_sub;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr m_key_pts_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_key_odom_pub;

  rclcpp::TimerBase::SharedPtr m_timer;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VinsEstimatorNode>());
  rclcpp::shutdown();
  return 0;
}