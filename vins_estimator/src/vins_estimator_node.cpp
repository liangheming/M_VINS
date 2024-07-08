#include <queue>
#include <iostream>
#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include "estimator/commons.h"
#include "estimator/sw_estimator.h"

struct NodeConfig
{
    std::string feature_topic = "";
    std::string imu_topic = "";
    std::string map_frame = "camera";
    std::string body_frame = "body";
};

struct NodeState
{
    bool first_feature = true;
    double last_imu_time = 0.0;
    double last_feature_time = 0.0;
    double propagate_time = -1.0;

    std::mutex imu_mutex;
    std::mutex feature_mutex;

    std::queue<sensor_msgs::ImuConstPtr> imu_buf;
    std::queue<sensor_msgs::PointCloudConstPtr> feature_buf;

    std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr> package;
};

class VinsEstimatorNode
{
public:
    VinsEstimatorNode() : m_nh("~")
    {
        loadParams();
        initSubscribers();
        initPublisher();

        m_sw_estimator = std::make_shared<SlidingWindowEstimator>(m_estimator_config);
        m_timer = m_nh.createTimer(ros::Duration(0.02), &VinsEstimatorNode::mainCallback, this);
    }
    void loadParams()
    {
        ROS_INFO("[VinsEstimator] LOAD CONFIG");
        m_nh.param<std::string>("feature_topic", m_node_config.feature_topic, "");
        m_nh.param<std::string>("imu_topic", m_node_config.imu_topic, "");
        m_estimator_config.g_norm = 9.81007;
        m_estimator_config.ric << 0.0148655429818, -0.999880929698, 0.00414029679422, 0.999557249008, 0.0149672133247, 0.025715529948, -0.0257744366974, 0.00375618835797, 0.999660727178;
        m_estimator_config.tic << -0.0216401454975, -0.064676986768, 0.00981073058949;
    }
    void initSubscribers()
    {
        m_feature_sub = m_nh.subscribe(m_node_config.feature_topic, 100, &VinsEstimatorNode::featureCallback, this);
        m_imu_sub = m_nh.subscribe(m_node_config.imu_topic, 100, &VinsEstimatorNode::imuCallback, this);
    }
    void initPublisher()
    {
        m_odom_pub = m_nh.advertise<nav_msgs::Odometry>("vio_odom", 100);
        m_keyframe_odom_pub = m_nh.advertise<nav_msgs::Odometry>("keyframe_odom", 100);
        m_keyframe_point_pub = m_nh.advertise<sensor_msgs::PointCloud>("keyframe_points", 100);
    }
    void featureCallback(const sensor_msgs::PointCloudConstPtr &msg)
    {
        if (m_node_state.first_feature)
        {
            m_node_state.first_feature = false;
            return;
        }
        if (msg->header.stamp.toSec() <= m_node_state.last_feature_time)
        {
            ROS_WARN("FEATURE MESSAGE IN DISORDER!");
            return;
        }
        std::lock_guard<std::mutex> lock(m_node_state.feature_mutex);
        m_node_state.last_feature_time = msg->header.stamp.toSec();
        m_node_state.feature_buf.push(msg);
    }

    void imuCallback(const sensor_msgs::ImuConstPtr &msg)
    {
        if (msg->header.stamp.toSec() <= m_node_state.last_imu_time)
        {
            ROS_WARN("IMU MESSAGE IN DISORDER!");
            return;
        }
        std::lock_guard<std::mutex> lock(m_node_state.imu_mutex);
        m_node_state.last_imu_time = msg->header.stamp.toSec();
        m_node_state.imu_buf.push(msg);
    }
    bool syncPackage()
    {
        // TODO: time estimator
        if (m_node_state.imu_buf.empty() || m_node_state.feature_buf.empty())
            return false;

        if (m_node_state.imu_buf.back()->header.stamp.toSec() <= m_node_state.feature_buf.front()->header.stamp.toSec())
            return false;

        if (m_node_state.imu_buf.front()->header.stamp.toSec() >= m_node_state.feature_buf.front()->header.stamp.toSec())
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
            while (m_node_state.imu_buf.front()->header.stamp.toSec() < m_node_state.package.second->header.stamp.toSec())
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
        double feature_time = m_node_state.package.second->header.stamp.toSec();
        for (auto &imu : m_node_state.package.first)
        {
            double imu_time = imu->header.stamp.toSec();
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
        sensor_msgs::PointCloudConstPtr &feature = m_node_state.package.second;
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
        m_sw_estimator->processFeature(feats, feature->header.stamp.toSec());
    }
    void publishOdom()
    {
        if (m_odom_pub.getNumSubscribers() < 1)
            return;
        if (m_sw_estimator->solve_flag == SolveFlag::INITIAL)
            return;
        nav_msgs::Odometry msg;
        Vec3d position = m_sw_estimator->state().ps[WINDOW_SIZE];
        Mat3d rotation = m_sw_estimator->state().rs[WINDOW_SIZE];
        Quatd quat(rotation);
        msg.header.stamp = ros::Time().fromSec(m_node_state.propagate_time);
        msg.header.frame_id = m_node_config.map_frame;
        msg.child_frame_id = m_node_config.body_frame;
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
        m_odom_pub.publish(msg);
    }

    void publishKeyFrameOdom()
    {
        if (m_keyframe_odom_pub.getNumSubscribers() < 1)
            return;
        if (m_sw_estimator->solve_flag == SolveFlag::INITIAL || m_sw_estimator->marginalization_flag == MarginFlag::MARGIN_SECOND_NEW)
            return;
        int i = WINDOW_SIZE - 2;
        Vec3d position = m_sw_estimator->state().ps[i];
        Mat3d rotation = m_sw_estimator->state().rs[i];
        Quatd quat(rotation);
        nav_msgs::Odometry odometry;
        odometry.header.frame_id = m_node_config.map_frame;
        odometry.header.stamp = ros::Time().fromSec(m_sw_estimator->state().timestamps[i]);
        odometry.child_frame_id = m_node_config.body_frame;
        odometry.pose.pose.position.x = position.x();
        odometry.pose.pose.position.y = position.y();
        odometry.pose.pose.position.z = position.z();
        odometry.pose.pose.orientation.x = quat.x();
        odometry.pose.pose.orientation.y = quat.y();
        odometry.pose.pose.orientation.z = quat.z();
        odometry.pose.pose.orientation.w = quat.w();
        m_keyframe_odom_pub.publish(odometry);
    }
    void publishKeyFramePoints()
    {
        if (m_keyframe_point_pub.getNumSubscribers() < 1)
            return;
        if (m_sw_estimator->solve_flag == SolveFlag::INITIAL || m_sw_estimator->marginalization_flag == MarginFlag::MARGIN_SECOND_NEW)
            return;
        int i = WINDOW_SIZE - 2;
        sensor_msgs::PointCloud point_cloud;

        point_cloud.header.stamp = ros::Time().fromSec(m_sw_estimator->state().timestamps[i]);
        for (auto &it_per_id : m_sw_estimator->feature_manager.features)
        {
            int frame_size = it_per_id.observations.size();
            if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {
                int imu_i = it_per_id.start_frame;
                Vec3d pts_i = it_per_id.observations[0].point * it_per_id.estimated_depth;
                Vec3d w_pts_i = m_sw_estimator->state().rs[imu_i] * (m_sw_estimator->state().ric * pts_i + m_sw_estimator->state().tic) + m_sw_estimator->state().ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);
                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.observations[imu_j].point.x());
                p_2d.values.push_back(it_per_id.observations[imu_j].point.y());
                p_2d.values.push_back(it_per_id.observations[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.observations[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }
        }
        point_cloud.header.frame_id = m_node_config.map_frame;
        m_keyframe_point_pub.publish(point_cloud);
    }
    void mainCallback(const ros::TimerEvent &event)
    {
        if (!syncPackage())
            return;
        processImus();
        processFeatures();
        publishOdom();
        publishKeyFrameOdom();
        publishKeyFramePoints();
    }

private:
    ros::NodeHandle m_nh;
    tf::TransformBroadcaster m_br;
    NodeConfig m_node_config;
    NodeState m_node_state;
    ros::Subscriber m_feature_sub;
    ros::Subscriber m_imu_sub;
    ros::Publisher m_odom_pub;
    ros::Publisher m_keyframe_odom_pub;
    ros::Publisher m_keyframe_point_pub;
    ros::Timer m_timer;
    EstimatorConfig m_estimator_config;
    std::shared_ptr<SlidingWindowEstimator> m_sw_estimator;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    VinsEstimatorNode node;
    ros::spin();
    return 0;
}