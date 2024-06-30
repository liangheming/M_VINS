#include <ros/ros.h>
#include <mutex>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <queue>
#include "estimators/sw_estimator.h"

struct NodeConfig
{
    std::string feature_topic = "";
    std::string imu_topic = "";
};
struct NodeState
{
    bool first_featur = true;
    double last_imu_time = 0.0;
    double last_feature_time = 0.0;
    double propagate_time = -1.0;

    std::mutex m_imu_mutex;

    std::mutex m_feature_mutex;

    std::queue<sensor_msgs::ImuConstPtr> imu_buf;

    std::queue<sensor_msgs::PointCloudConstPtr> feature_buf;

    std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr> package;
};

class EstimatorNode
{
public:
    EstimatorNode(NodeConfig &config) : m_nh("~"), m_node_config(config)
    {
        ROS_INFO("EstimatorNode Start!");
        loadParams();
        initSubscribers();
        m_estimator = std::make_shared<SlideWindowEstimator>(m_sw_config);
        m_estimator->r_ic << 0.0148655429818, -0.999880929698, 0.00414029679422, 0.999557249008, 0.0149672133247, 0.025715529948, -0.0257744366974, 0.00375618835797, 0.999660727178;
        m_estimator->t_ic << -0.0216401454975, -0.064676986768, 0.00981073058949;
        m_timer = m_nh.createTimer(ros::Duration(0.02), &EstimatorNode::mainCallback, this);
    }

    void loadParams()
    {
        m_nh.param<std::string>("feature_topic", m_node_config.feature_topic, "");
        m_nh.param<std::string>("imu_topic", m_node_config.imu_topic, "");
    }
    void initSubscribers()
    {
        m_feature_sub = m_nh.subscribe(m_node_config.feature_topic, 100, &EstimatorNode::featureCallback, this);
        m_imu_sub = m_nh.subscribe(m_node_config.imu_topic, 100, &EstimatorNode::imuCallback, this);
    }

    void featureCallback(const sensor_msgs::PointCloudConstPtr &msg)
    {
        if (m_node_state.first_featur)
        {
            m_node_state.first_featur = false;
            return;
        }
        if (msg->header.stamp.toSec() <= m_node_state.last_feature_time)
        {
            ROS_WARN("FEATURE MESSAGE IN DISORDER!");
            return;
        }
        std::lock_guard<std::mutex> lock(m_node_state.m_feature_mutex);
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
        std::lock_guard<std::mutex> lock(m_node_state.m_imu_mutex);
        m_node_state.last_imu_time = msg->header.stamp.toSec();
        m_node_state.imu_buf.push(msg);

        // TODO 按IMU的频率更新ODOM
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

        m_node_state.m_feature_mutex.lock();
        m_node_state.feature_buf.pop();
        m_node_state.m_feature_mutex.unlock();

        m_node_state.package.first.clear();
        {
            std::lock_guard<std::mutex> lock(m_node_state.m_imu_mutex);
            while (m_node_state.imu_buf.front()->header.stamp.toSec() < m_node_state.package.second->header.stamp.toSec())
            {
                m_node_state.package.first.push_back(m_node_state.imu_buf.front());
                m_node_state.imu_buf.pop();
            }
            m_node_state.package.first.push_back(m_node_state.imu_buf.front());
        }
        return true;
    }

    void mainCallback(const ros::TimerEvent &event)
    {
        if (!syncPackage())
            return;
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
                m_estimator->processImu(dt, acc, gyro);
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
                m_estimator->processImu(dt_1, acc, gyro);
            }
        }

        Feats feats;
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
        m_estimator->processFeature(feats, feature_time);
    }

private:
    ros::NodeHandle m_nh;
    NodeConfig m_node_config;
    NodeState m_node_state;
    SWConfig m_sw_config;
    ros::Subscriber m_feature_sub;
    ros::Subscriber m_imu_sub;
    ros::Timer m_timer;

    std::shared_ptr<SlideWindowEstimator> m_estimator;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "estimator_node");
    NodeConfig config;
    EstimatorNode estimator_node(config);
    ros::spin();
    return 0;
}