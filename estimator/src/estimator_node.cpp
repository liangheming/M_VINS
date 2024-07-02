#include <ros/ros.h>
#include <mutex>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <queue>
#include "estimators/sw_estimator.h"
#include <tf/transform_broadcaster.h>

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

    std::mutex state_mutex;

    std::queue<sensor_msgs::ImuConstPtr> imu_buf;

    std::queue<sensor_msgs::PointCloudConstPtr> feature_buf;

    std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr> package;

    bool init_imu = false;

    double last_predict_time = -1.0;

    Vec3d tmp_P = Vec3d::Zero();
    Quatd tmp_Q = Quatd::Identity();
    Vec3d tmp_V = Vec3d::Zero();
    Vec3d tmp_Ba = Vec3d::Zero();
    Vec3d tmp_Bg = Vec3d::Zero();
    Vec3d acc_0 = Vec3d::Zero();
    Vec3d gyr_0 = Vec3d::Zero();
};

class EstimatorNode
{
public:
    void predict(const sensor_msgs::ImuConstPtr &imu_msg)
    {
        double t = imu_msg->header.stamp.toSec();
        if (!m_node_state.init_imu)
        {
            m_node_state.last_predict_time = t;
            m_node_state.init_imu = true;
            return;
        }
        double dt = t - m_node_state.last_predict_time;
        m_node_state.last_predict_time = t;
        Vec3d linear_acceleration(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
        Vec3d angular_velocity(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
        Vec3d un_acc_0 = m_node_state.tmp_Q * (m_node_state.acc_0 - m_node_state.tmp_Ba) + m_estimator->g;
        Vec3d un_gyr = 0.5 * (m_node_state.gyr_0 + angular_velocity) - m_node_state.tmp_Bg;
        m_node_state.tmp_Q = m_node_state.tmp_Q * deltaQ(un_gyr * dt);
        Vec3d un_acc_1 = m_node_state.tmp_Q * (linear_acceleration - m_node_state.tmp_Ba) + m_estimator->g;
        Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        m_node_state.tmp_P = m_node_state.tmp_P + dt * m_node_state.tmp_V + 0.5 * dt * dt * un_acc;
        m_node_state.tmp_V = m_node_state.tmp_V + dt * un_acc;
        m_node_state.acc_0 = linear_acceleration;
        m_node_state.gyr_0 = angular_velocity;
    }

    void update()
    {
        m_node_state.last_predict_time = m_node_state.propagate_time;
        m_node_state.tmp_P = m_estimator->ps[WINDOW_SIZE];
        m_node_state.tmp_Q = m_estimator->rs[WINDOW_SIZE];
        m_node_state.tmp_V = m_estimator->vs[WINDOW_SIZE];
        m_node_state.tmp_Ba = m_estimator->bas[WINDOW_SIZE];
        m_node_state.tmp_Bg = m_estimator->bgs[WINDOW_SIZE];
        m_node_state.acc_0 = m_estimator->state().acc_0;
        m_node_state.gyr_0 = m_estimator->state().gyro_0;

        std::queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = m_node_state.imu_buf;
        for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
            predict(tmp_imu_buf.front());
    }
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
        {
            std::lock_guard<std::mutex> lg(m_node_state.state_mutex);
            predict(msg);
        }
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

        if (m_estimator->solve_flag == SolveFlag::NON_LINEAR)
        {
            tf::Transform transform;
            tf::Quaternion q;
            Vec3d correct_t = m_estimator->ps[WINDOW_SIZE];
            Quatd correct_q(m_estimator->rs[WINDOW_SIZE]);
            transform.setOrigin(tf::Vector3(correct_t(0),
                                            correct_t(1),
                                            correct_t(2)));
            q.setW(correct_q.w());
            q.setX(correct_q.x());
            q.setY(correct_q.y());
            q.setZ(correct_q.z());
            transform.setRotation(q);
            m_br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(feature_time), "world", "body"));
        }

        // m_node_state.state_mutex.lock();
        // if (m_estimator->solve_flag == SolveFlag::NON_LINEAR)
        //     update();
        // m_node_state.state_mutex.unlock();
    }

private:
    ros::NodeHandle m_nh;
    tf::TransformBroadcaster m_br;
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