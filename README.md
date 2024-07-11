# VINS_MONO FOR ROS2

## 主要工作
1. 基于原有[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)进行重构，以适配ROS2
2. 目前只提供了前端光流和后端滑窗优化两个基本模块，后续会跟进位姿图模块
3. 支持euroc数据包转ros2的bag

## 环境依赖
1. Ubuntu 22.04
2. ROS2 Humble

## 编译依赖
```text
Eigen3
sophus 1.22.10
opencv 4.0以上(ros2 humble 包含，无需额外安装)
ceres 2.1.0
```
## 转换脚本python依赖
```text
opencv-python
pandas
rclpy (安装ros2 humble后自带)
rosbag2_py (安装ros2 humble后自带)
```

## 详细说明
### 1. 编译 Sophus
```shell
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout 1.22.10
mkdir build && cd build
cmake .. -DSOPHUS_USE_BASIC_LOGGING=ON
make
sudo make install
```
### 2. 编译 ceres
```shell
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev

git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout 2.1.0
mkdir build && cd build
cmake ..
make -j4
sudo make install
```
### 3. 编译 M_VINS
```shell
mkdir ws_vins_ros2 && cd ws_vins_ros2
mkdir src && cd src
git clone https://github.com/liangheming/M_VINS.git
cd ..
colcon build
```

## 部分脚本

### 1. 数据包转换 euroc2rosbag2.py
```shell
python euroc2rosbag2.py --data_dir /home/username/euroc/V2_03_difficult --save_dir /home/username/rosbags
```
### 2. vins_estimator
```shell
ros2 launch vins_estimator vins_estimator_launch.py
```

### 3. 记录odom用来进行evo 评估
```shell
ros2 launch odom_recorder odom_recorder_launch.py
```

## TODO
- 添加后端回环检测和位姿图优化
- 添加ros相关消息和tf的发布
- 添加部分路径以及特征点的显示

## 特别感谢
- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
