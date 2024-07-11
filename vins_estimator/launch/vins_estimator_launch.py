import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("vins_estimator"), "config", "euroc.yaml"
    )
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="feature_tracker",
                executable="feature_tracker_node",
                name="feature_tracker",
                output="screen",
                namespace="feature_tracker",
                parameters=[
                    {
                        "config_file": config_path,
                        "img_topic": "/cam0/image_raw",
                        "freq": 10.0,
                    }
                ],
            ),
            launch_ros.actions.Node(
                package="vins_estimator",
                executable="vins_estimator_node",
                name="vins_estimator",
                output="screen",
                namespace="vins_estimator",
                parameters=[
                    {
                        "config_file": config_path,
                        "feature_topic": "/feature_tracker/feature",
                        "imu_topic": "/imu0",
                        "world_frame": "camera",
                        "body_frame": "body",
                    }
                ],
            ),
        ]
    )
