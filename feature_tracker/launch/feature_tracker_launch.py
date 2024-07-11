import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("feature_tracker"), "config", "euroc.yaml"
    )
    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="feature_tracker",
                executable="feature_tracker_node",
                name="feature_tracker",
                output="screen",
                parameters=[
                    {
                        "config_file": config_path,
                        "img_topic": "/cam0/image_raw",
                        "freq": 10.0,
                    }
                ],
            )
        ]
    )
