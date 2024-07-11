import os
import launch
import launch_ros.actions


def generate_launch_description():

    return launch.LaunchDescription(
        [
            launch_ros.actions.Node(
                package="odom_recorder",
                executable="odom_recorder_node",
                name="odom_recorder",
                namespace="odom_recorder",
                output="screen",
                parameters=[
                    {
                        "odom_topic": "/vins_estimator/odom",
                        "output_file": "/home/zhouzhou/temp/odom.csv",
                    }
                ],
            )
        ]
    )
