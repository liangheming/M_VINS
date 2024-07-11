import os
import cv2
import shutil
import rosbag2_py
import argparse
import pandas as pd

from cv_bridge import CvBridge
from sensor_msgs.msg import Imu
from rclpy.serialization import serialize_message


class ImageConvert:
    def __init__(self, data_dir, use_gray, save_dir="."):
        self.bridge = CvBridge()
        self.cam0_dir = os.path.join(data_dir, "mav0/cam0/")
        self.cam1_dir = os.path.join(data_dir, "mav0/cam1/")
        self.imu_dir = os.path.join(data_dir, "mav0/imu0/")
        self.bag_dir = os.path.join(save_dir, "ros_bag_" + os.path.basename(data_dir))
        self.use_gray = use_gray

    def convert(self):

        if os.path.exists(self.bag_dir):
            shutil.rmtree(self.bag_dir)

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_dir, storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        writer = rosbag2_py.SequentialWriter()
        writer.open(storage_options, converter_options)
        topic_name = "/cam0/image_raw"
        print("Writing topic: {}".format(topic_name))
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic_name,
                type="sensor_msgs/msg/Image",
                serialization_format="cdr",
            )
        )
        cam0_csv = pd.read_csv(
            os.path.join(self.cam0_dir, "data.csv"), header=None, names=["ns", "png"]
        ).to_numpy()[1:]
        for ns, img_name in cam0_csv:
            image_path = os.path.join(self.cam0_dir, "data", img_name)
            cv_img = cv2.imread(image_path)
            if self.use_gray:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="mono8")
            else:
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            img_msg.header.stamp.sec = int(float(ns) / 1e9)
            img_msg.header.stamp.nanosec = int(float(ns) % 1e9)
            img_msg.header.frame_id = "cam0"
            msg_str = serialize_message(img_msg)
            writer.write(topic_name, msg_str, int(ns))

        topic_name = "/cam1/image_raw"
        print("Writing topic: {}".format(topic_name))
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic_name,
                type="sensor_msgs/msg/Image",
                serialization_format="cdr",
            )
        )

        cam1_csv = pd.read_csv(
            os.path.join(self.cam1_dir, "data.csv"), header=None, names=["ns", "png"]
        ).to_numpy()[1:]

        for ns, img_name in cam1_csv:
            image_path = os.path.join(self.cam1_dir, "data", img_name)
            cv_img = cv2.imread(image_path)
            if self.use_gray:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="mono8")
            else:
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            img_msg.header.stamp.sec = int(float(ns) / 1e9)
            img_msg.header.stamp.nanosec = int(float(ns) % 1e9)
            img_msg.header.frame_id = "cam1"
            msg_str = serialize_message(img_msg)
            writer.write(topic_name, msg_str, int(ns))

        topic_name = "/imu0"
        print("Writing topic: {}".format(topic_name))
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic_name,
                type="sensor_msgs/msg/Imu",
                serialization_format="cdr",
            )
        )
        imu_csv = pd.read_csv(
            os.path.join(self.imu_dir, "data.csv"), header=None
        ).to_numpy()[1:]
        for ns, wx, wy, wz, ax, ay, az in imu_csv:
            imu_msg = Imu()
            imu_msg.header.stamp.sec = int(float(ns) / 1e9)
            imu_msg.header.stamp.nanosec = int(float(ns) % 1e9)
            imu_msg.angular_velocity.x = float(wx)
            imu_msg.angular_velocity.y = float(wy)
            imu_msg.angular_velocity.z = float(wz)
            imu_msg.linear_acceleration.x = float(ax)
            imu_msg.linear_acceleration.y = float(ay)
            imu_msg.linear_acceleration.z = float(az)
            writer.write(topic_name, serialize_message(imu_msg), int(ns))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_dir", default=".", type=str)
    parse.add_argument("--use_gray", default=True, type=bool)
    parse.add_argument("--save_dir", default=".", type=str)
    args = parse.parse_args()
    if(not os.path.exists(args.data_dir)):
        print("data_dir:", args.data_dir," not exists")
        exit(0)
        

    image_converter = ImageConvert(args.data_dir, args.use_gray, args.save_dir)
    image_converter.convert()
