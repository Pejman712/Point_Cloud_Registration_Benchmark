#!/usr/bin/env python3

import argparse
import importlib
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
    DurabilityPolicy,
)


def import_message_type(type_string: str):
    """
    Supports strings like:
      nav_msgs/msg/Odometry
      geometry_msgs/msg/PoseStamped
      geometry_msgs/msg/Pose
      geometry_msgs/msg/TransformStamped
    """
    cleaned = type_string.replace("/msg/", "/")
    parts = cleaned.split("/")

    if len(parts) != 2:
        raise ValueError(f"Unsupported message type format: {type_string}")

    package_name, message_name = parts
    module = importlib.import_module(f"{package_name}.msg")
    return getattr(module, message_name)


def stamp_to_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def extract_timestamp(msg: Any, node: Node) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        if stamp.sec != 0 or stamp.nanosec != 0:
            return stamp_to_float(stamp)

    return node.get_clock().now().nanoseconds * 1e-9


def extract_pose(msg: Any):
    """
    Handles:
      nav_msgs/msg/Odometry              -> msg.pose.pose
      geometry_msgs/msg/PoseStamped      -> msg.pose
      geometry_msgs/msg/Pose             -> msg
      geometry_msgs/msg/TransformStamped -> msg.transform
    """

    # nav_msgs/Odometry or geometry_msgs/PoseStamped
    if hasattr(msg, "pose"):
        pose = msg.pose

        # Odometry has msg.pose.pose
        if hasattr(pose, "pose"):
            pose = pose.pose

        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            return pose.position, pose.orientation

    # geometry_msgs/Pose
    if hasattr(msg, "position") and hasattr(msg, "orientation"):
        return msg.position, msg.orientation

    # geometry_msgs/TransformStamped
    if hasattr(msg, "transform"):
        return msg.transform.translation, msg.transform.rotation

    raise TypeError(
        "Could not extract pose. Expected Odometry, PoseStamped, Pose, or TransformStamped."
    )


class TumRecorder(Node):
    def __init__(self, topic: str, msg_type: str, output: str, delimiter: str):
        super().__init__("aft_mapped_tum_recorder")

        self.topic = topic
        self.msg_type = msg_type
        self.output = output
        self.delimiter = "," if delimiter == "comma" else " "
        self.count = 0
        self.subscription = None

        self.file = open(self.output, "w", buffering=1)

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        if self.msg_type == "auto":
            self.get_logger().info(f"Waiting for topic type of {self.topic}")
            self.timer = self.create_timer(0.2, self.try_auto_subscribe)
        else:
            self.subscribe_to_topic(self.msg_type)

    def try_auto_subscribe(self):
        topics = self.get_topic_names_and_types()

        preferred_types = [
            "nav_msgs/msg/Odometry",
            "geometry_msgs/msg/PoseStamped",
            "geometry_msgs/msg/Pose",
            "geometry_msgs/msg/TransformStamped",
        ]

        for name, types in topics:
            if name != self.topic:
                continue

            selected_type = None

            for preferred in preferred_types:
                if preferred in types:
                    selected_type = preferred
                    break

            if selected_type is None and len(types) > 0:
                selected_type = types[0]

            if selected_type is not None:
                self.subscribe_to_topic(selected_type)
                self.timer.cancel()
                return

    def subscribe_to_topic(self, msg_type: str):
        msg_class = import_message_type(msg_type)

        self.subscription = self.create_subscription(
            msg_class,
            self.topic,
            self.callback,
            self.qos,
        )

        self.get_logger().info(
            f"Recording {self.topic} as {msg_type} into {self.output}"
        )

    def callback(self, msg: Any):
        try:
            timestamp = extract_timestamp(msg, self)
            position, orientation = extract_pose(msg)

            values = [
                f"{timestamp:.9f}",
                f"{position.x:.9f}",
                f"{position.y:.9f}",
                f"{position.z:.9f}",
                f"{orientation.x:.9f}",
                f"{orientation.y:.9f}",
                f"{orientation.z:.9f}",
                f"{orientation.w:.9f}",
            ]

            self.file.write(self.delimiter.join(values) + "\n")
            self.count += 1

        except Exception as exc:
            self.get_logger().error(f"Failed to write pose: {exc}")

    def close(self):
        self.get_logger().info(f"Saved {self.count} poses to {self.output}")
        self.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/aft_mapped_to_init")
    parser.add_argument("--type", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--delimiter",
        choices=["space", "comma"],
        default="space",
        help="Use 'space' for normal TUM-style output or 'comma' for CSV-style output.",
    )

    args = parser.parse_args()

    rclpy.init()
    node = TumRecorder(
        topic=args.topic,
        msg_type=args.type,
        output=args.output,
        delimiter=args.delimiter,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
