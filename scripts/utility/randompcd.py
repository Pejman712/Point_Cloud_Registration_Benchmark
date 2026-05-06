#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


def find_ros2_bags(dataset_root: Path):
    """
    Find ROS 2 bag folders by locating metadata.yaml files.
    Returns the parent directory of each metadata.yaml.
    """
    bags = []
    for metadata in dataset_root.rglob("metadata.yaml"):
        bag_dir = metadata.parent
        if any(bag_dir.glob("*.db3")):
            bags.append(bag_dir)

    return sorted(bags)


def get_pointcloud2_topics(metadata_path: Path):
    """
    Parse metadata.yaml manually enough to find PointCloud2 topics.
    This avoids requiring PyYAML.
    """
    topics = []
    current_topic = None
    current_type = None

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("name:"):
                current_topic = stripped.split("name:", 1)[1].strip()

            elif stripped.startswith("type:"):
                current_type = stripped.split("type:", 1)[1].strip()

                if current_topic and current_type == "sensor_msgs/msg/PointCloud2":
                    topics.append(current_topic)

                current_topic = None
                current_type = None

    return topics


def open_reader(bag_dir: Path):
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def count_pointcloud_messages(bag_dir: Path, pointcloud_topics):
    reader = open_reader(bag_dir)
    count = 0

    while reader.has_next():
        topic, _, _ = reader.read_next()
        if topic in pointcloud_topics:
            count += 1

    return count


def read_random_pointcloud_message(bag_dir: Path, pointcloud_topics, random_index):
    reader = open_reader(bag_dir)
    seen = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic not in pointcloud_topics:
            continue

        if seen == random_index:
            msg = deserialize_message(data, PointCloud2)
            return topic, msg, timestamp

        seen += 1

    return None, None, None


def pointcloud2_to_xyz_array(msg: PointCloud2):
    """
    Convert PointCloud2 to Nx3 xyz numpy array.
    Skips NaN points.
    """
    points = []

    for p in pc2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    ):
        points.append([p[0], p[1], p[2]])

    if not points:
        return np.empty((0, 3), dtype=np.float32)

    return np.asarray(points, dtype=np.float32)


def save_ascii_pcd(path: Path, xyz: np.ndarray):
    """
    Save xyz points as ASCII PCD.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {xyz.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {xyz.shape[0]}\n")
        f.write("DATA ascii\n")

        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def make_output_name(dataset_root: Path, bag_dir: Path):
    """
    Create a unique readable filename based on the bag path.
    Example:
    CERN_unitree_unilidar_L1_BA51_BA51.pcd
    """
    rel = bag_dir.relative_to(dataset_root)
    return "_".join(rel.parts) + ".pcd"


def main():
    parser = argparse.ArgumentParser(
        description="Extract one random PointCloud2 message from each ROS 2 bag and save it as PCD."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root dataset folder. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("random_pcds"),
        help="Folder where PCD files will be saved. Default: random_pcds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatable sampling.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Optional PointCloud2 topic name to use. If omitted, all PointCloud2 topics are considered.",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.seed is not None:
        random.seed(args.seed)

    bags = find_ros2_bags(dataset_root)

    if not bags:
        print(f"No ROS 2 bags found under: {dataset_root}")
        sys.exit(1)

    print(f"Found {len(bags)} ROS 2 bags")
    print(f"Saving PCDs to: {output_dir}")

    saved = 0
    skipped = 0

    for bag_dir in bags:
        metadata_path = bag_dir / "metadata.yaml"

        pointcloud_topics = get_pointcloud2_topics(metadata_path)

        if args.topic is not None:
            if args.topic in pointcloud_topics:
                pointcloud_topics = [args.topic]
            else:
                print(f"[SKIP] {bag_dir}: topic '{args.topic}' not found")
                skipped += 1
                continue

        if not pointcloud_topics:
            print(f"[SKIP] {bag_dir}: no sensor_msgs/msg/PointCloud2 topic found")
            skipped += 1
            continue

        try:
            num_clouds = count_pointcloud_messages(bag_dir, pointcloud_topics)

            if num_clouds == 0:
                print(f"[SKIP] {bag_dir}: PointCloud2 topics exist but contain no messages")
                skipped += 1
                continue

            random_index = random.randint(0, num_clouds - 1)

            topic, msg, timestamp = read_random_pointcloud_message(
                bag_dir,
                pointcloud_topics,
                random_index,
            )

            if msg is None:
                print(f"[SKIP] {bag_dir}: failed to read random cloud")
                skipped += 1
                continue

            xyz = pointcloud2_to_xyz_array(msg)

            if xyz.shape[0] == 0:
                print(f"[SKIP] {bag_dir}: selected cloud has zero valid xyz points")
                skipped += 1
                continue

            output_name = make_output_name(dataset_root, bag_dir)
            output_path = output_dir / output_name

            save_ascii_pcd(output_path, xyz)

            print(
                f"[OK] {bag_dir} | topic={topic} | points={xyz.shape[0]} | "
                f"msg_index={random_index}/{num_clouds - 1} | saved={output_path}"
            )

            saved += 1

        except Exception as e:
            print(f"[ERROR] {bag_dir}: {e}")
            skipped += 1

    print()
    print(f"Done. Saved: {saved}, skipped/errors: {skipped}")


if __name__ == "__main__":
    main()