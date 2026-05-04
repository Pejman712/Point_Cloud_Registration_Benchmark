#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import yaml


POINTCLOUD_TYPES = {
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/msg/PointCloud",
}


EXPECTED_LIDAR_HINTS = {
    "unitree_unilidar_L1": ["point", "cloud", "unilidar", "lidar"],
    "livox_mid-360": ["livox", "point", "cloud", "lidar"],
    "Livox_ avia": ["livox", "avia", "point", "cloud", "lidar"],
    "Livox_horizen": ["livox", "horizon", "horizen", "point", "cloud", "lidar"],
}


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def find_sequences(root: Path):
    sequences = []

    for db3 in root.rglob("*.db3"):
        bag_inner_dir = db3.parent
        seq_dir = bag_inner_dir.parent
        seq_name = seq_dir.name

        metadata = bag_inner_dir / "metadata.yaml"

        gt_candidates = []
        gt_candidates.extend(seq_dir.glob("*.tum"))
        gt_candidates.extend(seq_dir.glob("*.csv"))
        gt_candidates.extend(seq_dir.glob("*.txt"))

        sequences.append(
            {
                "seq_name": seq_name,
                "seq_dir": seq_dir,
                "bag_dir": bag_inner_dir,
                "db3": db3,
                "metadata": metadata,
                "groundtruth_files": gt_candidates,
                "lidar_name": seq_dir.parent.name,
                "dataset_name": seq_dir.parent.parent.name if seq_dir.parent.parent else None,
            }
        )

    return sequences


def check_metadata(metadata_path: Path):
    if not metadata_path.exists():
        fail(f"Missing metadata.yaml: {metadata_path}")
        return None

    try:
        with open(metadata_path, "r") as f:
            data = yaml.safe_load(f)
        ok(f"metadata.yaml readable: {metadata_path}")
        return data
    except Exception as e:
        fail(f"metadata.yaml not readable: {metadata_path}: {e}")
        return None


def check_db3(db3_path: Path):
    if not db3_path.exists():
        fail(f"Missing db3: {db3_path}")
        return False, []

    try:
        conn = sqlite3.connect(str(db3_path))
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}

        required = {"topics", "messages"}
        missing = required - tables
        if missing:
            fail(f"Bag database missing tables {missing}: {db3_path}")
            conn.close()
            return False, []

        cur.execute("SELECT COUNT(*) FROM messages")
        n_messages = cur.fetchone()[0]

        cur.execute("SELECT name, type FROM topics")
        topics = cur.fetchall()

        conn.close()

        if n_messages == 0:
            fail(f"Bag has zero messages: {db3_path}")
            return False, topics

        ok(f"Bag readable with {n_messages} messages: {db3_path}")
        return True, topics

    except Exception as e:
        fail(f"Cannot read db3 bag: {db3_path}: {e}")
        return False, []


def check_pointcloud_topics(topics, lidar_name):
    if not topics:
        fail("No topics found")
        return []

    pc_topics = [(name, typ) for name, typ in topics if typ in POINTCLOUD_TYPES]

    if not pc_topics:
        fail("No PointCloud2 / PointCloud topic found")
        print("  Available topics:")
        for name, typ in topics:
            print(f"    {name}: {typ}")
        return []

    ok("Point cloud topics found:")
    for name, typ in pc_topics:
        print(f"    {name}: {typ}")

    hints = EXPECTED_LIDAR_HINTS.get(lidar_name, ["point", "cloud", "lidar"])
    topic_names = [name.lower() for name, _ in pc_topics]

    matched = any(any(h.lower() in t for h in hints) for t in topic_names)

    if matched:
        ok(f"Point cloud topic looks plausible for lidar '{lidar_name}'")
    else:
        warn(f"Point cloud topic exists, but name does not clearly match lidar '{lidar_name}'")

    return pc_topics


def parse_numeric_row(line):
    line = line.strip()

    if not line:
        return None

    if line.startswith("#"):
        return None

    parts = re.split(r"[,\s;]+", line)

    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(float(p))
        except ValueError:
            return "header_or_bad"

    return vals


def quaternion_norm(vals):
    return math.sqrt(sum(v * v for v in vals))


def classify_pose_row(vals):
    """
    Supported examples:

    TUM:
      timestamp x y z qx qy qz qw

    ROS time + quaternion:
      sec nsec x y z qx qy qz qw

    Euler:
      timestamp x y z roll pitch yaw

    Extra columns are allowed.
    """
    if len(vals) < 7:
        return None

    if len(vals) >= 8:
        q = vals[4:8]
        qnorm = quaternion_norm(q)
        if 0.5 <= qnorm <= 1.5:
            return "timestamp_xyz_quaternion"

    if len(vals) >= 9:
        q = vals[5:9]
        qnorm = quaternion_norm(q)
        if 0.5 <= qnorm <= 1.5:
            return "sec_nsec_xyz_quaternion"

    if len(vals) >= 7:
        return "timestamp_xyz_euler_or_extra"

    return None


def check_groundtruth_file(path: Path):
    valid = 0
    bad = []
    timestamps = []
    formats = {}

    try:
        with open(path, "r") as f:
            for i, line in enumerate(f, start=1):
                parsed = parse_numeric_row(line)

                if parsed is None:
                    continue

                if parsed == "header_or_bad":
                    continue

                pose_format = classify_pose_row(parsed)

                if pose_format is None:
                    bad.append((i, line.strip()))
                    continue

                formats[pose_format] = formats.get(pose_format, 0) + 1
                timestamps.append(parsed[0])
                valid += 1

    except Exception as e:
        fail(f"Cannot read ground truth file {path}: {e}")
        return False

    if valid == 0:
        fail(f"Ground truth has no valid pose rows: {path}")
        return False

    if bad:
        warn(f"Ground truth has {len(bad)} suspicious rows: {path}")
        for line_no, text in bad[:5]:
            print(f"    line {line_no}: {text}")

    if timestamps != sorted(timestamps):
        warn(f"Ground truth timestamps are not monotonically increasing: {path}")
    else:
        ok(f"Ground truth timestamps are monotonic: {path}")

    ok(f"Ground truth valid with {valid} pose rows: {path}")
    print("  Detected pose formats:")
    for fmt, count in formats.items():
        print(f"    {fmt}: {count}")

    return True


def check_groundtruth(seq):
    gt_files = seq["groundtruth_files"]

    if not gt_files:
        fail(f"No ground truth file found for sequence: {seq['seq_name']}")
        return False

    all_ok = True

    for gt in gt_files:
        suffix = gt.suffix.lower()

        if suffix not in {".tum", ".csv", ".txt"}:
            continue

        result = check_groundtruth_file(gt)
        all_ok = all_ok and result

    return all_ok


def ros2_bag_info(bag_dir: Path):
    try:
        result = subprocess.run(
            ["ros2", "bag", "info", str(bag_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )

        if result.returncode == 0:
            ok(f"ros2 bag info works: {bag_dir}")
            return True

        fail(f"ros2 bag info failed: {bag_dir}")
        print(result.stderr.strip())
        return False

    except FileNotFoundError:
        warn("ros2 command not found. Source ROS 2 first, for example:")
        print("  source /opt/ros/humble/setup.bash")
        return False

    except Exception as e:
        fail(f"ros2 bag info error for {bag_dir}: {e}")
        return False


def check_topic_message_counts(db3_path: Path, topics):
    """
    Counts messages per topic from SQLite directly.
    """
    try:
        conn = sqlite3.connect(str(db3_path))
        cur = conn.cursor()

        cur.execute("SELECT id, name, type FROM topics")
        topic_rows = cur.fetchall()

        topic_id_to_info = {
            row[0]: {
                "name": row[1],
                "type": row[2],
                "count": 0,
            }
            for row in topic_rows
        }

        cur.execute("SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id")
        for topic_id, count in cur.fetchall():
            if topic_id in topic_id_to_info:
                topic_id_to_info[topic_id]["count"] = count

        conn.close()

        print("  Topic message counts:")
        for info in topic_id_to_info.values():
            print(f"    {info['name']}: {info['count']} messages [{info['type']}]")

        zero_topics = [i for i in topic_id_to_info.values() if i["count"] == 0]
        if zero_topics:
            warn("Some topics have zero messages")
        else:
            ok("All listed topics have at least one message")

        return topic_id_to_info

    except Exception as e:
        warn(f"Could not count messages per topic: {e}")
        return {}


def visualize_bag(bag_dir: Path, pc_topic: str):
    print()
    print("Visualization test")
    print("------------------")
    print(f"Bag: {bag_dir}")
    print(f"Point cloud topic: {pc_topic}")
    print()
    print("RViz will open. Add the PointCloud2 display manually if needed.")
    print("Use the topic shown above.")

    rviz = None
    play = None

    try:
        rviz = subprocess.Popen(["rviz2"])
        time.sleep(3)

        play = subprocess.Popen(
            [
                "ros2",
                "bag",
                "play",
                str(bag_dir),
                "--clock",
                "--rate",
                "0.5",
            ]
        )

        print()
        print("In RViz:")
        print("  Add -> By topic -> select the point cloud topic")
        print("  Try Fixed Frame: map, odom, base_link, or the lidar frame")
        print()
        print("Press Ctrl+C in this terminal to stop visualization.")

        play.wait()

    except KeyboardInterrupt:
        pass

    finally:
        if play:
            play.terminate()
        if rviz:
            rviz.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()

    if not root.exists():
        fail(f"Dataset root does not exist: {root}")
        return

    sequences = find_sequences(root)

    if args.only:
        sequences = [
            s for s in sequences if args.only.lower() in s["seq_name"].lower()
        ]

    if not sequences:
        fail("No .db3 bags found")
        return

    print(f"Found {len(sequences)} bag sequences")
    print()

    summary = {
        "total": 0,
        "metadata_ok": 0,
        "db3_ok": 0,
        "ros2_info_ok": 0,
        "pointcloud_ok": 0,
        "groundtruth_ok": 0,
    }

    first_visualizable = None

    for seq in sequences:
        summary["total"] += 1

        print("=" * 90)
        print(f"Sequence: {seq['seq_name']}")
        print(f"Dataset:  {seq['dataset_name']}")
        print(f"LiDAR:    {seq['lidar_name']}")
        print(f"Bag dir:  {seq['bag_dir']}")
        print(f"DB3:      {seq['db3']}")
        print()

        metadata = check_metadata(seq["metadata"])
        if metadata is not None:
            summary["metadata_ok"] += 1

        db_ok, topics = check_db3(seq["db3"])
        if db_ok:
            summary["db3_ok"] += 1

        check_topic_message_counts(seq["db3"], topics)

        if ros2_bag_info(seq["bag_dir"]):
            summary["ros2_info_ok"] += 1

        pc_topics = check_pointcloud_topics(topics, seq["lidar_name"])
        if pc_topics:
            summary["pointcloud_ok"] += 1

            if first_visualizable is None:
                first_visualizable = (seq["bag_dir"], pc_topics[0][0])

        if check_groundtruth(seq):
            summary["groundtruth_ok"] += 1

        print()

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total sequences:             {summary['total']}")
    print(f"metadata.yaml OK:            {summary['metadata_ok']}/{summary['total']}")
    print(f"SQLite db3 readable:         {summary['db3_ok']}/{summary['total']}")
    print(f"ros2 bag info OK:            {summary['ros2_info_ok']}/{summary['total']}")
    print(f"Point cloud topic OK:        {summary['pointcloud_ok']}/{summary['total']}")
    print(f"Ground truth OK:             {summary['groundtruth_ok']}/{summary['total']}")

    if args.visualize:
        if first_visualizable:
            visualize_bag(first_visualizable[0], first_visualizable[1])
        else:
            fail("No visualizable point cloud topic found")


if __name__ == "__main__":
    main()