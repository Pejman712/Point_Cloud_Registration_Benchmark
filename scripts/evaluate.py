#!/usr/bin/env python3
import argparse
import csv
import importlib
import re
import sqlite3
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


POINTCLOUD_TYPES = {
    "sensor_msgs/msg/PointCloud2",
}


def log(msg):
    print(msg, flush=True)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def append_summary(summary_csv, row):
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "lidar",
        "sequence",
        "method",
        "status",
        "error",
        "pointcloud_topic",
        "num_clouds",
        "num_registrations",
        "runtime_sec",
        "mean_fitness",
        "mean_rmse",
        "ape_rmse",
        "ape_mean",
        "ape_median",
        "ape_max",
        "rpe_rmse",
        "rpe_mean",
        "rpe_median",
        "rpe_max",
    ]

    exists = summary_csv.exists()

    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow({k: row.get(k, "") for k in fieldnames})


def find_sequences(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    sequences = []

    for db3 in dataset_root.rglob("*.db3"):
        bag_dir = db3.parent
        seq_dir = bag_dir.parent
        seq_name = seq_dir.name

        gt_files = list(seq_dir.glob("*.tum")) + list(seq_dir.glob("*.csv"))

        if not gt_files:
            log(f"[WARN] No ground truth found for sequence: {seq_dir}")
            continue

        sequences.append(
            {
                "name": seq_name,
                "dataset": seq_dir.parent.parent.name,
                "lidar": seq_dir.parent.name,
                "seq_dir": seq_dir,
                "bag_dir": bag_dir,
                "db3": db3,
                "groundtruth": gt_files[0],
            }
        )

    return sorted(sequences, key=lambda x: (x["dataset"], x["lidar"], x["name"]))


def filter_sequences(sequences, cfg):
    seq_cfg = cfg.get("sequences", {})
    mode = seq_cfg.get("mode", "all")
    only = seq_cfg.get("only", [])
    skip = seq_cfg.get("skip", [])

    if mode == "only":
        sequences = [s for s in sequences if s["name"] in only]

    if skip:
        sequences = [s for s in sequences if s["name"] not in skip]

    return sequences


def list_bag_topics(db3_path):
    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    cur.execute("SELECT id, name, type FROM topics")
    rows = cur.fetchall()

    conn.close()
    return rows


def get_pointcloud_topic(db3_path):
    rows = list_bag_topics(db3_path)

    log("[DEBUG] Bag topics:")
    for topic_id, name, typ in rows:
        log(f"  id={topic_id} topic={name} type={typ}")

    for topic_id, name, typ in rows:
        if typ in POINTCLOUD_TYPES:
            return topic_id, name, typ

    raise RuntimeError(f"No PointCloud2 topic found in {db3_path}")


def read_pointcloud_message_count(db3_path, topic_id):
    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,))
    count = cur.fetchone()[0]

    conn.close()
    return count


def pointcloud2_to_open3d(msg, max_points=0):
    import sensor_msgs_py.point_cloud2 as pc2

    points_raw = pc2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )

    if isinstance(points_raw, np.ndarray):
        if points_raw.dtype.names is not None:
            points = np.vstack(
                [
                    points_raw["x"],
                    points_raw["y"],
                    points_raw["z"],
                ]
            ).T
        else:
            points = np.asarray(points_raw)[:, :3]
    else:
        points = np.array(list(points_raw), dtype=np.float64)

    if points.size == 0:
        return o3d.geometry.PointCloud()

    points = np.asarray(points, dtype=np.float64)

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if max_points and max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    return cloud


def load_real_clouds_from_bag(
    bag_dir,
    topic_name,
    max_frames,
    frame_step,
    max_points_per_cloud,
    min_points_per_cloud,
):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as e:
        raise RuntimeError(
            "Could not import ROS 2 Python bag tools. "
            "Run: source /opt/ros/humble/setup.bash. "
            f"Original error: {e}"
        )

    bag_dir = Path(bag_dir)

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }

    if topic_name not in topic_types:
        raise RuntimeError(
            f"Topic {topic_name} not found in rosbag2 reader topic list. "
            f"Available: {list(topic_types.keys())}"
        )

    msg_type_name = topic_types[topic_name]
    msg_type = get_message(msg_type_name)

    clouds = []
    timestamps = []

    seen_pointcloud_msgs = 0
    kept_pointcloud_msgs = 0

    log(f"[DEBUG] Loading PointCloud2 from bag: {bag_dir}")
    log(f"[DEBUG] Topic: {topic_name}")
    log(f"[DEBUG] Type:  {msg_type_name}")

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic != topic_name:
            continue

        if seen_pointcloud_msgs % frame_step != 0:
            seen_pointcloud_msgs += 1
            continue

        try:
            msg = deserialize_message(data, msg_type)

            cloud = pointcloud2_to_open3d(
                msg,
                max_points=max_points_per_cloud,
            )

            n_points = np.asarray(cloud.points).shape[0]

            if n_points < min_points_per_cloud:
                log(
                    f"[DEBUG] Skipping frame {seen_pointcloud_msgs}: "
                    f"only {n_points} points"
                )
                seen_pointcloud_msgs += 1
                continue

            clouds.append(cloud)
            timestamps.append(timestamp_ns * 1e-9)
            kept_pointcloud_msgs += 1

            log(
                f"[DEBUG] Loaded cloud {kept_pointcloud_msgs}: "
                f"bag_frame={seen_pointcloud_msgs}, "
                f"timestamp={timestamp_ns * 1e-9:.9f}, "
                f"points={n_points}"
            )

        except Exception as e:
            log(f"[WARN] Failed to deserialize frame {seen_pointcloud_msgs}: {e}")

        seen_pointcloud_msgs += 1

        if len(clouds) >= max_frames:
            break

    log(f"[DEBUG] PointCloud2 messages seen: {seen_pointcloud_msgs}")
    log(f"[DEBUG] PointCloud2 clouds kept:  {len(clouds)}")

    if len(clouds) == 0:
        raise RuntimeError("No valid point clouds loaded from bag")

    return timestamps, clouds


def load_dummy_clouds_from_bag(db3_path, topic_id, max_frames, frame_step):
    n = min(read_pointcloud_message_count(db3_path, topic_id), max_frames)

    clouds = []
    timestamps = []

    base = np.random.randn(2000, 3).astype(np.float64)

    for i in range(0, n, frame_step):
        points = base.copy()
        points[:, 0] += 0.05 * i

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        clouds.append(cloud)
        timestamps.append(float(i))

    log("[WARN] Using dummy clouds. These are not real bag point clouds.")
    return timestamps, clouds


def load_clouds(seq, topic_id, topic_name, cfg):
    reg_cfg = cfg.get("registration", {})

    max_frames = int(reg_cfg.get("max_frames", 200))
    frame_step = int(reg_cfg.get("frame_step", 1))
    max_points_per_cloud = int(reg_cfg.get("max_points_per_cloud", 50000))
    min_points_per_cloud = int(reg_cfg.get("min_points_per_cloud", 100))
    allow_dummy_clouds = bool(reg_cfg.get("allow_dummy_clouds", False))

    try:
        return load_real_clouds_from_bag(
            bag_dir=seq["bag_dir"],
            topic_name=topic_name,
            max_frames=max_frames,
            frame_step=frame_step,
            max_points_per_cloud=max_points_per_cloud,
            min_points_per_cloud=min_points_per_cloud,
        )

    except Exception as e:
        log(f"[ERROR] Real bag loading failed: {e}")

        if allow_dummy_clouds:
            return load_dummy_clouds_from_bag(
                db3_path=seq["db3"],
                topic_id=topic_id,
                max_frames=max_frames,
                frame_step=frame_step,
            )

        raise


def split_row(line):
    return [p for p in re.split(r"[,\s;]+", line.strip()) if p != ""]


def normalize_quaternion(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)

    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0

    q /= n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def try_parse_pose_values(vals, seq, row_idx):
    """
    Dataset-specific ground-truth parser.

    Tier CSV:
      x y z qx qy qz qw
      no timestamp

    iilab_benchmark:
      timestamp x y z qx qy qz qw

    CERN:
      timestamp x y z qx qy qz qw
    """

    dataset_name = seq["dataset"]

    if dataset_name == "Tier":
        if len(vals) < 7:
            return None

        x, y, z = vals[0], vals[1], vals[2]
        qx, qy, qz, qw = vals[3], vals[4], vals[5], vals[6]
        qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

        return {
            "timestamp": float(row_idx),
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "format": "tier_xyz_qx_qy_qz_qw_no_timestamp",
            "has_real_timestamp": False,
        }

    if dataset_name in {"CERN", "iilab_benchmark"}:
        if len(vals) < 8:
            return None

        timestamp = vals[0]
        x, y, z = vals[1], vals[2], vals[3]
        qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
        qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

        return {
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "format": "timestamp_xyz_qx_qy_qz_qw",
            "has_real_timestamp": True,
        }

    if len(vals) >= 8:
        timestamp = vals[0]
        x, y, z = vals[1], vals[2], vals[3]
        qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
        qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

        return {
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "z": z,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "format": "generic_timestamp_xyz_qx_qy_qz_qw",
            "has_real_timestamp": True,
        }

    return None


def debug_groundtruth_file(gt_path):
    gt_path = Path(gt_path)

    log(f"[DEBUG] Ground truth file: {gt_path}")

    if not gt_path.exists():
        log("[DEBUG] Ground truth file does not exist")
        return

    with open(gt_path, "r", errors="replace") as f:
        lines = f.readlines()

    log(f"[DEBUG] Ground truth total lines: {len(lines)}")

    shown = 0
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            continue

        parts = split_row(stripped)
        log(f"[DEBUG] GT sample line {i}: columns={len(parts)} text={stripped[:160]}")

        shown += 1
        if shown >= 5:
            break


def parse_groundtruth_to_tum(gt_path, output_tum, seq):
    """
    Always parses ALL ground-truth poses.

    This is important for Tier because GT can have thousands of poses while
    the loaded point cloud count can be much smaller.
    """
    gt_path = Path(gt_path)
    output_tum = Path(output_tum)
    output_tum.parent.mkdir(parents=True, exist_ok=True)

    debug_groundtruth_file(gt_path)

    poses = []
    skipped_header_or_bad = 0
    skipped_unrecognized = 0
    formats = {}
    has_real_timestamps = True

    with open(gt_path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = split_row(line)

            try:
                vals = [float(p) for p in parts]
            except Exception:
                skipped_header_or_bad += 1
                continue

            row_idx = len(poses)

            pose = try_parse_pose_values(
                vals=vals,
                seq=seq,
                row_idx=row_idx,
            )

            if pose is None:
                skipped_unrecognized += 1
                if skipped_unrecognized <= 5:
                    log(f"[DEBUG] Could not parse GT line {line_no}: {line[:160]}")
                continue

            poses.append(pose)
            formats[pose["format"]] = formats.get(pose["format"], 0) + 1

            if not pose["has_real_timestamp"]:
                has_real_timestamps = False

    if len(poses) == 0:
        raise RuntimeError(
            f"No valid ground-truth poses parsed from {gt_path}. "
            f"Skipped nonnumeric/header lines={skipped_header_or_bad}, "
            f"skipped numeric-but-unrecognized lines={skipped_unrecognized}."
        )

    if has_real_timestamps:
        poses = sorted(poses, key=lambda p: p["timestamp"])

    with open(output_tum, "w") as f:
        for p in poses:
            f.write(
                f"{p['timestamp']:.9f} "
                f"{p['x']:.9f} {p['y']:.9f} {p['z']:.9f} "
                f"{p['qx']:.9f} {p['qy']:.9f} {p['qz']:.9f} {p['qw']:.9f}\n"
            )

    log(f"[DEBUG] Parsed GT poses: {len(poses)}")
    log(f"[DEBUG] GT formats: {formats}")
    log(f"[DEBUG] GT has real timestamps: {has_real_timestamps}")
    log(f"[DEBUG] Wrote normalized GT: {output_tum}")

    gt_timestamps = [p["timestamp"] for p in poses]

    return output_tum, has_real_timestamps, gt_timestamps


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    return normalize_quaternion(qx, qy, qz, qw)


def matrix_to_tum_line(timestamp, T):
    R = T[:3, :3]
    t = T[:3, 3]

    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

    return (
        f"{timestamp:.9f} "
        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
    )


def validate_method_module(method_module):
    if not hasattr(method_module, "METHOD_NAME"):
        raise RuntimeError("Method module is missing METHOD_NAME")

    if not hasattr(method_module, "register_clouds"):
        raise RuntimeError("Method module is missing register_clouds()")


def validate_registration_result(result, method_name, frame_idx):
    if result is None:
        raise RuntimeError(
            f"{method_name}.register_clouds() returned None at frame {frame_idx}"
        )

    if isinstance(result, np.ndarray):
        result = {
            "transformation": result,
            "fitness": np.nan,
            "rmse": np.nan,
            "success": True,
        }

    if not isinstance(result, dict):
        raise RuntimeError(
            f"{method_name}.register_clouds() must return dict or 4x4 ndarray. "
            f"Got {type(result)} at frame {frame_idx}"
        )

    if "transformation" not in result:
        raise RuntimeError(
            f"{method_name}.register_clouds() result has no 'transformation' "
            f"key at frame {frame_idx}. Keys: {list(result.keys())}"
        )

    T = np.asarray(result["transformation"], dtype=float)

    if T.shape != (4, 4):
        raise RuntimeError(
            f"{method_name}.register_clouds() returned transformation with "
            f"shape {T.shape}, expected (4, 4), at frame {frame_idx}"
        )

    if not np.isfinite(T).all():
        raise RuntimeError(
            f"{method_name}.register_clouds() returned non-finite transform "
            f"at frame {frame_idx}"
        )

    result["transformation"] = T

    if "fitness" not in result:
        result["fitness"] = np.nan

    if "rmse" not in result:
        result["rmse"] = np.nan

    if "success" not in result:
        result["success"] = True

    return result


def run_registration_sequence(method_module, timestamps, clouds, output_tum, cfg):
    if len(clouds) < 2:
        raise RuntimeError(f"Need at least 2 clouds for registration, got {len(clouds)}")

    if len(timestamps) < len(clouds):
        raise RuntimeError(
            f"Need at least as many timestamps as clouds. "
            f"timestamps={len(timestamps)}, clouds={len(clouds)}"
        )

    method_name = method_module.METHOD_NAME
    reg_cfg = cfg.get("registration", {})

    print_every = int(reg_cfg.get("print_every_n_frames", 1))
    invert_result_transform = bool(reg_cfg.get("invert_result_transform", False))

    pose = np.eye(4)
    poses = [pose.copy()]
    stats = []

    log(f"[DEBUG] Starting registration with {method_name}")
    log(f"[DEBUG] Number of clouds: {len(clouds)}")
    log(f"[DEBUG] invert_result_transform: {invert_result_transform}")

    for i in range(1, len(clouds)):
        source = clouds[i]
        target = clouds[i - 1]

        source_points = np.asarray(source.points).shape[0]
        target_points = np.asarray(target.points).shape[0]

        if print_every > 0 and (i == 1 or i % print_every == 0):
            log(
                f"[DEBUG] Frame {i}/{len(clouds)-1}: "
                f"source_points={source_points}, target_points={target_points}"
            )

        start = time.time()

        result_raw = method_module.register_clouds(
            source_cloud=source,
            target_cloud=target,
            init_guess=np.eye(4),
        )

        result = validate_registration_result(result_raw, method_name, i)

        T_source_to_target = result["transformation"]

        if invert_result_transform:
            relative_motion = invert_transform(T_source_to_target)
        else:
            relative_motion = T_source_to_target

        pose = pose @ relative_motion

        poses.append(pose.copy())
        stats.append(result)

        dt = time.time() - start
        trans = T_source_to_target[:3, 3]
        trans_norm = float(np.linalg.norm(trans))

        if print_every > 0 and (i == 1 or i % print_every == 0):
            log(
                f"[DEBUG] Frame {i} result: "
                f"success={result.get('success')}, "
                f"fitness={result.get('fitness')}, "
                f"rmse={result.get('rmse')}, "
                f"translation_norm={trans_norm:.6f}, "
                f"time={dt:.3f}s"
            )

    with open(output_tum, "w") as f:
        for i, T in enumerate(poses):
            f.write(matrix_to_tum_line(timestamps[i], T))

    log(f"[DEBUG] Wrote estimated trajectory: {output_tum}")
    log(f"[DEBUG] Estimated poses written: {len(poses)}")

    return stats


def run_evo(gt_tum, est_tum, result_dir, cfg):
    result_dir = Path(result_dir)
    eval_cfg = cfg.get("evaluation", {})

    align = bool(eval_cfg.get("align", True))
    t_max_diff = eval_cfg.get("t_max_diff", None)

    ape_log = result_dir / "ape.log"
    rpe_log = result_dir / "rpe.log"

    align_flag = "--align" if align else ""
    t_max_diff_flag = f"--t_max_diff {t_max_diff}" if t_max_diff is not None else ""

    ape_cmd = f"evo_ape tum {gt_tum} {est_tum} {align_flag} {t_max_diff_flag}"
    rpe_cmd = f"evo_rpe tum {gt_tum} {est_tum} {align_flag} {t_max_diff_flag}"

    metrics = {}

    log(f"[DEBUG] Running: {ape_cmd}")
    with open(ape_log, "w") as f:
        ape_proc = subprocess.run(
            ape_cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    log(f"[DEBUG] Running: {rpe_cmd}")
    with open(rpe_log, "w") as f:
        rpe_proc = subprocess.run(
            rpe_cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    metrics["ape_returncode"] = ape_proc.returncode
    metrics["rpe_returncode"] = rpe_proc.returncode

    if ape_proc.returncode != 0:
        log(f"[WARN] evo_ape failed. Check: {ape_log}")

    if rpe_proc.returncode != 0:
        log(f"[WARN] evo_rpe failed. Check: {rpe_log}")

    metrics.update(parse_evo_log(ape_log, "ape"))
    metrics.update(parse_evo_log(rpe_log, "rpe"))

    return metrics


def parse_evo_log(path, prefix):
    metrics = {}

    if not Path(path).exists():
        return metrics

    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 2:
                continue

            key, value = parts

            if key in ["rmse", "mean", "median", "std", "min", "max"]:
                try:
                    metrics[f"{prefix}_{key}"] = float(value)
                except Exception:
                    pass

    return metrics


def read_tum_xy(path):
    xy = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = split_row(line)

            if len(parts) < 4:
                continue

            try:
                x = float(parts[1])
                y = float(parts[2])
            except Exception:
                continue

            xy.append([x, y])

    return np.asarray(xy, dtype=float)


def print_xy_stats(name, xy):
    if len(xy) == 0:
        log(f"[DEBUG] {name}: empty")
        return

    log(
        f"[DEBUG] {name}: "
        f"points={len(xy)}, "
        f"x=[{xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}], "
        f"y=[{xy[:, 1].min():.3f}, {xy[:, 1].max():.3f}], "
        f"start=({xy[0, 0]:.3f}, {xy[0, 1]:.3f}), "
        f"end=({xy[-1, 0]:.3f}, {xy[-1, 1]:.3f})"
    )


def center_xy_at_start(xy):
    if len(xy) == 0:
        return xy

    return xy - xy[0]


def plot_sequence_all_methods_xy(gt_tum, method_trajectories, output_png, title):
    gt_xy_raw = read_tum_xy(gt_tum)

    print_xy_stats("ground_truth_raw_FULL", gt_xy_raw)

    if len(gt_xy_raw) == 0:
        raise RuntimeError(f"No valid ground-truth XY trajectory: {gt_tum}")

    gt_xy = center_xy_at_start(gt_xy_raw)

    print_xy_stats("ground_truth_centered_FULL", gt_xy)

    plt.figure(figsize=(9, 9))

    plt.plot(
        gt_xy[:, 0],
        gt_xy[:, 1],
        linewidth=3,
        linestyle="--",
        label=f"ground_truth_all_{len(gt_xy)}_poses",
        zorder=10,
    )

    plotted_methods = 0

    for method_name, est_tum in method_trajectories.items():
        est_tum = Path(est_tum)

        if not est_tum.exists():
            log(f"[DEBUG] Missing estimated TUM for plot: {est_tum}")
            continue

        est_xy_raw = read_tum_xy(est_tum)

        print_xy_stats(f"{method_name}_raw", est_xy_raw)

        if len(est_xy_raw) == 0:
            continue

        est_xy = center_xy_at_start(est_xy_raw)

        print_xy_stats(f"{method_name}_centered", est_xy)

        plt.plot(
            est_xy[:, 0],
            est_xy[:, 1],
            linewidth=2,
            label=f"{method_name}_{len(est_xy)}_poses",
        )

        plotted_methods += 1

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title + " | full GT, start-aligned XY")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()

    log(f"[DEBUG] Plotted methods: {plotted_methods}")


def safe_lidar_name(name):
    return name.replace(" ", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/evaluate.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_root = Path(cfg.get("dataset_root", "dataset"))
    results_root = Path(cfg.get("results_root", "results"))

    sequences = find_sequences(dataset_root)
    sequences = filter_sequences(sequences, cfg)

    methods = cfg.get("methods", [])

    if not methods:
        raise RuntimeError("No methods listed in evaluate.yaml")

    log(f"Sequences: {len(sequences)}")
    log(f"Methods:   {len(methods)}")

    summary_csv = results_root / "summary.csv"

    for seq in sequences:
        log("=" * 90)
        log(f"Sequence: {seq['dataset']} / {seq['lidar']} / {seq['name']}")
        log(f"Bag dir:  {seq['bag_dir']}")
        log(f"DB3:      {seq['db3']}")
        log(f"GT:       {seq['groundtruth']}")

        try:
            topic_id, topic_name, topic_type = get_pointcloud_topic(seq["db3"])
            topic_count = read_pointcloud_message_count(seq["db3"], topic_id)

            log(f"PointCloud topic: {topic_name}")
            log(f"PointCloud type:  {topic_type}")
            log(f"PointCloud msgs:  {topic_count}")

            timestamps, clouds = load_clouds(
                seq=seq,
                topic_id=topic_id,
                topic_name=topic_name,
                cfg=cfg,
            )

            log(f"[DEBUG] Loaded clouds: {len(clouds)}")

        except Exception as e:
            log(f"[ERROR] Sequence loading failed: {e}")
            traceback.print_exc()
            continue

        sequence_plot_dir = (
            results_root
            / seq["dataset"]
            / safe_lidar_name(seq["lidar"])
            / seq["name"]
        )
        sequence_plot_dir.mkdir(parents=True, exist_ok=True)

        try:
            gt_tum, gt_has_real_timestamps, gt_timestamps = parse_groundtruth_to_tum(
                gt_path=seq["groundtruth"],
                output_tum=sequence_plot_dir / "groundtruth.tum",
                seq=seq,
            )

            if gt_has_real_timestamps:
                trajectory_timestamps = timestamps[:len(clouds)]
            else:
                trajectory_timestamps = [float(i) for i in range(len(clouds))]

            if len(trajectory_timestamps) < len(clouds):
                raise RuntimeError(
                    f"Not enough trajectory timestamps. "
                    f"timestamps={len(trajectory_timestamps)}, clouds={len(clouds)}"
                )

        except Exception as e:
            log(f"[ERROR] Ground truth parsing failed: {e}")
            traceback.print_exc()
            continue

        method_trajectories = {}

        for method_path in methods:
            result_dir = None

            try:
                log("-" * 90)
                log(f"Importing method: {method_path}")

                method_module = importlib.import_module(method_path)
                validate_method_module(method_module)

                method_name = method_module.METHOD_NAME

                result_dir = sequence_plot_dir / method_name
                result_dir.mkdir(parents=True, exist_ok=True)

                output_tum = result_dir / "estimated.tum"

                log(f"Running method: {method_name}")

                start = time.time()

                stats = run_registration_sequence(
                    method_module=method_module,
                    timestamps=trajectory_timestamps,
                    clouds=clouds,
                    output_tum=output_tum,
                    cfg=cfg,
                )

                runtime = time.time() - start

                fitness_values = [
                    float(s.get("fitness"))
                    for s in stats
                    if s.get("fitness") is not None and np.isfinite(s.get("fitness"))
                ]

                rmse_values = [
                    float(s.get("rmse"))
                    for s in stats
                    if s.get("rmse") is not None and np.isfinite(s.get("rmse"))
                ]

                metrics = {
                    "dataset": seq["dataset"],
                    "lidar": seq["lidar"],
                    "sequence": seq["name"],
                    "method": method_name,
                    "status": "ok",
                    "error": "",
                    "runtime_sec": float(runtime),
                    "pointcloud_topic": topic_name,
                    "num_clouds": len(clouds),
                    "num_registrations": len(stats),
                    "mean_fitness": float(np.mean(fitness_values)) if fitness_values else "",
                    "mean_rmse": float(np.mean(rmse_values)) if rmse_values else "",
                    "estimated_tum": str(output_tum),
                    "groundtruth_tum": str(gt_tum),
                    "gt_has_real_timestamps": bool(gt_has_real_timestamps),
                    "gt_poses_total": len(gt_timestamps),
                }

                if cfg.get("evaluation", {}).get("use_evo", True):
                    evo_metrics = run_evo(
                        gt_tum=gt_tum,
                        est_tum=output_tum,
                        result_dir=result_dir,
                        cfg=cfg,
                    )
                    metrics.update(evo_metrics)

                method_trajectories[method_name] = output_tum

                log("Status: ok")
                log(f"Estimated trajectory: {output_tum}")

            except Exception as e:
                err = str(e)
                tb = traceback.format_exc()

                log(f"[ERROR] Method failed: {method_path}")
                log(f"[ERROR] {err}")
                log(tb)

                method_name = Path(method_path).name

                if result_dir is None:
                    result_dir = sequence_plot_dir / method_name
                    result_dir.mkdir(parents=True, exist_ok=True)

                metrics = {
                    "dataset": seq["dataset"],
                    "lidar": seq["lidar"],
                    "sequence": seq["name"],
                    "method": method_name,
                    "status": "failed",
                    "error": err,
                    "traceback": tb,
                    "pointcloud_topic": topic_name,
                    "num_clouds": len(clouds),
                }

            save_yaml(result_dir / "metrics.yaml", metrics)
            append_summary(summary_csv, metrics)

        plot_path = sequence_plot_dir / "trajectory_xy_all_methods.png"

        try:
            plot_sequence_all_methods_xy(
                gt_tum=gt_tum,
                method_trajectories=method_trajectories,
                output_png=plot_path,
                title=f"{seq['dataset']} / {seq['lidar']} / {seq['name']}",
            )
            log(f"Saved XY plot: {plot_path}")
        except Exception as e:
            log(f"[ERROR] Plot failed: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()