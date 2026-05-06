#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import struct
import urllib.error
import urllib.request

import numpy as np
import open3d as o3d
import open3d.ml as _ml3d


S3DIS_CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

S3DIS_COLORS = np.array(
    [
        [0.65, 0.65, 0.65],  # ceiling
        [0.35, 0.35, 0.35],  # floor
        [0.80, 0.40, 0.40],  # wall
        [0.60, 0.30, 0.30],  # beam
        [0.60, 0.60, 0.20],  # column
        [0.20, 0.60, 0.80],  # window
        [0.80, 0.60, 0.20],  # door
        [0.40, 0.80, 0.40],  # table
        [0.20, 0.80, 0.20],  # chair
        [0.80, 0.20, 0.80],  # sofa
        [0.50, 0.30, 0.80],  # bookcase
        [0.20, 0.30, 0.80],  # board
        [0.90, 0.90, 0.20],  # clutter
    ],
    dtype=np.float32,
)

KNOWN_MD5 = {
    "randlanet_s3dis_202201071330utc.pth": None,
    "randlanet_s3dis_202010091238.pth": "5f993ef4a52065e4f882764568e6f378",
    "randlanet_s3dis_area5_202010091333utc.pth": "44e45c8244a70e899065d1c1c4719b73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a ROS 2 bag PointCloud2 topic, segment each point cloud with "
            "Open3D-ML RandLA-Net, and visualize the segmented result in Open3D."
        )
    )

    parser.add_argument(
        "--bag",
        required=True,
        help="Path to ROS 2 bag directory or bag file.",
    )

    parser.add_argument(
        "--topic",
        required=False,
        default=None,
        help="PointCloud2 topic to segment, e.g. /points_raw or /cloud_registered.",
    )

    parser.add_argument(
        "--list_topics",
        action="store_true",
        help="List topics in the ROS 2 bag and exit.",
    )

    parser.add_argument(
        "--storage_id",
        default="sqlite3",
        choices=["sqlite3", "mcap"],
        help="ROS 2 bag storage backend.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save segmented .pcd and .npz files per frame.",
    )

    parser.add_argument(
        "--open3d_ml_root",
        required=True,
        help="Path to cloned Open3D-ML repo, e.g. ../../lib/Open3D-ML",
    )

    parser.add_argument(
        "--cfg_file",
        default=None,
        help="Open3D-ML config file. Default: <open3d_ml_root>/ml3d/configs/randlanet_s3dis.yml",
    )

    parser.add_argument(
        "--ckpt",
        "--ckpt_path",
        default="./weights/randlanet_s3dis_202201071330utc.pth",
        help="Path to RandLA-Net S3DIS .pth checkpoint.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel downsampling size. Use 0 to disable.",
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=45056,
        help="RandLA-Net number of sampled points. Lower is lighter on CPU.",
    )

    parser.add_argument(
        "--no_auto_download",
        action="store_true",
        help="Disable automatic checkpoint download.",
    )

    parser.add_argument(
        "--feature_mode",
        choices=["auto", "zeros", "geometry"],
        default="auto",
        help=(
            "Feature fallback when PointCloud2 has no RGB. "
            "auto=RGB if available else geometry; zeros=zero RGB; geometry=height/range/ones."
        ),
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum number of point cloud frames to process. Use 0 for all frames.",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Skip point cloud frames before this index.",
    )

    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Process every Nth point cloud frame.",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize segmented point clouds in Open3D.",
    )

    parser.add_argument(
        "--visualize_each_frame",
        action="store_true",
        help="Open a labeled Open3D window for each processed frame. Close the window to continue.",
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Open3D visualization point size.",
    )

    parser.add_argument(
        "--axis_size",
        type=float,
        default=1.0,
        help="Coordinate-frame axis size in Open3D visualization. Use 0 to hide.",
    )

    return parser.parse_args()


def md5sum(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_md5_if_known(path):
    filename = os.path.basename(path)
    expected = KNOWN_MD5.get(filename)

    if filename not in KNOWN_MD5:
        print(f"No MD5 entry for {filename}; skipping checksum verification.")
        return

    if expected is None:
        print(f"No fixed MD5 stored for {filename}; skipping checksum verification.")
        return

    actual = md5sum(path)

    if actual != expected:
        raise RuntimeError(
            "Checkpoint MD5 mismatch.\n"
            f"  File:     {path}\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            "Delete the file and download it again."
        )

    print(f"Checkpoint MD5 OK: {filename}")


def download_file(url, output_path, timeout=60):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Trying checkpoint URL:\n  {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                print(f"  Failed: HTTP {response.status}")
                return False

            tmp_path = output_path + ".tmp"

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)

            size = os.path.getsize(tmp_path)

            if size < 1024 * 1024:
                os.remove(tmp_path)
                print("  Failed: downloaded file is too small.")
                return False

            os.replace(tmp_path, output_path)
            print(f"Downloaded checkpoint:\n  {output_path}")
            return True

    except urllib.error.HTTPError as e:
        print(f"  Failed: HTTP {e.code}")
        return False

    except urllib.error.URLError as e:
        print(f"  Failed: URL error: {e}")
        return False

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def find_checkpoint_url_from_model_zoo(open3d_ml_root, filename):
    model_zoo_path = os.path.join(open3d_ml_root, "model_zoo.md")

    if not os.path.isfile(model_zoo_path):
        return None

    with open(model_zoo_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"\((https?://[^)]*" + re.escape(filename) + r"[^)]*)\)"
    match = re.search(pattern, text)

    if match:
        return match.group(1)

    pattern = r"https?://\S*" + re.escape(filename)
    match = re.search(pattern, text)

    if match:
        return match.group(0).rstrip(")")

    return None


def auto_download_checkpoint(open3d_ml_root, ckpt_path):
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 1024 * 1024:
        print(f"Checkpoint already exists:\n  {ckpt_path}")
        return

    filename = os.path.basename(ckpt_path)

    urls = []

    url_from_model_zoo = find_checkpoint_url_from_model_zoo(open3d_ml_root, filename)

    if url_from_model_zoo:
        urls.append(url_from_model_zoo)

    urls.append(
        "https://storage.googleapis.com/open3d-releases/model-zoo/"
        "randlanet_s3dis_202201071330utc.pth"
    )

    urls.extend(
        [
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_202010091238.pth",
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_area5_202010091333utc.pth",
        ]
    )

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    attempted = []

    for url in urls:
        attempted.append(url)

        if download_file(url, ckpt_path):
            return

    raise FileNotFoundError(
        "Could not automatically download checkpoint.\n\n"
        "Tried:\n"
        + "\n".join(f"  {u}" for u in attempted)
        + "\n\nManual download command:\n"
        "  mkdir -p weights\n"
        "  wget -O ./weights/randlanet_s3dis_202201071330utc.pth "
        "https://storage.googleapis.com/open3d-releases/model-zoo/"
        "randlanet_s3dis_202201071330utc.pth\n"
    )


def make_features_from_arrays(points, rgb, feature_mode):
    """
    RandLA-Net S3DIS expects 3 feature channels in addition to XYZ.

    In S3DIS, these are RGB values. For LiDAR-only PointCloud2 topics,
    this script synthesizes 3 feature channels.
    """

    if rgb is not None and feature_mode == "auto":
        print("Using RGB from PointCloud2 as features.")
        feat = rgb.astype(np.float32)

        if feat.max() > 1.0:
            feat = feat / 255.0

        return feat

    if feature_mode == "zeros":
        print("Input feature mode: zero RGB features.")
        return np.zeros((points.shape[0], 3), dtype=np.float32)

    print("PointCloud2 has no RGB colors. Using geometric pseudo-features.")

    z = points[:, 2:3]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

    xy_dist = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
    xy_dist = (xy_dist - xy_dist.min()) / (xy_dist.max() - xy_dist.min() + 1e-6)

    ones = np.ones_like(z_norm)

    feat = np.concatenate([z_norm, xy_dist, ones], axis=1).astype(np.float32)

    return feat


def unpack_rgb_value(value):
    """
    Convert packed ROS PointCloud2 rgb/rgba field to normalized RGB.

    Handles both float32-packed and integer-packed RGB.
    """

    if isinstance(value, float) or isinstance(value, np.floating):
        packed = struct.unpack("I", struct.pack("f", float(value)))[0]
    else:
        packed = int(value)

    r = (packed >> 16) & 255
    g = (packed >> 8) & 255
    b = packed & 255

    return [r / 255.0, g / 255.0, b / 255.0]


def pointcloud2_to_xyz_rgb(msg):
    """
    Convert sensor_msgs/msg/PointCloud2 to:
      points: Nx3 float32 XYZ
      rgb:    Nx3 float32 RGB in [0, 1], or None if no rgb/rgba field exists
    """

    from sensor_msgs_py import point_cloud2

    field_names = [field.name for field in msg.fields]

    required = {"x", "y", "z"}
    if not required.issubset(set(field_names)):
        raise RuntimeError(
            f"PointCloud2 topic does not contain x/y/z fields. Found fields: {field_names}"
        )

    rgb_field = None
    if "rgb" in field_names:
        rgb_field = "rgb"
    elif "rgba" in field_names:
        rgb_field = "rgba"

    if rgb_field is None:
        read_fields = ["x", "y", "z"]
    else:
        read_fields = ["x", "y", "z", rgb_field]

    raw_points = list(
        point_cloud2.read_points(
            msg,
            field_names=read_fields,
            skip_nans=True,
        )
    )

    if len(raw_points) == 0:
        return None, None

    if rgb_field is None:
        points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 3)
        rgb = None
    else:
        xyz = []
        colors = []

        for p in raw_points:
            xyz.append([p[0], p[1], p[2]])
            colors.append(unpack_rgb_value(p[3]))

        points = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(colors, dtype=np.float32)

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if rgb is not None:
        rgb = rgb[finite_mask]

    if points.shape[0] == 0:
        return None, None

    return points, rgb


def create_pcd_from_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def save_label_npz(output_pcd_path, points, labels, scores):
    npz_path = os.path.splitext(output_pcd_path)[0] + "_labels.npz"

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
        scores=scores.astype(np.float32),
        class_names=np.array(S3DIS_CLASS_NAMES),
    )

    print(f"Saved label data:\n  {npz_path}")


def print_class_legend(labels=None):
    print("\nS3DIS class/topic legend:")

    if labels is None:
        used_labels = set(range(len(S3DIS_CLASS_NAMES)))
    else:
        used_labels = set(np.asarray(labels).astype(np.int32).tolist())

    for idx, name in enumerate(S3DIS_CLASS_NAMES):
        if idx not in used_labels:
            continue

        color = S3DIS_COLORS[idx]

        print(
            f"  {idx:2d}  {name:10s}  "
            f"RGB=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
        )


def create_colored_point_cloud(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    colors = S3DIS_COLORS[labels % len(S3DIS_COLORS)]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def make_legend_marker(position, color, size):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    sphere.translate(position)
    sphere.paint_uniform_color(color.tolist())
    sphere.compute_vertex_normals()

    return sphere


def visualize_legacy_point_cloud(
    pcd,
    window_name="Open3D Point Cloud",
    point_size=2.0,
    axis_size=1.0,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    vis.add_geometry(pcd)

    if axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        vis.add_geometry(axis)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.asarray([0.02, 0.02, 0.02])

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.0, -1.0, 0.4])
    view_ctl.set_lookat(pcd.get_center())
    view_ctl.set_up([0.0, 0.0, 1.0])
    view_ctl.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


def visualize_point_cloud_with_legend(
    pcd,
    labels=None,
    window_name="Open3D Point Cloud with Class Legend",
    point_size=2.0,
    axis_size=1.0,
):
    """
    Open one Open3D window with:
      segmented cloud + colored legend markers + class names.

    Close the window to continue to the next ROS 2 bag frame.
    """

    if labels is None:
        used_labels = list(range(len(S3DIS_CLASS_NAMES)))
    else:
        used_labels = sorted(set(np.asarray(labels).astype(np.int32).tolist()))

    used_labels = [
        label for label in used_labels
        if 0 <= label < len(S3DIS_CLASS_NAMES)
    ]

    print_class_legend(np.asarray(used_labels, dtype=np.int32))

    if not hasattr(o3d.visualization, "O3DVisualizer"):
        print(
            "\nWARNING: o3d.visualization.O3DVisualizer is not available in this Open3D version."
        )
        print("Falling back to basic visualization. Text labels will only appear in terminal.")
        visualize_legacy_point_cloud(
            pcd,
            window_name=window_name,
            point_size=point_size,
            axis_size=axis_size,
        )
        return

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(window_name, 1280, 720)
    vis.show_settings = True

    pcd_mat = o3d.visualization.rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = point_size

    vis.add_geometry("segmented_point_cloud", pcd, pcd_mat)

    if axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        vis.add_geometry("axis", axis)

    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    extent = bbox.get_extent()

    max_extent = max(float(np.max(extent)), 1.0)
    diagonal = max(float(np.linalg.norm(extent)), 1.0)

    legend_x = max_bound[0] + 0.12 * max_extent
    legend_y = min_bound[1]
    legend_z = max_bound[2]

    spacing = 0.07 * diagonal
    marker_size = 0.012 * diagonal

    if spacing <= 0:
        spacing = 0.1

    if marker_size <= 0:
        marker_size = 0.05

    for row, label in enumerate(used_labels):
        class_name = S3DIS_CLASS_NAMES[label]
        color = S3DIS_COLORS[label]

        marker_position = np.array(
            [
                legend_x,
                legend_y,
                legend_z - row * spacing,
            ],
            dtype=np.float64,
        )

        marker = make_legend_marker(
            position=marker_position,
            color=color,
            size=marker_size,
        )

        marker_mat = o3d.visualization.rendering.MaterialRecord()
        marker_mat.shader = "defaultLit"

        vis.add_geometry(
            f"legend_marker_{label}_{class_name}",
            marker,
            marker_mat,
        )

        text_position = marker_position + np.array(
            [marker_size * 3.0, 0.0, 0.0],
            dtype=np.float64,
        )

        vis.add_3d_label(
            text_position,
            f"{label}: {class_name}",
        )

    legend_min = np.array(
        [
            legend_x,
            legend_y,
            legend_z - max(len(used_labels) - 1, 0) * spacing,
        ],
        dtype=np.float64,
    )

    legend_max = np.array(
        [
            legend_x + 0.5 * max_extent,
            legend_y,
            legend_z,
        ],
        dtype=np.float64,
    )

    full_min = np.minimum(min_bound, legend_min)
    full_max = np.maximum(max_bound, legend_max)

    full_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=full_min,
        max_bound=full_max,
    )

    center = full_bbox.get_center().astype(np.float32)
    extent_full = full_bbox.get_extent()
    radius = max(float(np.linalg.norm(extent_full)), 1.0)

    eye = (
        center
        + np.array(
            [
                0.0,
                -1.8 * radius,
                0.8 * radius,
            ],
            dtype=np.float32,
        )
    )

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vis.setup_camera(
        60.0,
        center,
        eye,
        up,
    )

    app.add_window(vis)
    app.run()


class LiveOpen3DViewer:
    """
    Live viewer for many ROS 2 bag frames.

    Note:
      This uses the legacy Visualizer, so it can update frames smoothly,
      but it cannot draw text labels in the window. The class legend is
      printed in the terminal. Use --visualize_each_frame if you need
      in-window text labels for every frame.
    """

    def __init__(self, point_size=2.0, axis_size=1.0):
        self.point_size = point_size
        self.axis_size = axis_size
        self.vis = None
        self.pcd = None
        self.initialized = False

    def update(self, pcd):
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="ROS 2 Bag Segmented PointCloud2",
                width=1280,
                height=720,
            )

            self.pcd = pcd
            self.vis.add_geometry(self.pcd)

            if self.axis_size > 0:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size)
                self.vis.add_geometry(axis)

            render_opt = self.vis.get_render_option()
            render_opt.point_size = self.point_size
            render_opt.background_color = np.asarray([0.02, 0.02, 0.02])

            self.initialized = True

        else:
            self.pcd.points = pcd.points
            self.pcd.colors = pcd.colors
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None


def create_ros2_bag_reader(bag_path, storage_id):
    import rosbag2_py

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    return reader


def get_bag_topics(reader):
    topics_and_types = reader.get_all_topics_and_types()
    return {topic.name: topic.type for topic in topics_and_types}


def print_bag_topics(reader):
    topics = get_bag_topics(reader)

    print("\nROS 2 bag topics:")
    for name, msg_type in sorted(topics.items()):
        print(f"  {name:40s}  {msg_type}")


def get_pointcloud2_topic_or_raise(reader, requested_topic):
    topics = get_bag_topics(reader)

    pointcloud_topics = [
        name for name, msg_type in topics.items()
        if msg_type == "sensor_msgs/msg/PointCloud2"
    ]

    if requested_topic is not None:
        if requested_topic not in topics:
            raise RuntimeError(
                f"Topic not found in bag: {requested_topic}\n"
                f"Available topics:\n"
                + "\n".join(f"  {name}: {typ}" for name, typ in sorted(topics.items()))
            )

        if topics[requested_topic] != "sensor_msgs/msg/PointCloud2":
            raise RuntimeError(
                f"Requested topic is not sensor_msgs/msg/PointCloud2:\n"
                f"  {requested_topic}: {topics[requested_topic]}"
            )

        return requested_topic

    if len(pointcloud_topics) == 0:
        raise RuntimeError(
            "No sensor_msgs/msg/PointCloud2 topic found in the bag.\n"
            "Use --list_topics to inspect available topics."
        )

    if len(pointcloud_topics) > 1:
        raise RuntimeError(
            "Multiple PointCloud2 topics found. Please specify one with --topic:\n"
            + "\n".join(f"  {topic}" for topic in pointcloud_topics)
        )

    return pointcloud_topics[0]


def segment_points_with_pipeline(pipeline, points, rgb, feature_mode):
    feat = make_features_from_arrays(points, rgb, feature_mode)

    if feat.shape[0] != points.shape[0] or feat.shape[1] != 3:
        raise RuntimeError(
            f"Feature shape must be Nx3. Got {feat.shape}, points {points.shape}."
        )

    data = {
        "point": points.astype(np.float32),
        "feat": feat.astype(np.float32),
        "label": np.zeros((points.shape[0],), dtype=np.int32),
    }

    result = pipeline.run_inference(data)

    pred_labels = result["predict_labels"].astype(np.int32)
    pred_scores = result["predict_scores"].astype(np.float32)

    return pred_labels, pred_scores


def save_segmented_frame(output_dir, frame_index, timestamp_ns, points, labels, scores):
    os.makedirs(output_dir, exist_ok=True)

    pcd = create_colored_point_cloud(points, labels)

    pcd_path = os.path.join(
        output_dir,
        f"segmented_frame_{frame_index:06d}_{timestamp_ns}.pcd",
    )

    ok = o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=False)

    if not ok:
        raise RuntimeError(f"Failed to write output point cloud: {pcd_path}")

    save_label_npz(pcd_path, points, labels, scores)

    print(f"Saved segmented frame:\n  {pcd_path}")


def print_predicted_counts(labels):
    print("\nPredicted class counts:")
    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if 0 <= label < len(S3DIS_CLASS_NAMES):
            class_name = S3DIS_CLASS_NAMES[label]
        else:
            class_name = "unknown"

        print(f"  {label:2d}  {class_name:10s}  {count}")


def main():
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError("--frame_step must be >= 1")

    # ROS 2 imports are kept here so the script still prints a clean error
    # if it is not run inside a sourced ROS 2 environment.
    try:
        import rosbag2_py  # noqa: F401
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as e:
        raise RuntimeError(
            "Failed to import ROS 2 Python modules.\n"
            "Run this script from a sourced ROS 2 environment, for example:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "or source your workspace install/setup.bash\n\n"
            f"Original error: {e}"
        )

    reader = create_ros2_bag_reader(args.bag, args.storage_id)

    if args.list_topics:
        print_bag_topics(reader)
        return

    topic_name = get_pointcloud2_topic_or_raise(reader, args.topic)

    print(f"Using PointCloud2 topic:\n  {topic_name}")

    # Import torch backend only after argument parsing.
    import open3d.ml.torch as ml3d

    if args.cfg_file is None:
        args.cfg_file = os.path.join(
            args.open3d_ml_root,
            "ml3d",
            "configs",
            "randlanet_s3dis.yml",
        )

    if not os.path.isfile(args.cfg_file):
        raise FileNotFoundError(f"Config file not found:\n  {args.cfg_file}")

    if not args.no_auto_download:
        auto_download_checkpoint(args.open3d_ml_root, args.ckpt)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {args.ckpt}\n"
            "Use --ckpt or allow automatic download."
        )

    verify_md5_if_known(args.ckpt)

    print("Loading config...")
    cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

    print("Creating RandLA-Net model...")
    cfg.model.num_points = args.num_points
    model = ml3d.models.RandLANet(**cfg.model)

    print("Creating semantic segmentation pipeline on CPU...")
    pipeline = ml3d.pipelines.SemanticSegmentation(
        model=model,
        dataset=None,
        device="cpu",
        batch_size=1,
        test_batch_size=1,
        val_batch_size=1,
    )

    print("Loading checkpoint...")
    pipeline.load_ckpt(ckpt_path=args.ckpt)

    topics = get_bag_topics(reader)
    msg_type = get_message(topics[topic_name])

    live_viewer = None

    if args.visualize and not args.visualize_each_frame:
        live_viewer = LiveOpen3DViewer(
            point_size=args.point_size,
            axis_size=args.axis_size,
        )

    seen_cloud_frames = 0
    processed_frames = 0

    try:
        while reader.has_next():
            topic, serialized_data, timestamp_ns = reader.read_next()

            if topic != topic_name:
                continue

            current_frame_index = seen_cloud_frames
            seen_cloud_frames += 1

            if current_frame_index < args.start_frame:
                continue

            if (current_frame_index - args.start_frame) % args.frame_step != 0:
                continue

            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            msg = deserialize_message(serialized_data, msg_type)

            print(
                f"\nProcessing frame {current_frame_index} "
                f"(timestamp_ns={timestamp_ns})"
            )

            points, rgb = pointcloud2_to_xyz_rgb(msg)

            if points is None or points.shape[0] == 0:
                print("Skipping empty PointCloud2 frame.")
                continue

            pcd = create_pcd_from_points(points, rgb)

            if args.voxel_size > 0:
                print(f"Voxel downsampling with voxel_size={args.voxel_size}")
                pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
                points = np.asarray(pcd.points).astype(np.float32)

                if pcd.has_colors():
                    rgb = np.asarray(pcd.colors).astype(np.float32)
                else:
                    rgb = None

            print(f"Running CPU inference on {points.shape[0]} points...")
            pred_labels, pred_scores = segment_points_with_pipeline(
                pipeline=pipeline,
                points=points,
                rgb=rgb,
                feature_mode=args.feature_mode,
            )

            print_predicted_counts(pred_labels)
            print_class_legend(pred_labels)

            segmented_pcd = create_colored_point_cloud(points, pred_labels)

            if args.output_dir is not None:
                save_segmented_frame(
                    output_dir=args.output_dir,
                    frame_index=current_frame_index,
                    timestamp_ns=timestamp_ns,
                    points=points,
                    labels=pred_labels,
                    scores=pred_scores,
                )

            if args.visualize_each_frame:
                visualize_point_cloud_with_legend(
                    segmented_pcd,
                    labels=pred_labels,
                    window_name=f"Segmented ROS 2 bag frame {current_frame_index}",
                    point_size=args.point_size,
                    axis_size=args.axis_size,
                )
            elif live_viewer is not None:
                live_viewer.update(segmented_pcd)

            processed_frames += 1

    finally:
        if live_viewer is not None:
            live_viewer.close()

    print(
        f"\nDone. Seen PointCloud2 frames: {seen_cloud_frames}. "
        f"Processed frames: {processed_frames}."
    )


if __name__ == "__main__":
    main()